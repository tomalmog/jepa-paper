"""
Multi-block masking for I-JEPA.

Port of ``src.masks.multiblock.MaskCollator`` from facebookresearch/ijepa,
simplified for small images.  We sample:

    * ``num_targets`` target blocks (rectangular regions).
    * A single context block (default: the full image), from which the
      union of target patches is removed.

The collator returns two lists of LongTensors:

    masks_enc[k]  : (B, N_ctx)  — indices of patches fed to the context
                                  encoder for example k.
    masks_pred[k] : (B, N_tgt)  — indices of patches each predicted block
                                  should recover.

For a single GPU run we keep ``num_targets=4`` (paper default).
"""
from __future__ import annotations

import math
from multiprocessing import Value
from typing import List, Tuple

import torch


# -----------------------------------------------------------------------------
# apply_masks is used by the ViT encoders
# -----------------------------------------------------------------------------
def apply_masks(x: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    """Gather patch tokens according to a list of index tensors.

    x:      (B, N, D)
    masks:  list of (B, N_k)  long tensors.  All masks are concatenated
            along the batch dimension, so the output is
            ``(B * len(masks), N_k, D)``.

    I-JEPA uses this with a single context mask -> ``(B, N_ctx, D)``.
    """
    all_x = []
    for m in masks:
        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        all_x.append(torch.gather(x, 1, idx))
    return torch.cat(all_x, dim=0)


# -----------------------------------------------------------------------------
# Mask collator
# -----------------------------------------------------------------------------
class MultiBlockMaskCollator:
    """Callable used as DataLoader ``collate_fn``."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 4,
        enc_mask_scale: Tuple[float, float] = (0.85, 1.0),
        pred_mask_scale: Tuple[float, float] = (0.15, 0.2),
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        num_enc_masks: int = 1,
        num_pred_masks: int = 4,
        min_keep: int = 4,
        allow_overlap: bool = False,
    ):
        self.patch_size = patch_size
        self.height = input_size // patch_size
        self.width = input_size // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.num_enc_masks = num_enc_masks
        self.num_pred_masks = num_pred_masks
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value("i", -1)  # deterministic per-epoch seeds

    def step(self) -> int:
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    # ----- geometry helpers ---------------------------------------------------
    def _sample_block_size(self, scale_range, g: torch.Generator) -> Tuple[int, int]:
        s_min, s_max = scale_range
        ar_min, ar_max = self.aspect_ratio
        _rand = torch.rand(1, generator=g).item()
        scale = s_min + _rand * (s_max - s_min)
        max_keep = int(self.height * self.width * scale)
        _rand = torch.rand(1, generator=g).item()
        ar = math.log(ar_min) + _rand * (math.log(ar_max) - math.log(ar_min))
        ar = math.exp(ar)
        h = int(round(math.sqrt(max_keep * ar)))
        w = int(round(math.sqrt(max_keep / ar)))
        h = max(1, min(self.height, h))
        w = max(1, min(self.width, w))
        return h, w

    def _sample_block_mask(self, b_size, acceptable_regions=None,
                           g: torch.Generator = None):
        """Return (indices_in_block, complement_of_block)."""
        h, w = b_size
        for _ in range(20):
            top = torch.randint(0, self.height - h + 1, (1,), generator=g).item()
            left = torch.randint(0, self.width - w + 1, (1,), generator=g).item()
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top + h, left:left + w] = 1

            if acceptable_regions is not None:
                # enforce that the target block lies inside an acceptable region
                allowed = acceptable_regions
                if (mask * allowed).sum() < self.min_keep:
                    continue

            mask = mask.flatten()
            mask_inv = 1 - mask
            if mask.sum() >= self.min_keep:
                break
        idx = torch.nonzero(mask).flatten()
        idx_complement = torch.nonzero(mask_inv).flatten()
        return idx, idx_complement

    # ----- callable -----------------------------------------------------------
    def __call__(self, batch):
        """Return (images, masks_enc, masks_pred)."""
        B = len(batch)
        collated = torch.stack([b[0] for b in batch], dim=0)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        # Sample a common *size* for all predictor masks in a minibatch
        pred_size = self._sample_block_size(self.pred_mask_scale, g)
        enc_size = self._sample_block_size(self.enc_mask_scale, g)

        masks_pred_all, masks_enc_all = [], []

        # Per-example sampling (needed because different minibatch entries
        # can get different valid positions).
        for _ in range(B):
            # 1) pick predictor target indices
            pred_idx_list = []
            tgt_union_2d = torch.ones((self.height, self.width), dtype=torch.int32)
            for _ in range(self.num_pred_masks):
                idx, _ = self._sample_block_mask(pred_size, g=g)
                pred_idx_list.append(idx)
                tgt_union_2d.view(-1)[idx] = 0            # mark as "not acceptable"

            # 2) pick context indices (must avoid target union if !allow_overlap)
            if self.allow_overlap:
                enc_idx, _ = self._sample_block_mask(enc_size, g=g)
            else:
                enc_idx, _ = self._sample_block_mask(
                    enc_size, acceptable_regions=tgt_union_2d, g=g
                )

            masks_pred_all.append(pred_idx_list)
            masks_enc_all.append([enc_idx])

        # Truncate to a common length within each of (pred_k / enc) so we
        # can stack into tensors.
        masks_enc = self._collate(masks_enc_all, n_masks=self.num_enc_masks)
        masks_pred = self._collate(masks_pred_all, n_masks=self.num_pred_masks)

        return collated, masks_enc, masks_pred

    @staticmethod
    def _collate(per_sample_mask_lists: List[List[torch.Tensor]],
                 n_masks: int) -> List[torch.Tensor]:
        """Stack a list of ``[B][n_masks]`` index tensors into ``n_masks``
        tensors of shape ``(B, N_k)``.

        We truncate every block to the minimum length across the batch so
        indices stack cleanly (I-JEPA does the same).
        """
        stacked = []
        for k in range(n_masks):
            col = [sample[k] for sample in per_sample_mask_lists]
            min_len = min(t.numel() for t in col)
            col = [t[:min_len] for t in col]
            stacked.append(torch.stack(col, dim=0).long())
        return stacked
