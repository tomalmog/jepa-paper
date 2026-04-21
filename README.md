# Hyperbolic-JEPA

A faithful port of **I-JEPA** ([Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243))
where the representation space is moved from Euclidean R^d onto the
**Poincaré ball** so hierarchical structure in the data is captured by the
geometry of the embedding space itself.

The architecture (ViT context encoder, EMA target encoder, ViT predictor,
multi-block masking) and training recipe are unchanged relative to the
reference [`facebookresearch/ijepa`](https://github.com/facebookresearch/ijepa).
The single substantive modification is the **loss**:

```text
I-JEPA        :  loss = smooth_L1( pred , target )         # Euclidean
Hyperbolic-JEPA:  loss = d_Poincaré( exp_0(pred) , exp_0(target) )^p
```

Everything else — masking, EMA, AdamW, schedulers — is unchanged.
Network parameters remain Euclidean, so no Riemannian optimizer is needed
(this matches the recipe used by Khrulkov et al. 2020, *Hyperbolic Image
Embeddings*, CVPR).

---

## Why hyperbolic?

- The volume of a ball of radius *r* in hyperbolic space grows **exponentially**
  in *r*, while in Euclidean space it grows polynomially.  That makes hyperbolic
  space a natural host for **trees / hierarchies** with bounded distortion.
- I-JEPA's latent predictor task is itself a natural fit: predicting a target
  embedding given a context is essentially a *hierarchical completion* task
  (context → more-specific target), which hyperbolic distance penalises
  consistently across scales.

References if you want the full picture:

- Ganea, Becigneul, Hofmann. *Hyperbolic Neural Networks.* NeurIPS 2018.
- Khrulkov et al. *Hyperbolic Image Embeddings.* CVPR 2020.
- Guo, Wang, Tang, Yeung. *Free Hyperbolic Neural Networks with Limited Radius.*
  ICML 2022. *(source of the feature-clipping trick used here for stability.)*
- Nickel & Kiela. *Poincaré Embeddings.* NeurIPS 2017.

---

## Repository layout

```
HyperbolicJEPA/
├── configs/
│   ├── hyperbolic_ijepa_cifar10.yaml
│   └── hyperbolic_ijepa_cifar100.yaml
├── src/
│   ├── models/
│   │   ├── vision_transformer.py       # ViT + I-JEPA predictor
│   │   └── hyperbolic.py               # Poincaré-ball ops + loss  (NOVEL)
│   ├── masks/
│   │   └── multiblock.py               # I-JEPA multi-block masking
│   ├── datasets/cifar.py
│   ├── utils/{schedulers.py,tensors.py}
│   ├── helper.py                       # model / optimiser factories
│   └── train.py
├── main.py
├── eval_linear_probe.py
└── requirements.txt
```

---

## Quick start (single GPU / Colab)

```bash
pip install -r requirements.txt

# Pretrain Hyperbolic-JEPA on CIFAR-100 with a ViT-Tiny (~6M params)
python main.py --config configs/hyperbolic_ijepa_cifar100.yaml

# Linear probe the EMA target encoder
python eval_linear_probe.py \
    --checkpoint runs/hjepa_cifar100_vit_tiny/last.pt \
    --use-tangent          # probe in the log-map (tangent) coordinates
```

Ballpark cost of a run:

| Dataset     | Encoder   | Epochs | Hardware         | Wall-clock |
|-------------|-----------|--------|------------------|------------|
| CIFAR-10    | ViT-Tiny  | 200    | 1× T4 (Colab)    | ~3 h       |
| CIFAR-100   | ViT-Tiny  | 300    | 1× A100 (Runpod) | ~1 h       |
| CIFAR-100   | ViT-Small | 300    | 1× A100          | ~3 h       |

---

## Key knobs for research

All hyperbolic-specific knobs live in the config under `curvature`, `clip_radius`,
`loss_power`, and in `src/models/hyperbolic.py`.  Interesting things to sweep:

| Knob          | Effect                                                              |
|---------------|---------------------------------------------------------------------|
| `curvature`   | `c=1` is standard; `c→0` recovers Euclidean I-JEPA.                 |
| `clip_radius` | Guo et al. feature-clipping.  Smaller = safer, larger = more range. |
| `loss_power`  | 2 ≈ MSE, 1 ≈ smooth-L1.                                             |
| `predictor_*` | Whether the predictor should itself use Möbius ops (not yet wired). |

---

## Where this differs from vanilla I-JEPA

1. `src/models/hyperbolic.py` (**new file**) — Poincaré-ball primitives and
   `PoincareRegressionLoss`.
2. `src/train.py` — calls `PoincareRegressionLoss` instead of smooth-L1.
   The rest of the training loop (EMA, masking, AdamW + warmup-cosine,
   layer-norm of targets) is byte-for-byte the same spirit as I-JEPA.
3. `eval_linear_probe.py` — can optionally read features in tangent space
   (`--use-tangent`) for a fair probe against Euclidean baselines.

## Reproducing a Euclidean baseline

Set `curvature: 1e-6` in the config (or pass `loss_power: 2, curvature: small`).
As `c → 0` the Poincaré-distance loss collapses to the Euclidean L2 loss,
so you recover vanilla I-JEPA with the same code path and the same hyper-params.

---

## License

Research code released under the same license as the upstream I-JEPA code
(Attribution-NonCommercial 4.0).  The Poincaré-ball code in `hyperbolic.py`
is adapted from several standard implementations and re-derived from the
references above.
