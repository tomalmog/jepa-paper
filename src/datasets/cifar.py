"""CIFAR-10 / CIFAR-100 dataset wrapper for Hyperbolic-JEPA pretraining."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR_STD = (0.2673, 0.2564, 0.2762)


def make_pretrain_transform(img_size: int = 32) -> transforms.Compose:
    # I-JEPA uses random resized crop + horizontal flip + color jitter
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])


def make_eval_transform(img_size: int = 32) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])


def _pin_memory_default() -> bool:
    """pin_memory only helps on CUDA; on MPS/CPU it's a no-op (and on some
    Mac setups has caused warnings), so disable it unless CUDA is present.
    """
    return bool(torch.cuda.is_available())


def build_pretrain_loader(
    root: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    collate_fn,
) -> Tuple[DataLoader, int]:
    root = str(Path(root).expanduser())
    transform = make_pretrain_transform(img_size)
    if dataset.lower() == "cifar10":
        ds = datasets.CIFAR10(root, train=True, download=False, transform=transform)
        n_classes = 10
    elif dataset.lower() == "cifar100":
        ds = datasets.CIFAR100(root, train=True, download=False, transform=transform)
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory_default(),
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )
    return loader, n_classes


def build_eval_loaders(
    root: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
):
    root = str(Path(root).expanduser())
    tfm = make_eval_transform(img_size)
    if dataset.lower() == "cifar10":
        train = datasets.CIFAR10(root, train=True, download=False, transform=tfm)
        test = datasets.CIFAR10(root, train=False, download=False, transform=tfm)
        n_classes = 10
    else:
        train = datasets.CIFAR100(root, train=True, download=False, transform=tfm)
        test = datasets.CIFAR100(root, train=False, download=False, transform=tfm)
        n_classes = 100
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=_pin_memory_default(),
              persistent_workers=num_workers > 0)
    return DataLoader(train, shuffle=False, **kw), DataLoader(test, shuffle=False, **kw), n_classes
