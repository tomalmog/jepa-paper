"""
Tiny ImageNet-200 and ImageNet-100 dataset wrappers for Hyperbolic-JEPA.

Tiny ImageNet: 200 classes, 100K train images, 10K val images, 64×64.
ImageNet-100: 100-class subset of ImageNet-1K, 224×224.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Tiny ImageNet uses ImageNet stats
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _pin_memory_default() -> bool:
    return bool(torch.cuda.is_available())


def _fix_tiny_imagenet_val(root: str) -> None:
    """Reorganize Tiny ImageNet val directory into class subdirectories.

    Tiny ImageNet val comes as flat images + val_annotations.txt.
    torchvision.datasets.ImageFolder needs class subdirectories.
    """
    val_dir = os.path.join(root, "val")
    annot_file = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.exists(annot_file):
        return  # Already reorganized

    # Read annotations
    with open(annot_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, cls = parts[0], parts[1]
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(val_dir, "images", fname)
            dst = os.path.join(cls_dir, fname)
            try:
                if os.path.exists(src) and not os.path.exists(dst):
                    os.rename(src, dst)
            except (FileNotFoundError, OSError):
                pass  # Another process already moved this file

    # Clean up
    images_dir = os.path.join(val_dir, "images")
    if os.path.exists(images_dir):
        try:
            os.rmdir(images_dir)
        except OSError:
            pass  # Not empty, some files remain
    # Don't remove annotations file — it's harmless and useful for debugging


def make_pretrain_transform_imagenet(img_size: int = 64) -> transforms.Compose:
    """SSL pretrain augmentation for ImageNet-scale data."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def make_eval_transform_imagenet(img_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14), antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def build_pretrain_loader_imagenet(
    root: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    collate_fn,
) -> Tuple[DataLoader, int]:
    root = str(Path(root).expanduser())

    if dataset.lower() == "tiny_imagenet":
        data_dir = os.path.join(root, "tiny-imagenet-200")
        train_dir = os.path.join(data_dir, "train")
        n_classes = 200
    elif dataset.lower() == "imagenet100":
        data_dir = os.path.join(root, "imagenet-100")
        train_dir = os.path.join(data_dir, "train")
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    transform = make_pretrain_transform_imagenet(img_size)
    ds = datasets.ImageFolder(train_dir, transform=transform)
    print(f"Dataset: {dataset}, samples: {len(ds)}, classes: {n_classes}")

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


def build_eval_loaders_imagenet(
    root: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
):
    root = str(Path(root).expanduser())

    if dataset.lower() == "tiny_imagenet":
        data_dir = os.path.join(root, "tiny-imagenet-200")
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        _fix_tiny_imagenet_val(data_dir)
        n_classes = 200
    elif dataset.lower() == "imagenet100":
        data_dir = os.path.join(root, "imagenet-100")
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    tfm = make_eval_transform_imagenet(img_size)
    train = datasets.ImageFolder(train_dir, transform=tfm)
    test = datasets.ImageFolder(val_dir, transform=tfm)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=_pin_memory_default(),
              persistent_workers=num_workers > 0)
    return DataLoader(train, shuffle=False, **kw), DataLoader(test, shuffle=False, **kw), n_classes
