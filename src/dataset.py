"""
src/dataset.py
==============
Handles all data loading, preprocessing, and augmentation for the
Pneumonia Classifier project.

Key responsibilities:
  - Load images from chest_xray/train, val, test folders
  - Convert Grayscale (L) → RGB (required for pretrained models)
  - Apply augmentation on train split only
  - Fix tiny val set by re-splitting from train (90/10)
  - Handle class imbalance with WeightedRandomSampler
  - Return ready-to-use PyTorch DataLoaders
"""

import logging
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


# =============================================================================
# Transform Builders
# =============================================================================

def convert_to_rgb(img: Image.Image) -> Image.Image:
    """Named function to avoid PicklingError on Windows multiprocessing."""
    return img.convert("RGB")

def build_train_transforms(image_size: int) -> transforms.Compose:
    """
    Build the augmentation pipeline for the TRAINING set.

    Augmentations chosen for chest X-rays:
      - Horizontal flip: patients can be imaged either way
      - Small rotation: mimic slight patient tilt (not too much — anatomy matters)
      - Color jitter: simulate different X-ray machine exposures
      - Affine translate: minor positional shifts

    Grayscale → RGB conversion is the FIRST step because all images in this
    dataset are single-channel (mode='L'), but pretrained ImageNet models
    expect 3-channel (RGB) input.
    """
    return transforms.Compose([
        # Step 1: Convert single-channel grayscale to 3-channel RGB
        transforms.Lambda(convert_to_rgb),

        # Step 2: Resize to model input size (224x224 is ImageNet standard)
        transforms.Resize((image_size, image_size)),

        # Step 3: Augmentation — only applied during training
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),

        # Step 4: Convert PIL image to PyTorch tensor [0, 1]
        transforms.ToTensor(),

        # Step 5: Normalize using ImageNet mean/std (required for pretrained models)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def build_eval_transforms(image_size: int) -> transforms.Compose:
    """
    Build the transform pipeline for VALIDATION and TEST sets.
    No augmentation — only the mandatory preprocessing steps.
    """
    return transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# =============================================================================
# Class-Imbalance Sampler
# =============================================================================

def build_weighted_sampler(subset: torch.utils.data.Subset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to fix class imbalance.

    Strategy: assign each sample a weight = 1 / (count of its class).
    Minority class (NORMAL) gets higher weight → sampled more often.
    """
    # Get labels for just the items in this subset
    labels = [subset.dataset.targets[i] for i in subset.indices]
    
    # Count images per class (assuming binary 0 and 1)
    class_counts = [0, 0]
    for label in labels:
        class_counts[label] += 1

    logger.info("Train subset class counts: NORMAL=%d, PNEUMONIA=%d", class_counts[0], class_counts[1])

    # Weight for each class: rare class gets higher weight
    class_weights = [1.0 / count for count in class_counts]

    # Assign the weight of its class to each sample in the subset
    sample_weights = [class_weights[label] for label in labels]

    # Sampler draws len(subset) samples with replacement
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(subset),
        replacement=True
    )
    return sampler


# =============================================================================
# Main DataLoader Builder
# =============================================================================

def get_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build and return DataLoaders for train, validation, and test splits.

    The original val/ folder has only 16 images — far too small for reliable
    validation. Instead, we:
      1. Load the entire train/ folder
      2. Split it 90% train / 10% val using random_split

    Args:
        data_dir:    Root path of the dataset (e.g. "chest_xray/")
        image_size:  Target H×W for all images (default 224)
        batch_size:  Number of images per batch
        val_split:   Fraction of train data to use as validation
        num_workers: Parallel CPU workers for data loading
        random_seed: For reproducible splits

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    test_dir  = data_path / "test"

    # ── Load full training dataset with train transforms ──────────────────
    # We apply train transforms initially; val subset will use eval transforms
    full_train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=build_train_transforms(image_size),
        loader=_pil_loader,
    )
    class_names = full_train_dataset.classes  # ['NORMAL', 'PNEUMONIA']
    logger.info("Classes found: %s", class_names)
    logger.info("Total training images (before split): %d", len(full_train_dataset))

    # ── Split into train / val ─────────────────────────────────────────────
    torch.manual_seed(random_seed)
    val_size   = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    logger.info("Train: %d  |  Val: %d", train_size, val_size)

    # ── Override val transforms (no augmentation) ──────────────────────────
    # random_split returns a Subset that shares the parent dataset's transform.
    # We wrap val_subset with a custom dataset to apply eval_transforms instead.
    val_dataset = _TransformSubset(
        subset=val_subset,
        transform=build_eval_transforms(image_size),
    )

    # ── Build WeightedSampler for training (fixes class imbalance) ─────────
    sampler = build_weighted_sampler(train_subset)

    # ── Train DataLoader — uses sampler (no shuffle, sampler handles it) ───
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,    # Faster GPU transfer
    )

    # ── Val DataLoader — no sampler, no shuffle ────────────────────────────
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── Test DataLoader ────────────────────────────────────────────────────
    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=build_eval_transforms(image_size),
        loader=_pil_loader,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    logger.info("Test images: %d", len(test_dataset))

    return train_loader, val_loader, test_loader, class_names


# =============================================================================
# Helper: Custom PIL Loader (handles grayscale gracefully)
# =============================================================================

def _pil_loader(path: str) -> Image.Image:
    """
    Custom image loader that opens the image without converting to RGB yet.
    Conversion happens inside the transform pipeline so it's logged/controlled.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()  # Force load before file closes
    return img


# =============================================================================
# Helper: Subset with independent transform
# =============================================================================

class _TransformSubset(torch.utils.data.Dataset):
    """
    Wraps a Subset and applies a DIFFERENT transform than the parent dataset.
    Used to apply eval_transforms to the validation subset without affecting
    the train subset (they share the same parent ImageFolder).
    """

    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get raw image and label from the parent subset
        img, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = _pil_loader(img)      # Load PIL image
        img = self.transform(img)   # Apply eval transform
        return img, label
