"""
src/model.py
============
Defines all model architectures for the Pneumonia Classifier project.

Supported models (in complexity order):
  1. baseline_cnn    — Simple CNN built from scratch (establishes baseline)
  2. vgg16           — Transfer learning with VGG-16
  3. resnet50        — Transfer learning with ResNet-50
  4. efficientnet_b3 — Transfer learning with EfficientNet-B3 (recommended ✅)
  5. densenet121     — Transfer learning with DenseNet-121 (CheXNet architecture)

Usage:
    from src.model import build_model
    model = build_model(architecture="efficientnet_b3", pretrained=True, dropout=0.3)
"""

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


# =============================================================================
# Model Factory — single entry point
# =============================================================================

def build_model(
    architecture: str = "efficientnet_b3",
    pretrained: bool = True,
    dropout: float = 0.3,
    num_classes: int = 1,
) -> nn.Module:
    """
    Factory function: returns the requested model with a custom classifier head.

    All transfer-learning models follow the same pattern:
      1. Load pretrained backbone (ImageNet weights)
      2. FREEZE all backbone parameters
      3. Replace the final classifier with a new head for binary classification
         Output: 1 logit (used with BCEWithLogitsLoss — no sigmoid needed here)

    Args:
        architecture: Model name string (see supported list above)
        pretrained:   Load ImageNet weights if True
        dropout:      Dropout probability on the classifier head
        num_classes:  Number of output neurons (1 for binary classification)

    Returns:
        nn.Module ready for training
    """
    architecture = architecture.lower()
    builders = {
        "baseline_cnn":    _build_baseline_cnn,
        "vgg16":           _build_vgg16,
        "resnet50":        _build_resnet50,
        "efficientnet_b3": _build_efficientnet_b3,
        "densenet121":     _build_densenet121,
    }

    if architecture not in builders:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(builders.keys())}"
        )

    model = builders[architecture](pretrained=pretrained, dropout=dropout,
                                   num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | Total params: %s | Trainable: %s",
                architecture, f"{n_params:,}", f"{n_trainable:,}")
    return model


# =============================================================================
# 1. Baseline CNN — built from scratch
# =============================================================================

class _BaselineCNN(nn.Module):
    """
    Simple 3-block CNN to establish a performance baseline.

    Architecture:
      [Conv → BN → ReLU → MaxPool] × 3
      → GlobalAvgPool → Dropout → Linear(1)

    No pretrained weights. This shows what's achievable without transfer learning.
    Expected accuracy: ~85-88%.
    """

    def __init__(self, dropout: float = 0.5, num_classes: int = 1):
        super().__init__()

        # Feature extractor: 3 convolutional blocks
        self.features = nn.Sequential(
            # Block 1: input (3, 224, 224) → output (32, 112, 112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (32, 112, 112) → (64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (64, 56, 56) → (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Collapse spatial dimensions to a single vector per image
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),  # 128 channels → 1 logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)              # Extract spatial features
        x = self.global_avg_pool(x)       # (B, 128, 1, 1)
        x = torch.flatten(x, 1)           # (B, 128)
        x = self.classifier(x)            # (B, 1)
        return x


def _build_baseline_cnn(pretrained: bool, dropout: float, num_classes: int) -> nn.Module:
    # pretrained=True has no effect here (no pretrained baseline CNN)
    return _BaselineCNN(dropout=dropout, num_classes=num_classes)


# =============================================================================
# 2. VGG-16
# =============================================================================

def _build_vgg16(pretrained: bool, dropout: float, num_classes: int) -> nn.Module:
    """
    VGG-16 with frozen backbone and a lightweight binary classifier head.
    Simple and interpretable — good as a first transfer learning reference.
    Expected accuracy: ~91-92%.
    """
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)

    # Freeze all backbone parameters (features + avgpool)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the heavy VGG classifier with a lightweight binary head
    in_features = model.classifier[0].in_features  # 25088
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return model


# =============================================================================
# 3. ResNet-50
# =============================================================================

def _build_resnet50(pretrained: bool, dropout: float, num_classes: int) -> nn.Module:
    """
    ResNet-50 with frozen backbone.
    Residual (skip) connections help avoid vanishing gradients.
    Expected accuracy: ~93-94%.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze the entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully-connected layer
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# =============================================================================
# 4. EfficientNet-B3 ✅ (Recommended)
# =============================================================================

def _build_efficientnet_b3(pretrained: bool, dropout: float, num_classes: int) -> nn.Module:
    """
    EfficientNet-B3 — best accuracy-to-parameter ratio among all options.
    Compound scaling (depth + width + resolution) gives excellent performance
    at only 12M parameters vs VGG's 138M.
    Expected accuracy after fine-tuning: ~96-97%.
    """
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    # Freeze all backbone parameters initially (Phase: feature extraction)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head for binary output
    in_features = model.classifier[1].in_features  # 1536
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# =============================================================================
# 5. DenseNet-121
# =============================================================================

def _build_densenet121(pretrained: bool, dropout: float, num_classes: int) -> nn.Module:
    """
    DenseNet-121 — used in CheXNet (Stanford's chest X-ray SOTA paper).
    Dense connections (each layer receives input from ALL previous layers)
    allow efficient feature reuse. Great for medical imaging.
    Expected accuracy: ~94-95%.
    """
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# =============================================================================
# Fine-tuning Helper — unfreeze specific backbone blocks
# =============================================================================

def unfreeze_blocks(model: nn.Module, block_names: list[str]) -> None:
    """
    Selectively unfreeze backbone layers for fine-tuning (Phase 3 → Step 3).

    After initial transfer learning (frozen backbone), we unfreeze the LAST
    few blocks so they can adapt to chest X-ray features. We keep early layers
    frozen because they already learned general features (edges, textures).

    Args:
        model:       The pretrained model
        block_names: List of parameter name substrings to unfreeze
                     e.g. ["features.6", "features.7", "features.8"] for EfficientNet

    Example:
        unfreeze_blocks(model, ["features.7", "features.8"])
    """
    unfrozen = 0
    for name, param in model.named_parameters():
        if any(block in name for block in block_names):
            param.requires_grad = True
            unfrozen += 1

    logger.info("Unfrozen %d parameter tensors for fine-tuning: %s", unfrozen, block_names)


def get_optimizer_param_groups(
    model: nn.Module,
    head_lr: float = 1e-4,
    backbone_lr: float = 1e-5,
) -> list[dict]:
    """
    Build differential learning rate parameter groups for fine-tuning.

    Fine-tuning best practice:
      - Backbone (pretrained layers) → very small LR (don't destroy learned features)
      - Classifier head (random init) → larger LR (needs to learn quickly)

    Args:
        model:       Fine-tuning model with some unfrozen backbone layers
        head_lr:     LR for classifier head parameters
        backbone_lr: LR for unfrozen backbone parameters

    Returns:
        List of param groups to pass directly to an optimizer
    """
    head_params     = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": head_params,     "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]
