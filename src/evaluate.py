"""
src/evaluate.py
===============
Comprehensive model evaluation with all medical-grade metrics and visualizations.

Generates:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix (normalized)
  - ROC curve + AUC score
  - Precision-Recall curve
  - Training history plots (loss & accuracy curves)
  - All plots saved to logs/plots/

Usage:
    from src.evaluate import evaluate_model
    evaluate_model(model, test_loader, device, save_dir="logs/plots")
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)

# Use a clean, non-interactive backend for saving plots
plt.style.use("seaborn-v0_8-darkgrid")


# =============================================================================
# Run Full Evaluation on Test Set
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str] = ["NORMAL", "PNEUMONIA"],
    save_dir: str = "logs/plots",
    threshold: float = 0.5,
) -> dict:
    """
    Run full evaluation on a DataLoader (typically test set).

    Args:
        model:       Trained PyTorch model
        loader:      DataLoader for test/val set
        device:      torch.device (cuda / cpu)
        class_names: Label names for display
        save_dir:    Directory to save all plots
        threshold:   Decision threshold (default 0.5; lower = more sensitive)

    Returns:
        dict of all computed metrics
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model.eval()  # Switch to inference mode

    all_probs  = []   # Raw sigmoid probabilities (for ROC/PR curves)
    all_preds  = []   # Binary predictions at threshold
    all_labels = []   # Ground truth labels

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)                          # Raw logits
        probs  = torch.sigmoid(logits).cpu().squeeze()  # Convert to probabilities [0,1]
        preds  = (probs >= threshold).long()            # Apply decision threshold

        all_probs.extend(probs.tolist() if probs.dim() > 0 else [probs.item()])
        all_preds.extend(preds.tolist() if preds.dim() > 0 else [preds.item()])
        all_labels.extend(labels.tolist())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Compute all metrics ───────────────────────────────────────────────────
    accuracy  = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    auc_roc   = roc_auc_score(all_labels, all_probs)
    avg_prec  = average_precision_score(all_labels, all_probs)

    metrics = {
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,          # Most important for medical AI
        "f1":               f1,
        "auc_roc":          auc_roc,
        "avg_precision":    avg_prec,
    }

    # ── Print classification report ───────────────────────────────────────────
    logger.info("\n%s", "=" * 55)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 55)
    logger.info("Accuracy  : %.4f", accuracy)
    logger.info("Precision : %.4f", precision)
    logger.info("Recall    : %.4f  ← (most critical for medical AI)", recall)
    logger.info("F1-Score  : %.4f", f1)
    logger.info("AUC-ROC   : %.4f", auc_roc)
    logger.info("\n%s", classification_report(all_labels, all_preds,
                                              target_names=class_names))

    # ── Generate all plots ────────────────────────────────────────────────────
    _plot_confusion_matrix(all_labels, all_preds, class_names, save_dir)
    _plot_roc_curve(all_labels, all_probs, auc_roc, save_dir)
    _plot_pr_curve(all_labels, all_probs, avg_prec, save_dir)

    logger.info("All plots saved to: %s", save_dir)
    return metrics


# =============================================================================
# Confusion Matrix
# =============================================================================

def _plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    save_dir: str,
) -> None:
    """
    Plot a normalized confusion matrix.
    Normalization (dividing by row sums) shows the proportion of each class
    that was correctly classified — more informative than raw counts.
    """
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # Row-normalize

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"]
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax)

        # Annotate each cell with value
        fmt = ".2f" if data.dtype == float else "d"
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=14,
                        color="white" if data[i, j] > thresh else "black")

    plt.tight_layout()
    path = Path(save_dir) / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


# =============================================================================
# ROC Curve
# =============================================================================

def _plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    auc_score: float,
    save_dir: str,
) -> None:
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    ROC curve shows the tradeoff between:
      - True Positive Rate (Recall/Sensitivity) on Y-axis
      - False Positive Rate on X-axis

    AUC = 1.0 → perfect model
    AUC = 0.5 → random guessing
    """
    fpr, tpr, _ = roc_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1F6FEB", lw=2,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1F6FEB")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — Pneumonia Classifier", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    path = Path(save_dir) / "roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


# =============================================================================
# Precision-Recall Curve
# =============================================================================

def _plot_pr_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    avg_precision: float,
    save_dir: str,
) -> None:
    """
    Plot the Precision-Recall curve.

    More informative than ROC when classes are imbalanced.
    A high area under PR curve means good performance on the positive (PNEUMONIA) class.
    """
    precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_vals, precision_vals, color="#238636", lw=2,
            label=f"PR Curve (AP = {avg_precision:.4f})")
    ax.fill_between(recall_vals, precision_vals, alpha=0.1, color="#238636")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Pneumonia Classifier", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    path = Path(save_dir) / "pr_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


# =============================================================================
# Training History Plots
# =============================================================================

def plot_training_history(history: dict, save_dir: str = "logs/plots") -> None:
    """
    Plot loss and accuracy curves from the training history dict.

    Args:
        history: Dict with keys:
                 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                 'train_f1', 'val_f1', 'train_recall', 'val_recall'
        save_dir: Where to save the plots
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History — Pneumonia Classifier", fontsize=16, fontweight="bold")

    # Pairs: (train_key, val_key, ylabel, title)
    plots = [
        ("train_loss",   "val_loss",   "Loss",    "Loss"),
        ("train_acc",    "val_acc",    "Accuracy","Accuracy"),
        ("train_f1",     "val_f1",     "F1 Score","F1 Score"),
        ("train_recall", "val_recall", "Recall",  "Recall (Sensitivity)"),
    ]

    colors = [("#1F6FEB", "#D29922"), ("#238636", "#D73A49"),
              ("#6F42C1", "#E36209"), ("#0D7377", "#E36209")]

    for ax, (tk, vk, ylabel, title), (tc, vc) in zip(axes.flat, plots, colors):
        if tk in history:
            ax.plot(epochs, history[tk], color=tc, lw=2, label="Train")
        if vk in history:
            ax.plot(epochs, history[vk], color=vc, lw=2, linestyle="--", label="Validation")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)

    plt.tight_layout()
    path = Path(save_dir) / "training_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training history plot saved: %s", path)

# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import yaml
    from src.model import build_model
    from src.dataset import get_dataloaders

    # Set up simple console logging if run directly
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Evaluate Pneumonia Classifier")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_efficientnet_b3_stage1.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Test Data
    logger.info("Loading test dataset...")
    _, _, test_loader, class_names = get_dataloaders(
        data_dir=cfg["paths"]["data_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        num_workers=cfg["data"]["num_workers"],
        random_seed=cfg["data"]["random_seed"],
    )

    # Build Model & Load Weights
    logger.info(f"Loading model from {args.checkpoint}...")
    model = build_model(
        architecture=cfg["model"]["architecture"],
        pretrained=False,
        num_classes=cfg["model"]["num_classes"],
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)

    # Run Evaluation
    evaluate_model(model, test_loader, device, class_names=class_names, save_dir="logs/plots")

