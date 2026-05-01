"""
src/train.py
============
Full training loop for the Pneumonia Classifier.

Responsibilities:
  - Train for N epochs with early stopping
  - Track loss, accuracy, F1, recall per epoch
  - Save the best model checkpoint (based on val_f1)
  - Log all metrics to MLflow
  - Support both Stage 1 (frozen backbone) and Stage 2 (fine-tuning)

Usage:
    python src/train.py --config configs/config.yaml --arch efficientnet_b3
    python src/train.py --config configs/config.yaml --arch efficientnet_b3 --finetune
"""

import argparse
import logging
import os
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.dataset import get_dataloaders
from src.model import build_model, get_optimizer_param_groups, unfreeze_blocks

# Set up structured logging (not print statements)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Loader
# =============================================================================

def load_config(path: str) -> dict:
    """Load YAML config file and return as a Python dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from: %s", path)
    return cfg


# =============================================================================
# Device Setup
# =============================================================================

def get_device() -> torch.device:
    """
    Automatically select the best available device.
    Priority: CUDA GPU → Apple MPS → CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no GPU found)")
    return device


# =============================================================================
# Loss Function Builder
# =============================================================================

def build_loss_fn(train_loader, device: torch.device) -> nn.BCEWithLogitsLoss:
    """
    Build BCEWithLogitsLoss with class weighting to handle imbalance.

    BCEWithLogitsLoss = Sigmoid + BinaryCrossEntropy combined.
    This is numerically more stable than applying sigmoid then BCE separately.

    pos_weight: upweights PNEUMONIA class (positive class) in the loss.
    Since NORMAL is the minority → we actually want to upweight NORMAL (class 0)?
    Wait: in PyTorch BCEWithLogitsLoss, class index 1 = "positive" class.

    Class mapping (ImageFolder sorts alphabetically):
      0 = NORMAL    (minority — want high recall for finding sick patients)
      1 = PNEUMONIA (majority)

    Here, pos_weight > 1 means: penalise the model MORE when it misses class 1.
    But medically we want to not miss PNEUMONIA (class 1), so pos_weight >= 1 is fine.
    We compute it automatically from the actual class counts.
    """
    # Count samples per class from the DataLoader's dataset
    dataset = train_loader.dataset
    normal_count    = sum(1 for _, l in dataset.dataset.samples if l == 0)
    pneumonia_count = sum(1 for _, l in dataset.dataset.samples if l == 1)

    # pos_weight = count_negative / count_positive
    pos_weight = torch.tensor([normal_count / pneumonia_count], dtype=torch.float32).to(device)
    logger.info("Loss pos_weight (NORMAL/PNEUMONIA ratio): %.4f", pos_weight.item())

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# =============================================================================
# One Epoch of Training
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Run one full training epoch.

    Returns:
        dict with keys: loss, accuracy, f1, recall
    """
    model.train()  # Enable dropout, batch norm training mode

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to GPU/CPU
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # Shape: (B, 1)

        # Forward pass — compute predictions
        optimizer.zero_grad()           # Clear gradients from previous batch
        logits = model(images)          # Raw logits (no sigmoid yet)
        loss   = criterion(logits, labels)  # Compute loss

        # Backward pass — compute gradients
        loss.backward()
        optimizer.step()                # Update model weights

        # Accumulate metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().squeeze().tolist()
        lbls  = labels.long().cpu().squeeze().tolist()

        # Handle single-item batches (squeeze may remove batch dim)
        if isinstance(preds, int): preds = [preds]
        if isinstance(lbls, int):  lbls  = [lbls]

        all_preds.extend(preds)
        all_labels.extend(lbls)

    avg_loss = total_loss / len(loader)
    acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    recall   = recall_score(all_labels, all_preds, zero_division=0)

    return {"loss": avg_loss, "accuracy": acc, "f1": f1, "recall": recall}


# =============================================================================
# One Epoch of Validation
# =============================================================================

@torch.no_grad()  # Disable gradient computation (saves memory + speeds up eval)
def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on the validation set.
    No gradient updates — inference only.

    Returns:
        dict with keys: loss, accuracy, f1, recall
    """
    model.eval()  # Disable dropout, use running stats for batch norm

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().squeeze().tolist()
        lbls  = labels.long().cpu().squeeze().tolist()

        if isinstance(preds, int): preds = [preds]
        if isinstance(lbls, int):  lbls  = [lbls]

        all_preds.extend(preds)
        all_labels.extend(lbls)

    avg_loss = total_loss / len(loader)
    acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    recall   = recall_score(all_labels, all_preds, zero_division=0)

    return {"loss": avg_loss, "accuracy": acc, "f1": f1, "recall": recall}


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Stop training when the monitored metric stops improving.

    patience: How many epochs to wait after last improvement.
    min_delta: Minimum change to count as improvement.
    mode: 'min' for loss, 'max' for accuracy/f1/recall.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score + self.min_delta) if self.mode == "max" \
                   else (score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            logger.info("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                logger.info("Early stopping triggered.")
                self.stop = True

        return self.stop


# =============================================================================
# Checkpoint Saver
# =============================================================================

def save_checkpoint(model, optimizer, epoch, metrics, path: str) -> None:
    """
    Save model weights, optimizer state, and metrics to a .pth file.
    Allows training to be resumed or the best model to be reloaded.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":    metrics,
    }, path)
    logger.info("Checkpoint saved: %s (epoch %d, val_f1=%.4f)", path, epoch, metrics["f1"])


# =============================================================================
# Main Training Function
# =============================================================================

def train(config_path: str, architecture: str = None, finetune: bool = False) -> None:
    """
    End-to-end training pipeline.

    Stage 1 (finetune=False): Backbone frozen, only train classifier head.
    Stage 2 (finetune=True):  Unfreeze last blocks, train entire model with
                               differential LR (very small for backbone).

    Args:
        config_path:  Path to configs/config.yaml
        architecture: Override architecture from config (optional)
        finetune:     If True, unfreeze backbone blocks and fine-tune
    """
    # ── Load config ──────────────────────────────────────────────────────────
    cfg   = load_config(config_path)
    arch  = architecture or cfg["model"]["architecture"]
    device = get_device()

    # Seed everything for reproducibility
    torch.manual_seed(cfg["data"]["random_seed"])

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir    = cfg["paths"]["data_dir"],
        image_size  = cfg["data"]["image_size"],
        batch_size  = cfg["training"]["batch_size"],
        val_split   = cfg["data"]["val_split"],
        num_workers = cfg["data"]["num_workers"],
        random_seed = cfg["data"]["random_seed"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        architecture = arch,
        pretrained   = cfg["model"]["pretrained"],
        dropout      = cfg["model"]["dropout"],
        num_classes  = cfg["model"]["num_classes"],
    ).to(device)

    # Stage 2: unfreeze last blocks for fine-tuning
    if finetune and arch == "efficientnet_b3":
        unfreeze_blocks(model, cfg["finetune"]["unfreeze_blocks"])

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if finetune:
        # Differential LR: backbone gets 10× smaller LR than head
        param_groups = get_optimizer_param_groups(
            model,
            head_lr     = cfg["training"]["learning_rate"],
            backbone_lr = cfg["training"]["backbone_lr"],
        )
        optimizer = Adam(param_groups)
    else:
        # Only train classifier head parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(trainable_params, lr=cfg["training"]["learning_rate"])

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = build_loss_fn(train_loader, device)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # CosineAnnealingLR: smoothly reduces LR from max → near-0 over T_max epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])

    # ── Checkpoint path ───────────────────────────────────────────────────────
    stage   = "finetune" if finetune else "stage1"
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"best_{arch}_{stage}.pth"
    best_f1   = 0.0

    # ── MLflow Experiment ─────────────────────────────────────────────────────
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    run_name = f"{arch}_{stage}"

    with mlflow.start_run(run_name=run_name):
        # Log all hyperparameters from config to MLflow
        mlflow.log_params({
            "architecture": arch,
            "pretrained":   cfg["model"]["pretrained"],
            "dropout":      cfg["model"]["dropout"],
            "epochs":       cfg["training"]["epochs"],
            "batch_size":   cfg["training"]["batch_size"],
            "lr":           cfg["training"]["learning_rate"],
            "finetune":     finetune,
        })

        logger.info("=" * 60)
        logger.info("Training started: %s | Stage: %s", arch, stage)
        logger.info("=" * 60)

        # ── Epoch Loop ────────────────────────────────────────────────────────
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            epoch_start = time.time()

            # --- Train ---
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)

            # --- Validate ---
            val_metrics = validate_one_epoch(model, val_loader, criterion, device)

            # --- Scheduler step ---
            scheduler.step()

            epoch_time = time.time() - epoch_start

            # --- Logging ---
            logger.info(
                "Epoch %02d/%02d | "
                "Train Loss: %.4f Acc: %.4f F1: %.4f Recall: %.4f | "
                "Val Loss: %.4f Acc: %.4f F1: %.4f Recall: %.4f | "
                "Time: %.1fs",
                epoch, cfg["training"]["epochs"],
                train_metrics["loss"], train_metrics["accuracy"],
                train_metrics["f1"],   train_metrics["recall"],
                val_metrics["loss"],   val_metrics["accuracy"],
                val_metrics["f1"],     val_metrics["recall"],
                epoch_time,
            )

            # --- Log to MLflow ---
            mlflow.log_metrics({
                "train_loss":    train_metrics["loss"],
                "train_acc":     train_metrics["accuracy"],
                "train_f1":      train_metrics["f1"],
                "train_recall":  train_metrics["recall"],
                "val_loss":      val_metrics["loss"],
                "val_acc":       val_metrics["accuracy"],
                "val_f1":        val_metrics["f1"],
                "val_recall":    val_metrics["recall"],
            }, step=epoch)

            # --- Save best model checkpoint ---
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                save_checkpoint(model, optimizer, epoch, val_metrics, str(best_ckpt))
                mlflow.pytorch.log_model(model, artifact_path="best_model")

            # --- Early stopping check ---
            if early_stop(val_metrics["f1"]):
                break

        logger.info("Training complete. Best val_f1: %.4f | Checkpoint: %s", best_f1, best_ckpt)
        mlflow.log_metric("best_val_f1", best_f1)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pneumonia Classifier")
    parser.add_argument("--config",   default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--arch",     default=None, help="Override architecture from config")
    parser.add_argument("--finetune", action="store_true", help="Enable fine-tuning mode")
    args = parser.parse_args()

    train(config_path=args.config, architecture=args.arch, finetune=args.finetune)
