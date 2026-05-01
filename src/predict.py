"""
src/predict.py
==============
Inference pipeline for the Pneumonia Classifier.

Responsibilities:
  - Load a trained model checkpoint
  - Preprocess a raw image (Grayscale → RGB → Resize → Normalize)
  - Run inference and return the predicted class and confidence
  - This is the module used by the FastAPI backend.

Usage:
    from src.predict import PneumoniaPredictor
    predictor = PneumoniaPredictor("models/checkpoints/best_efficientnet_b3_finetune.pth")
    result = predictor.predict("path/to/image.jpeg")
"""

import logging
from pathlib import Path

import torch
from PIL import Image

from src.dataset import build_eval_transforms
from src.model import build_model

logger = logging.getLogger(__name__)


class PneumoniaPredictor:
    """
    Stateful predictor class. Keeps the model loaded in memory for fast inference.
    """

    def __init__(
        self,
        checkpoint_path: str,
        architecture: str = "efficientnet_b3",
        image_size: int = 224,
    ):
        """
        Initialize the predictor and load model weights into VRAM/RAM.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing predictor on device: %s", self.device)

        # Build model architecture (pretrained=False because we load our own weights)
        self.model = build_model(
            architecture=architecture,
            pretrained=False,
            num_classes=1,
        )

        # Load weights
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        logger.info("Loading weights from: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle raw state_dict vs dictionary wrapper (from save_checkpoint)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded checkpoint from epoch %d", checkpoint.get("epoch", -1))
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()  # Crucial for inference (disables dropout, fixes batch norm)

        # Image transform pipeline (same as validation)
        self.transform = build_eval_transforms(image_size)
        self.class_names = ["NORMAL", "PNEUMONIA"]

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Run inference on a single image.

        Returns:
            dict containing label string, class index, and confidence score.
        """
        # Load image (PIL loader handles grayscale/RGB gracefully)
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img.load()

        # Preprocess -> add batch dimension -> send to device
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Inference
        logit = self.model(tensor)
        prob = torch.sigmoid(logit).item()

        # Thresholding
        pred_idx = 1 if prob >= 0.5 else 0
        label_name = self.class_names[pred_idx]

        # Confidence: if predicted 0, confidence is (1 - prob)
        confidence = prob if pred_idx == 1 else (1.0 - prob)

        result = {
            "prediction": label_name,
            "class_index": pred_idx,
            "confidence": round(confidence * 100, 2),  # Percentage
            "raw_probability": prob,
        }

        logger.info("Predicted: %s (%.2f%%)", label_name, result["confidence"])
        return result
