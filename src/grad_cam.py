"""
src/grad_cam.py
===============
Explainability module using Grad-CAM (Gradient-weighted Class Activation Mapping).

Grad-CAM produces a heatmap highlighting the regions of the image that
contributed most to the model's prediction. This is critical for medical AI
to ensure the model is looking at the lungs and not artifacts (like text or pacemakers).

Usage:
    from src.grad_cam import generate_gradcam
    heatmap_path = generate_gradcam(model, "image.jpeg", "logs/plots/cam.png")
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import build_eval_transforms

logger = logging.getLogger(__name__)


def get_target_layer(model: nn.Module, architecture: str) -> list[nn.Module]:
    """
    Find the target layer for Grad-CAM.
    Usually the last convolutional layer of the backbone.
    """
    arch = architecture.lower()
    if "efficientnet" in arch:
        # Last conv layer before pooling in EfficientNet
        return [model.features[-1]]
    elif "resnet" in arch:
        return [model.layer4[-1]]
    elif "vgg" in arch:
        return [model.features[-1]]
    elif "densenet" in arch:
        return [model.features.denseblock4.denselayer16.conv2]
    else:
        # Fallback for baseline CNN
        return [model.features[-1]]


def generate_gradcam(
    model: nn.Module,
    architecture: str,
    image_path: str,
    save_path: str,
    image_size: int = 224,
) -> str:
    """
    Generate and save a Grad-CAM heatmap overlay.

    Args:
        model:        Loaded PyTorch model
        architecture: Model name (used to find target layer)
        image_path:   Path to input image
        save_path:    Where to save the output heatmap
        image_size:   Resize dimension

    Returns:
        Path to the saved heatmap
    """
    device = next(model.parameters()).device
    model.eval()

    # 1. Load and prepare original image (for visualization overlay)
    raw_img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    # Convert PIL to float numpy array [0, 1] for show_cam_on_image
    rgb_img = np.float32(raw_img) / 255.0

    # 2. Preprocess image for model (normalize, etc.)
    transform = build_eval_transforms(image_size)
    with open(image_path, "rb") as f:
        img_pil = Image.open(f)
        img_pil.load()
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 3. Setup Grad-CAM
    target_layers = get_target_layer(model, architecture)

    # Ensure target layers require grad so GradCAM can compute gradients
    # (Since we loaded a Stage 1 model where the backbone is frozen)
    for layer in target_layers:
        for param in layer.parameters():
            param.requires_grad = True

    # Initialize CAM object
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # For binary classification (1 output node), target category is 0
        # PyTorch GradCAM expects a list of targets per image in the batch
        targets = [ClassifierOutputTarget(0)]

        # 4. Generate heatmap
        # grayscale_cam is shape (H, W)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # 5. Overlay heatmap on original image
        # Returns numpy array (H, W, 3) in uint8 [0, 255]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 6. Save result
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # Convert RGB to BGR for OpenCV saving
        cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        logger.info("Grad-CAM saved to: %s", save_path)

        return save_path
