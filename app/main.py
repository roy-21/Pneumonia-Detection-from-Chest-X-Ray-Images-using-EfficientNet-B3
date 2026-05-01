"""
app/main.py
===========
FastAPI backend for serving the Pneumonia Classifier.

Endpoints:
  - GET  /health   : Check if the API is running and model is loaded.
  - POST /predict  : Upload a Chest X-ray image for prediction.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import HealthResponse, PredictionResponse
from src.grad_cam import generate_gradcam
from src.predict import PneumoniaPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CKPT_FINETUNE = "models/checkpoints/best_efficientnet_b3_finetune.pth"
CKPT_STAGE1 = "models/checkpoints/best_efficientnet_b3_stage1.pth"
MODEL_CHECKPOINT = CKPT_FINETUNE if Path(CKPT_FINETUNE).exists() else CKPT_STAGE1
ARCHITECTURE = "efficientnet_b3"
UPLOAD_DIR = Path("logs/uploads")
HEATMAP_DIR = Path("logs/heatmaps")
STATIC_DIR = Path("app/static")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Pneumonia Classifier API",
    description="Deep Learning API for detecting Pneumonia from Chest X-Rays.",
    version="1.0.0",
)

# Mount static files to serve heatmaps and frontend
app.mount("/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    """Serve the frontend index.html"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found. Please create app/static/index.html"}

# Global predictor instance
predictor = None

@app.on_event("startup")
def load_model():
    """Load model weights into memory on startup to avoid loading per-request."""
    global predictor
    try:
        # Note: In a real production scenario, you'd download the model from an S3 bucket 
        # or MLflow model registry if it's not present locally.
        if Path(MODEL_CHECKPOINT).exists():
            predictor = PneumoniaPredictor(checkpoint_path=MODEL_CHECKPOINT, architecture=ARCHITECTURE)
            logger.info("Model loaded successfully.")
        else:
            logger.warning("Checkpoint %s not found. Model not loaded.", MODEL_CHECKPOINT)
    except Exception as e:
        logger.error("Failed to load model: %s", e)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=(predictor is not None),
        device=str(predictor.device) if predictor else "none"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image, get the pneumonia prediction and a Grad-CAM heatmap.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Generate unique filenames to avoid collisions
    job_id = str(uuid.uuid4())
    img_ext = Path(file.filename).suffix or ".jpeg"
    save_path = UPLOAD_DIR / f"{job_id}{img_ext}"
    heatmap_path = HEATMAP_DIR / f"{job_id}_cam.png"

    try:
        # Save uploaded file temporarily
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("Processing image: %s", save_path)

        # 1. Run prediction
        result = predictor.predict(str(save_path))

        # 2. Generate Explainability Heatmap (Grad-CAM)
        generate_gradcam(
            model=predictor.model,
            architecture=ARCHITECTURE,
            image_path=str(save_path),
            save_path=str(heatmap_path)
        )

        # Attach heatmap URL to response
        result["heatmap_url"] = f"/heatmaps/{heatmap_path.name}"
        
        return PredictionResponse(**result)

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        # Clean up the original uploaded image if needed (optional)
        # if save_path.exists():
        #     save_path.unlink()
        pass
