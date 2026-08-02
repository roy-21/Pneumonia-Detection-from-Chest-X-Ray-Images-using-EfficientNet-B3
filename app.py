try:
    import spaces  # Must be imported before torch for HF ZeroGPU
except ImportError:
    pass

import gradio as gr
import logging
from pathlib import Path

from src.predict import PneumoniaPredictor
from src.grad_cam import generate_gradcam

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CKPT_FINETUNE = "models/checkpoints/best_efficientnet_b3_finetune.pth"
CKPT_STAGE1 = "models/checkpoints/best_efficientnet_b3_stage1.pth"
MODEL_CHECKPOINT = CKPT_FINETUNE if Path(CKPT_FINETUNE).exists() else CKPT_STAGE1
ARCHITECTURE = "efficientnet_b3"
HEATMAP_DIR = Path("logs/heatmaps")
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# Load predictor globally
predictor = None
try:
    if Path(MODEL_CHECKPOINT).exists():
        predictor = PneumoniaPredictor(checkpoint_path=MODEL_CHECKPOINT, architecture=ARCHITECTURE)
        logger.info("Model loaded successfully.")
    else:
        logger.warning(f"Checkpoint {MODEL_CHECKPOINT} not found.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

def _predict_and_visualize(image_path):
    if predictor is None:
        return "Model is not loaded. Please check if checkpoint exists.", None
    if image_path is None:
        return "Please upload an image.", None
    
    try:
        # Run prediction
        result = predictor.predict(image_path)
        
        # Format the result text
        prediction_text = f"### Prediction: **{result['prediction']}**\n### Confidence: **{result['confidence']}%**"
        
        # Generate Explainability Heatmap (Grad-CAM)
        heatmap_path = HEATMAP_DIR / f"gradcam_temp.png"
        generate_gradcam(
            model=predictor.model,
            architecture=ARCHITECTURE,
            image_path=image_path,
            save_path=str(heatmap_path)
        )
        
        return prediction_text, str(heatmap_path)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return f"Error during prediction: {str(e)}", None

# Wrap for ZeroGPU if available
try:
    predict_and_visualize = spaces.GPU(_predict_and_visualize)
except NameError:
    predict_and_visualize = _predict_and_visualize

# Build Gradio Interface
with gr.Blocks(title="Pneumonia Classifier API") as demo:
    gr.Markdown("# 🫁 Pneumonia Classification AI")
    gr.Markdown("Upload a Chest X-ray to predict whether it is NORMAL or PNEUMONIA. The AI will also generate a Grad-CAM heatmap to show which parts of the lungs it focused on.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Chest X-ray Image")
            submit_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            output_text = gr.Markdown(label="Results")
            output_heatmap = gr.Image(type="filepath", label="Grad-CAM Heatmap (Explainability)")
            
    submit_btn.click(fn=predict_and_visualize, inputs=image_input, outputs=[output_text, output_heatmap])
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
