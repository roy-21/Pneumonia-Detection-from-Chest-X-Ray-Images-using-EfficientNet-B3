from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class ('NORMAL' or 'PNEUMONIA')")
    confidence: float = Field(..., description="Confidence percentage (0 to 100)")
    raw_probability: float = Field(..., description="Raw output probability (0.0 to 1.0)")
    heatmap_url: str | None = Field(None, description="URL path to the generated Grad-CAM heatmap overlay")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Indicates if the model weights are successfully loaded")
    device: str = Field(..., description="Compute device being used (cpu/cuda)")
