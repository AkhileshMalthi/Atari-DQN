import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import AtariCNN
import os

# Global variable to hold the model
model = None
device = torch.device("cpu")


def load_model():
    """
    METHODOLOGY: Load model once on startup (Singleton Pattern).
    The model path is typically passed via environment variables.
    """
    global model
    model_path = os.getenv("MODEL_PATH", "final_model.pth")
    
    # We only need the action dimension for initialization
    # In Pong-v5, this is typically 6
    action_dim = int(os.getenv("ACTION_DIM", "6"))
    
    model = AtariCNN(action_dim)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️ Warning: No model found at {model_path}. API will fail on /predict.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    load_model()
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(title="Atari DQN Inference Service", lifespan=lifespan)


class Observation(BaseModel):
    state: list  # Expecting a nested list representing (4, 84, 84)

@app.post("/predict")
async def predict(observation: Observation):
    """
    METHODOLOGY: Convert JSON list to Tensor and perform forward pass.
    Returns the optimal action as a JSON response.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert list to numpy then torch tensor
        state_np = np.array(observation.state, dtype=np.uint8)
        
        # Validation of input shape (4, 84, 84)
        if state_np.shape != (4, 84, 84):
            raise ValueError(f"Expected shape (4, 84, 84), got {state_np.shape}")
            
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = model(state_t)
            action = q_values.argmax().item()
            
        return {"action": action}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }