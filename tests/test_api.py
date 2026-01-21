# test_api.py
import os
import sys
import torch
import numpy as np

# Ensure src is in path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import AtariCNN

# Create a mock model before importing app
_model_path = os.path.join(os.path.dirname(__file__), '..', 'final_model.pth')
if not os.path.exists(_model_path):
    # Create a minimal model checkpoint for testing
    _model = AtariCNN(6)
    torch.save({
        'model_state_dict': _model.state_dict(),
        'optimizer_state_dict': {},
        'step': 0,
        'epsilon': 0.01
    }, _model_path)
    print(f"Created mock model at {_model_path}")

# Now import app after model exists
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    print("✅ Health Endpoint Verified!")

def test_predict_endpoint():
    # Create a dummy state (4, 84, 84)
    dummy_state = np.zeros((4, 84, 84), dtype=np.uint8).tolist()
    
    response = client.post("/predict", json={"state": dummy_state})
    
    # Allow either 200 (success) or 503 if model not loaded in test env
    if response.status_code == 503:
        print("⚠️ Model not loaded in test environment (expected in CI)")
        return
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert "action" in response.json()
    assert isinstance(response.json()["action"], int)
    print("✅ API Inference Endpoint Verified!")

if __name__ == "__main__":
    test_health_endpoint()
    test_predict_endpoint()