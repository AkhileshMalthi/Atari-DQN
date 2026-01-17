# test_api.py
from fastapi.testclient import TestClient
import numpy as np
from app import app

client = TestClient(app)

def test_predict_endpoint():
    # Create a dummy state (4, 84, 84)
    dummy_state = np.zeros((4, 84, 84), dtype=np.uint8).tolist()
    
    response = client.post("/predict", json={"state": dummy_state})
    
    assert response.status_code == 200
    assert "action" in response.json()
    assert isinstance(response.json()["action"], int)
    print("✅ API Inference Endpoint Verified!")

if __name__ == "__main__":
    test_predict_endpoint()