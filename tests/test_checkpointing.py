import torch
import os
from model import AtariCNN
from agent import DQNAgent
from train import save_checkpoint

def test_checkpoint_contents():
    device = torch.device("cpu")
    agent = DQNAgent(AtariCNN, action_dim=6, device=device)
    filename = "test_ckpt.pth"
    
    save_checkpoint(agent, 100, filename)
    
    # Check in checkpoints directory where save_checkpoint actually saves
    checkpoint_path = os.path.join("checkpoints", filename)
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    required_keys = ['step', 'model_state_dict', 'optimizer_state_dict', 'epsilon']
    for key in required_keys:
        assert key in checkpoint, f"Missing key {key} in checkpoint"
    
    os.remove(checkpoint_path)
    print("✅ Checkpoint Integrity Verified!")

if __name__ == "__main__":
    test_checkpoint_contents()