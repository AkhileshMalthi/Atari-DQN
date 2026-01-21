import torch
import numpy as np
from model import AtariCNN
from memory import ExperienceReplayBuffer

def test_dqn_output():
    model = AtariCNN(action_dim=6)
    dummy_input = torch.randn(1, 4, 84, 84)
    output = model(dummy_input)
    assert output.shape == (1, 6), "CNN output shape mismatch"
    print("✅ Model Forward Pass Verified!")

def test_buffer_sampling():
    buffer = ExperienceReplayBuffer(capacity=100, state_shape=(4, 84, 84))
    for _ in range(50):
        buffer.push(np.zeros((4,84,84)), 1, 0.5, np.zeros((4,84,84)), False)
    
    states, actions, rewards, next_states, dones = buffer.sample(32)
    assert states.shape == (32, 4, 84, 84)
    assert isinstance(states, torch.Tensor)
    print("✅ Replay Buffer Sampling Verified!")

if __name__ == "__main__":
    test_dqn_output()
    test_buffer_sampling()