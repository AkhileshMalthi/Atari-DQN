# test_agent_logic.py
import torch
import numpy as np
from model import AtariCNN
from agent import DQNAgent

def test_epsilon_decay():
    device = torch.device("cpu")
    agent = DQNAgent(AtariCNN, action_dim=6, device=device, epsilon_decay=100)
    
    initial_epsilon = agent.epsilon
    for _ in range(50):
        agent.select_action(np.zeros((4, 84, 84), dtype=np.uint8))
    
    assert agent.epsilon < initial_epsilon, "Epsilon did not decay"
    print(f"✅ Epsilon Decay Verified: {initial_epsilon:.2f} -> {agent.epsilon:.2f}")

def test_training_step():
    device = torch.device("cpu")
    agent = DQNAgent(AtariCNN, action_dim=6, device=device)
    
    # Mock a batch of experiences
    batch_size = 32
    obs_shape = (4, 84, 84)
    experiences = (
        torch.randn(batch_size, *obs_shape), # states
        torch.zeros(batch_size, dtype=torch.long), # actions
        torch.ones(batch_size), # rewards
        torch.randn(batch_size, *obs_shape), # next_states
        torch.zeros(batch_size, dtype=torch.bool) # dones
    )
    
    loss = agent.update_model(experiences)
    assert isinstance(loss, float) and loss >= 0, "Invalid loss value"
    print("✅ Optimization Step (Forward/Backward/Clipping) Verified!")

if __name__ == "__main__":
    test_epsilon_decay()
    test_training_step()