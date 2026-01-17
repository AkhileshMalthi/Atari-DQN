"""Quick test script to verify the environment setup works correctly."""

import sys
print("Testing environment setup...")

try:
    from gymnasium.spaces import Discrete
    import torch
    
    from src.atari_env_wrapper import make_atari_env
    from src.model import AtariCNN
    from src.agent import DQNAgent
    from src.memory import ExperienceReplayBuffer
    
    print("✅ All imports successful")
    
    # Test environment creation
    env = make_atari_env("ALE/Pong-v5")
    print(f"✅ Environment created: {env.observation_space.shape}")
    
    # Test model creation
    device = torch.device("cpu")
    action_dim = env.action_space.n if isinstance(env.action_space, Discrete) else 6
    model = AtariCNN(action_dim).to(device)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test agent
    agent = DQNAgent(AtariCNN, action_dim, device)
    print("✅ Agent initialized")
    
    # Test replay buffer
    buffer = ExperienceReplayBuffer(1000, env.observation_space.shape)
    print("✅ Replay buffer created")
    
    # Quick environment step
    state, _ = env.reset()
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Environment step successful (action: {action}, reward: {reward})")
    
    print("\n🎉 All tests passed! Environment is ready for training.")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
