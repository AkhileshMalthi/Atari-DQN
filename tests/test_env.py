import numpy as np
from atari_env_wrapper import make_atari_env

def test_environment_output_shape():
    env = make_atari_env("ALE/Pong-v5")
    state, info = env.reset()
    
    # Check shape: Expect (4, 84, 84)
    assert state.shape == (4, 84, 84), f"Expected (4, 84, 84), got {state.shape}"
    
    # Check data type: Expect uint8 for memory efficiency
    assert state.dtype == np.uint8, "Expected uint8 data type"
    
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    
    assert next_state.shape == (4, 84, 84)
    print("✅ Environment Step and Shapes Verified!")

if __name__ == "__main__":
    test_environment_output_shape()