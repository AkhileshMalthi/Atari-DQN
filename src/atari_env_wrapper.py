import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

class AtariPreprocessing(gym.Wrapper):
    """
    Handles Frame Skipping, Max-pooling, Grayscaling, and Resizing.
    METHODOLOGY: 
    - Grayscale reduces the input dimensionality by 3x (RGB -> Y).
    - Max-pooling over the last 2 frames removes 'flickering' artifacts 
      common in Atari 2600 sprites.
    - Resizing to 84x84 is the industry standard for Nature DQN.
    """
    def __init__(self, env, screen_size=84):
        super(AtariPreprocessing, self).__init__(env)
        self.screen_size = screen_size
        # Define the new observation space (Single frame at this stage)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.screen_size, self.screen_size, 1), dtype=np.uint8
        )

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        # Buffer to store frames for max-pooling (to handle sprite flickering)
        frame_buffer = np.zeros((2, 210, 160, 3), dtype=np.uint8)
        
        for i in range(4): # Frame skipping: 4 frames
            obs, reward, term, trunc, info = self.env.step(action)
            if i == 2:
                frame_buffer[0] = obs
            if i == 3:
                frame_buffer[1] = obs
            total_reward += float(reward)
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        
        # Take max over last two frames
        max_frame = frame_buffer.max(axis=0)
        processed_frame = self._preprocess(max_frame)
        return processed_frame, total_reward, terminated, truncated, info

    def _preprocess(self, frame):
        # Convert RGB to Grayscale
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        img = cv2.resize(img, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        return img[:, :, np.newaxis] # Add channel dimension

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess(obs), info

class FrameStack(gym.Wrapper):
    """
    Stacks 'k' consecutive frames to give the agent a sense of motion.
    METHODOLOGY:
    A single static frame doesn't convey velocity (e.g., is the ball moving up or down?).
    Stacking 4 frames allows the CNN to infer direction and speed.
    """
    def __init__(self, env, k=4):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = []
        shp = env.observation_space.shape
        # New shape: (k, 84, 84) - following PyTorch's (C, H, W) convention
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(k, shp[0], shp[1]), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs for _ in range(self.k)] # Initialize with the first frame
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        # Stack frames along the first dimension (Channel dimension)
        return np.stack(self.frames, axis=0).squeeze()

def make_atari_env(game_id):
    """Factory function to build the wrapped environment."""
    env = gym.make(game_id)
    env = AtariPreprocessing(env)
    env = FrameStack(env, k=4)
    return env