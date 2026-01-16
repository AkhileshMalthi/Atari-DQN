import numpy as np
import torch

class ExperienceReplayBuffer:
    """
    High-performance circular buffer using pre-allocated numpy arrays.
    METHODOLOGY:
    - Pre-allocation prevents memory fragmentation during long training runs.
    - Circular indexing (ptr) ensures constant-time O(1) insertions.
    """
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Observation shape is (4, 84, 84)
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices])
        )

    def __len__(self):
        """Return the current size of the buffer."""
        return self.size