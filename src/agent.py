import torch
import torch.optim as optim
import torch.nn.functional as F
import random

class DQNAgent:
    """
    Main Agent class encapsulating the DQN logic.
    METHODOLOGY:
    - Uses an epsilon-greedy policy for exploration/exploitation balance.
    - Employs a target network to reduce oscillations in Q-values.
    - Implements Huber Loss and reward clipping for training stability.
    """
    def __init__(self, model_class, action_dim, device, lr=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=1000000):
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        
        # Hyperparameters for epsilon decay
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Networks
        self.policy_net = model_class(action_dim).to(device)
        self.target_net = model_class(action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is always in evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        METHODOLOGY: Decays epsilon linearly until epsilon_final is reached.
        """
        # Linear Epsilon Decay
        self.epsilon = max(self.epsilon_final, 
                           1.0 - (self.steps_done / self.epsilon_decay))
        self.steps_done += 1

        if random.random() > self.epsilon:
            with torch.no_grad():
                # State is (4, 84, 84), add batch dimension
                state_t = torch.tensor(state, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_dim)

    def update_model(self, experiences):
        """
        Performs a single optimization step using a batch of transitions.
        METHODOLOGY: Implements the Bellman equation and Huber Loss.
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Move to device and cast
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).float()

        # Reward clipping to stabilize training [-1, 1]
        rewards = torch.clamp(rewards, -1.0, 1.0)

        # Get current Q-values from policy network
        q_values_all = self.policy_net(states)
        current_q = q_values_all.gather(1, actions)
        
        # Calculate average Q-value for logging
        avg_q_value = current_q.mean().item()

        # Compute TD Target using Target Network
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute Huber Loss (less sensitive to outliers than MSE)
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for extra stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), avg_q_value

    def sync_target_network(self):
        """Copies weights from policy_net to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())