# Methodology: Deep Q-Network for Atari Games

This document outlines the implementation details, hyperparameter selections, and architectural decisions made for the development of the production-grade DQN agent.

## 1. Algorithmic Foundation

The core of this system is based on the **Deep Q-Network (DQN)** algorithm introduced by DeepMind. To ensure stable learning in the high-dimensional visual state space of Atari, several key components were implemented from scratch:

### Experience Replay

* **Design**: A circular buffer using pre-allocated NumPy arrays.
* **Methodology**: By storing  tuples and sampling mini-batches randomly, we break the temporal correlation of the agent's experiences. This makes the data distribution independent and identically distributed (i.i.d.), which is a requirement for stable gradient descent.

### Target Networks

* **Design**: A separate Q-network that is updated only every 10,000 steps.
* **Methodology**: Using a stationary target for the Bellman equation prevents the "chasing your own tail" problem where the network updates shift the target values it is trying to predict, leading to divergence.

## 2. Neural Network Architecture

The "Nature DQN" CNN was selected for its proven efficacy in vision-based RL tasks:

* **Input**:  tensor (stacked grayscale frames).
* **Convolutional Layers**: 3 layers designed to extract spatial features (e.g., the position of the ball and the velocity of the paddle).
* **Activation**: ReLU is used throughout for non-linearity.
* **Output**: Linear layer representing the Q-value for each discrete action in the game's action space.

## 3. Training Stability & Hyperparameters

To achieve the requirement of an average reward  on Pong-v5, the following stability measures were implemented:

| Hyperparameter | Value | Reasoning |
| --- | --- | --- |
| **Optimizer** | Adam | Efficient handling of sparse gradients in RL. |
| **Learning Rate** | 1e-4 | Balanced for steady convergence without overshooting optima. |
| **Gamma ()** | 0.99 | High discount factor to prioritize long-term rewards (winning the point). |
| **Batch Size** | 32 | Standard size for efficient GPU utilization and gradient estimation. |
| **Loss Function** | Huber (Smooth L1) | Less sensitive to outliers/extreme rewards than MSE. |
| **Reward Clipping** | [-1, 1] | Normalizes gradients across different game scenarios. |

## 4. Exploration Strategy

An **Epsilon-Greedy** strategy was implemented to manage the exploration-exploitation tradeoff:

* **Schedule**: Linear decay from  (pure exploration) to  (pure exploitation) over the first 1,000,000 steps.
* **Logic**: This ensures the agent thoroughly explores the environment early on before settling into the optimal learned policy.

## 5. MLOps & Reproducibility

* **Experiment Tracking**: MLflow is used to log rewards, loss, and epsilon values in real-time.
* **Containerization**: Separate Dockerfiles for training and inference ensure that the deployment environment is slim and contains only necessary dependencies.
* **Checkpoints**: The system saves both the model weights and the optimizer state at regular intervals to allow for training resumption.

## 6. Performance Validation

The agent is evaluated over 100 consecutive episodes using a deterministic policy (). A video demonstration is generated to visualize the agent's ability to track the ball and position the paddle effectively.