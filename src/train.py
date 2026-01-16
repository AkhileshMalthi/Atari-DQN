import os
import torch
import numpy as np
import mlflow
from collections import deque
from gymnasium.spaces import Discrete
from atari_env_wrapper import make_atari_env
from agent import DQNAgent
from model import AtariCNN
from memory import ExperienceReplayBuffer

def load_checkpoint(agent, buffer, checkpoint_path):
    """Load checkpoint to resume training."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        agent.target_net.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        agent.steps_done = checkpoint.get('steps_done', 0)
        start_step = checkpoint['step']
        print(f"✅ Resumed from step {start_step}, epsilon={agent.epsilon:.4f}")
        return start_step
    return 1

def train(resume_from=None):
    # --- Configuration ---
    GAME_ID = "ALE/Pong-v5"
    BATCH_SIZE = 32
    LR = 1e-4
    GAMMA = 0.99
    REPLAY_SIZE = 100000
    TARGET_UPDATE_FREQ = 10000 # steps
    TOTAL_STEPS = 10000000
    CHECKPOINT_FREQ = 50000
    EARLY_STOP_REWARD = 10.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_atari_env(GAME_ID)
    
    action_dim = env.action_space.n if isinstance(env.action_space, Discrete) else 6
    agent = DQNAgent(AtariCNN, action_dim, device, lr=LR, gamma=GAMMA)
    buffer = ExperienceReplayBuffer(REPLAY_SIZE, env.observation_space.shape)
    
    # Load checkpoint if resuming
    start_step = 1
    if resume_from:
        start_step = load_checkpoint(agent, buffer, resume_from)
    
    # MLflow tracking initialization
    mlflow.set_experiment("DQN_Atari_Pong")
    with mlflow.start_run():
        mlflow.log_params({"lr": LR, "gamma": GAMMA, "batch_size": BATCH_SIZE})
        
        state, _ = env.reset()
        episode_rewards = deque(maxlen=100)
        current_episode_reward = 0.0
        current_episode_steps = 0
        q_values_log = []
        
        for step in range(start_step, TOTAL_STEPS + 1):
            # 1. Action Selection
            action = agent.select_action(state)
            
            # 2. Environment Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            current_episode_reward += float(reward)
            current_episode_steps += 1
            
            # 3. Store Experience
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # 4. Optimization Step
            if buffer.size > BATCH_SIZE:
                experiences = buffer.sample(BATCH_SIZE)
                loss, avg_q = agent.update_model(experiences)
                q_values_log.append(avg_q)
                
                if step % 100 == 0:
                    mlflow.log_metric("loss", loss, step=step)
                    if q_values_log:
                        mlflow.log_metric("avg_q_value", float(np.mean(q_values_log)), step=step)
                        q_values_log = []

            # 5. Target Network Sync
            if step % TARGET_UPDATE_FREQ == 0:
                agent.sync_target_network()

            # 6. Episode End Handling
            if done:
                episode_rewards.append(current_episode_reward)
                avg_reward = np.mean(episode_rewards)
                
                # Logging
                mlflow.log_metric("reward", current_episode_reward, step=step)
                mlflow.log_metric("avg_reward_100", float(avg_reward), step=step)
                mlflow.log_metric("epsilon", agent.epsilon, step=step)
                mlflow.log_metric("episode_duration", current_episode_steps, step=step)
                
                print(f"Step: {step} | Reward: {current_episode_reward:.1f} | Avg: {avg_reward:.2f} | Duration: {current_episode_steps} | Epsilon: {agent.epsilon:.2f}")
                
                # Early Stopping
                if len(episode_rewards) == 100 and avg_reward >= EARLY_STOP_REWARD:
                    print(f"Goal reached! Training finished at step {step}")
                    # Save final model to root directory for Docker volume mounting
                    torch.save({
                        'step': step,
                        'model_state_dict': agent.policy_net.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'epsilon': agent.epsilon
                    }, "final_model.pth")
                    mlflow.log_artifact("final_model.pth")
                    save_checkpoint(agent, step, "final_model.pth")
                    break
                
                state, _ = env.reset()
                current_episode_reward = 0.0
                current_episode_steps = 0

            # 7. Periodic Checkpointing
            if step % CHECKPOINT_FREQ == 0:
                save_checkpoint(agent, step, f"checkpoint_{step}.pth")

def save_checkpoint(agent, step, filename):
    """Saves model and optimizer state for reproducibility."""
    os.makedirs("checkpoints", exist_ok=True)
    filepath = os.path.join("checkpoints", filename)
    torch.save({
        'step': step,
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'steps_done': agent.steps_done
    }, filepath)
    mlflow.log_artifact(filepath)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(resume_from=args.resume)