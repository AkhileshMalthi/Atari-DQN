from gymnasium.wrappers import RecordVideo
from gymnasium.spaces import Discrete
import torch
import numpy as np
from atari_env_wrapper import make_atari_env
from model import AtariCNN
import os

def evaluate(game_id="ALE/Pong-v5", num_episodes=100, record_video=False):
    """
    METHODOLOGY:
    Runs the agent for N episodes with epsilon=0.
    Calculates average reward to verify the 'Average Reward > 10' requirement.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment with render_mode for video recording
    if record_video:
        env = make_atari_env(game_id, render_mode="rgb_array")
        os.makedirs("./videos", exist_ok=True)
        env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    else:
        env = make_atari_env(game_id)

    # Load Model
    assert isinstance(env.action_space, Discrete), "Action space must be Discrete"
    action_dim = env.action_space.n
    model = AtariCNN(action_dim).to(device)
    
    if not os.path.exists("final_model.pth"):
        print("❌ Error: final_model.pth not found. Please train the model first.")
        return 0.0
    
    checkpoint = torch.load("final_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_rewards = []

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            state_t = torch.as_tensor(state, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model(state_t).argmax().item()
            
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated
            
        total_rewards.append(episode_reward)
        print(f"Episode {i+1}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"\nFINAL EVALUATION SCORE: {avg_reward}")
    
    # Save results for automated parsing
    with open("eval_results.txt", "w") as f:
        f.write(f"average_reward: {avg_reward}")
    
    # Close environment to finalize video recording
    env.close()
        
    return avg_reward

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--record", type=str, default="False", help="Record video (True/False)")
    args = parser.parse_args()
    
    record = args.record.lower() in ["true", "1", "yes"]
    evaluate(num_episodes=args.episodes, record_video=record)