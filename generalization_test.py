import os
import glob
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import DQN, A2C, PPO
from environment.custom_env import EcoTrackEnv

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# --- Helper to Find Files ---
def find_model_path(algo_folder, search_string):
    """
    Searches models/{algo_folder}/ for any file containing 'search_string'
    """
    search_pattern = f"models/{algo_folder}/*{search_string}*"
    files = glob.glob(search_pattern)
    
    if files:
        # Prefer the one with .zip or .pth if multiple match
        for f in files:
            if f.endswith(".zip") or f.endswith(".pth"):
                print(f"Found {algo_folder.upper()} Model: {f}")
                return f
        # Fallback to first match
        print(f"Found {algo_folder.upper()} Model: {files[0]}")
        return files[0]
    else:
        print(f"ERROR: Could not find model for {algo_folder} with pattern '{search_string}'")
        return None

# --- 1. Define REINFORCE Class ---
class ReinforcePolicy(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size=64):
        super(ReinforcePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

def get_reinforce_action(model, obs):
    state_tensor = torch.from_numpy(obs).float().unsqueeze(0)
    probs = model(state_tensor)
    action = torch.argmax(probs).item()
    return action

# --- 2. Evaluation Logic ---
def evaluate(model, env_algo, seeds=[101, 102, 103, 104, 105]):
    env = EcoTrackEnv(render_mode=None)
    rewards = []
    
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0
        
        while not done:
            if env_algo == "reinforce":
                action = get_reinforce_action(model, obs)
            else:
                action, _ = model.predict(obs, deterministic=True)
                
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
        rewards.append(ep_reward)
    
    env.close()
    return np.mean(rewards), np.std(rewards)

# --- 3. Main Runner ---
if __name__ == "__main__":
    results_data = [] 

    print(f"\n{'Algorithm':<12} | {'Mean Reward (New Seeds)':<25} | {'Std Dev':<10}")
    print("-" * 55)

    # --- DQN (Best: Run 7) ---
    # DQN filenames do not have 'run_7', so we search for the specific params of Run 7
    # Params from Card 14: buf5000_bat128
    path = find_model_path("dqn", "buf5000_bat128") 
    if path:
        try:
            load_path = path.replace(".zip", "")
            dqn_model = DQN.load(load_path)
            mean, std = evaluate(dqn_model, "dqn")
            print(f"{'DQN':<12} | {mean:.2f} +/- {std:.2f}".ljust(40))
            results_data.append({"Algorithm": "DQN", "Mean Reward": mean, "Std Dev": std})
        except Exception as e:
            print(f"Error loading DQN: {e}")

    # --- REINFORCE (Best: Run 7) ---
    path = find_model_path("pg", "run_7")
    if path:
        try:
            reinforce_model = ReinforcePolicy(27, 7, hidden_size=64)
            reinforce_model.load_state_dict(torch.load(path))
            mean, std = evaluate(reinforce_model, "reinforce")
            print(f"{'REINFORCE':<12} | {mean:.2f} +/- {std:.2f}".ljust(40))
            results_data.append({"Algorithm": "REINFORCE", "Mean Reward": mean, "Std Dev": std})
        except Exception as e:
            print(f"Error loading REINFORCE: {e}")

    # --- A2C (Best: Run 3) ---
    path = find_model_path("a2c", "run_3")
    if path:
        try:
            load_path = path.replace(".zip", "")
            a2c_model = A2C.load(load_path)
            mean, std = evaluate(a2c_model, "a2c")
            print(f"{'A2C':<12} | {mean:.2f} +/- {std:.2f}".ljust(40))
            results_data.append({"Algorithm": "A2C", "Mean Reward": mean, "Std Dev": std})
        except Exception as e:
            print(f"Error loading A2C: {e}")

    # --- PPO (Best: Run 10) ---
    path = find_model_path("ppo", "run_10")
    if path:
        try:
            load_path = path.replace(".zip", "")
            ppo_model = PPO.load(load_path)
            mean, std = evaluate(ppo_model, "ppo")
            print(f"{'PPO':<12} | {mean:.2f} +/- {std:.2f}".ljust(40))
            results_data.append({"Algorithm": "PPO", "Mean Reward": mean, "Std Dev": std})
        except Exception as e:
            print(f"Error loading PPO: {e}")

    # --- Save to CSV ---
    if results_data:
        df = pd.DataFrame(results_data)
        csv_path = "results/generalization_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSUCCESS: Results saved to {csv_path}")