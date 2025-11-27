import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd  # <--- Added for logging
from torch.distributions import Categorical

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import EcoTrackEnv

os.makedirs("models/pg", exist_ok=True)
os.makedirs("logs/pg", exist_ok=True) # <--- Create logs directory

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

def run_reinforce(params, run_id):
    env = EcoTrackEnv(render_mode=None)
    
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    policy = ReinforcePolicy(obs_size, act_size, hidden_size=params["hidden"])
    optimizer = optim.Adam(policy.parameters(), lr=params["lr"])
    
    # Training Config
    MAX_EPISODES = 500 
    BATCH_SIZE = params["batch_size"] 
    ENT_COEF = params["ent_coef"] 
    
    # Logging Data
    log_data = [] # List to store per-episode stats
    
    optimizer.zero_grad()
    episodes_in_batch = 0
    
    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        entropies = [] 
        done = False
        
        while not done:
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            probs = policy(state_tensor)
            m = Categorical(probs)
            action = m.sample()
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            
            log_probs.append(m.log_prob(action))
            entropies.append(m.entropy()) 
            rewards.append(reward)
            done = terminated or truncated

        # --- Logging ---
        ep_reward = sum(rewards)
        avg_entropy = torch.stack(entropies).mean().item()
        log_data.append({
            "episode": episode,
            "reward": ep_reward,
            "entropy": avg_entropy
        })

        # --- Policy Update ---
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + params["gamma"] * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R, entropy in zip(log_probs, returns, entropies):
            loss_step = -log_prob * R - (ENT_COEF * entropy)
            policy_loss.append(loss_step)
            
        if len(policy_loss) > 0:
            batch_loss = torch.cat(policy_loss).sum()
            (batch_loss / BATCH_SIZE).backward()
        
        episodes_in_batch += 1
        
        if episodes_in_batch >= BATCH_SIZE:
            optimizer.step()
            optimizer.zero_grad()
            episodes_in_batch = 0
        
    # Save Logs to CSV
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(f"logs/pg/run_{run_id}.csv", index=False)

    # Save Model
    filename = f"models/pg/reinforce_run_{run_id}_lr{params['lr']}_ent{params['ent_coef']}.pth"
    torch.save(policy.state_dict(), filename)
    env.close()
    
    return np.mean([x['reward'] for x in log_data[-50:]])

def train_reinforce_experiments():
    # Experiments Table
    experiments = [
        # Baseline
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5, "hidden": 64, "ent_coef": 0.01}, 
        
        # Exploration Tests (Entropy)
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5, "hidden": 64, "ent_coef": 0.00}, 
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5, "hidden": 64, "ent_coef": 0.1},  
        
        # Batch Size Tests
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 1,  "hidden": 64, "ent_coef": 0.01},
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 20, "hidden": 64, "ent_coef": 0.01},
        
        # Learning Rate Tests
        {"lr": 1e-4, "gamma": 0.99, "batch_size": 5,  "hidden": 64, "ent_coef": 0.01},
        {"lr": 1e-2, "gamma": 0.99, "batch_size": 5,  "hidden": 64, "ent_coef": 0.01},
        
        # Architecture
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5,  "hidden": 128,"ent_coef": 0.01},
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5,  "hidden": 32, "ent_coef": 0.01},
        
        # Gamma Test
        {"lr": 1e-3, "gamma": 0.90, "batch_size": 5,  "hidden": 64, "ent_coef": 0.01}
    ]

    print(f"{'Run':<4} | {'LR':<8} | {'Ent':<5} | {'Bat':<4} | {'Hid':<4} | {'Mean Reward':<12}")
    print("-" * 55)

    for i, p in enumerate(experiments, 1):
        mean_reward = run_reinforce(p, i)
        print(f"{i:<4} | {p['lr']:<8} | {p['ent_coef']:<5} | {p['batch_size']:<4} | {p['hidden']:<4} | {mean_reward:.2f}")

if __name__ == "__main__":
    train_reinforce_experiments()