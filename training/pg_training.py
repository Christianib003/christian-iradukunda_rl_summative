import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.distributions import Categorical
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import EcoTrackEnv

# Ensure directories exist
os.makedirs("models/pg", exist_ok=True)
os.makedirs("models/a2c", exist_ok=True)
os.makedirs("logs/pg", exist_ok=True)
os.makedirs("logs/a2c", exist_ok=True)

# ==========================================
# PART 1: REINFORCE Implementation (Custom)
# ==========================================
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
    log_data = [] 
    
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
        avg_entropy = torch.stack(entropies).mean().item() if entropies else 0
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
        
    # Save Logs
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(f"logs/pg/run_{run_id}.csv", index=False)

    filename = f"models/pg/reinforce_run_{run_id}_lr{params['lr']}_ent{params['ent_coef']}.pth"
    torch.save(policy.state_dict(), filename)
    env.close()
    return np.mean([x['reward'] for x in log_data[-50:]])

def train_reinforce_experiments():
    print("\n>>> Running REINFORCE Experiments...")
    experiments = [
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5, "hidden": 64, "ent_coef": 0.01}, 
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5, "hidden": 64, "ent_coef": 0.00}, 
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5, "hidden": 64, "ent_coef": 0.1},  
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 1,  "hidden": 64, "ent_coef": 0.01},
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 20, "hidden": 64, "ent_coef": 0.01},
        {"lr": 1e-4, "gamma": 0.99, "batch_size": 5,  "hidden": 64, "ent_coef": 0.01},
        {"lr": 1e-2, "gamma": 0.99, "batch_size": 5,  "hidden": 64, "ent_coef": 0.01},
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5,  "hidden": 128,"ent_coef": 0.01},
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 5,  "hidden": 32, "ent_coef": 0.01},
        {"lr": 1e-3, "gamma": 0.90, "batch_size": 5,  "hidden": 64, "ent_coef": 0.01}
    ]
    print(f"{'Run':<4} | {'LR':<8} | {'Ent':<5} | {'Bat':<4} | {'Hid':<4} | {'Mean Reward':<12}")
    print("-" * 55)
    for i, p in enumerate(experiments, 1):
        mean_reward = run_reinforce(p, i)
        print(f"{i:<4} | {p['lr']:<8} | {p['ent_coef']:<5} | {p['batch_size']:<4} | {p['hidden']:<4} | {mean_reward:.2f}")

# ==========================================
# PART 2: A2C Implementation (Stable Baselines 3)
# ==========================================
def train_a2c_experiments():
    print("\n>>> Running A2C Experiments...")
    
    experiments = [
        # Baseline: Balanced Actor (ent) and Critic (vf)
        {"lr": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent": 0.0,  "vf": 0.5, "net": [64, 64]},
        
        # Entropy Tests (Actor Exploration)
        {"lr": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent": 0.01, "vf": 0.5, "net": [64, 64]},
        {"lr": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent": 0.05, "vf": 0.5, "net": [64, 64]},
        
        # Critic Tests (Value Function Coefficient) <--- NEW!
        {"lr": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent": 0.0,  "vf": 0.1, "net": [64, 64]}, # Weak Critic
        {"lr": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent": 0.0,  "vf": 0.9, "net": [64, 64]}, # Strong Critic
        
        # Batch Size (n_steps) Tests
        {"lr": 7e-4, "gamma": 0.99, "n_steps": 20, "ent": 0.0,  "vf": 0.5, "net": [64, 64]},
        
        # Learning Rate Tests
        {"lr": 1e-4, "gamma": 0.99, "n_steps": 5,  "ent": 0.0,  "vf": 0.5, "net": [64, 64]},
        {"lr": 1e-2, "gamma": 0.99, "n_steps": 5,  "ent": 0.0,  "vf": 0.5, "net": [64, 64]},
        
        # Architecture Tests
        {"lr": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent": 0.0,  "vf": 0.5, "net": [32, 32]},
        {"lr": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent": 0.0,  "vf": 0.5, "net": [128, 128]}
    ]

    print(f"{'Run':<4} | {'LR':<8} | {'Ent':<5} | {'VF':<4} | {'Steps':<5} | {'Arch':<10} | {'Mean Reward':<12}")
    print("-" * 75)

    for i, params in enumerate(experiments, 1):
        env = EcoTrackEnv(render_mode=None)
        env = Monitor(env)

        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=params["lr"],
            gamma=params["gamma"],
            n_steps=params["n_steps"],
            ent_coef=params["ent"],
            vf_coef=params["vf"], # <--- Added Value Function Coef
            policy_kwargs={"net_arch": params["net"]},
            verbose=0,
            tensorboard_log="./logs/a2c/"
        )

        model.learn(total_timesteps=50000, tb_log_name=f"run_{i}")
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

        arch_str = "-".join(map(str, params['net']))
        model_name = (
            f"models/a2c/a2c_run_{i}_lr{params['lr']}_ent{params['ent']}_"
            f"vf{params['vf']}_steps{params['n_steps']}"
        )
        model.save(model_name)

        print(f"{i:<4} | {params['lr']:<8} | {params['ent']:<5} | {params['vf']:<4} | {params['n_steps']:<5} | {str(params['net']):<10} | {mean_reward:.2f}")

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PG Agents")
    parser.add_argument("--algo", type=str, choices=["reinforce", "a2c"], default="a2c")
    args = parser.parse_args()
    
    if args.algo == "reinforce":
        train_reinforce_experiments()
    elif args.algo == "a2c":
        train_a2c_experiments()