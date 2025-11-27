import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import EcoTrackEnv

# Ensure directories exist
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)

def train_dqn():
    # UPDATED Experiments List
    experiments = [
        # Baseline
        {"lr": 1e-3, "gamma": 0.99, "buffer": 10000, "batch": 64,  "explore": 0.1, "net_arch": [64, 64]},
        
        # Exploration Tests
        {"lr": 1e-3, "gamma": 0.99, "buffer": 10000, "batch": 64,  "explore": 0.5, "net_arch": [64, 64]},
        {"lr": 1e-3, "gamma": 0.99, "buffer": 10000, "batch": 64,  "explore": 0.05,"net_arch": [64, 64]},
        
        # Discount Factor Tests
        {"lr": 1e-3, "gamma": 0.90, "buffer": 10000, "batch": 64,  "explore": 0.1, "net_arch": [64, 64]},
        {"lr": 1e-3, "gamma": 0.999,"buffer": 10000, "batch": 64,  "explore": 0.1, "net_arch": [64, 64]},
        
        # Buffer & Batch Tests
        {"lr": 1e-4, "gamma": 0.99, "buffer": 50000, "batch": 32,  "explore": 0.1, "net_arch": [64, 64]},
        {"lr": 1e-3, "gamma": 0.99, "buffer": 5000,  "batch": 128, "explore": 0.2, "net_arch": [64, 64]},
        
        # Network Architecture Tests
        {"lr": 5e-4, "gamma": 0.99, "buffer": 10000, "batch": 64,  "explore": 0.1, "net_arch": [32, 32]},
        {"lr": 5e-4, "gamma": 0.99, "buffer": 10000, "batch": 64,  "explore": 0.1, "net_arch": [128, 128]},
        
        # The "Kitchen Sink"
        {"lr": 2e-4, "gamma": 0.99, "buffer": 20000, "batch": 64,  "explore": 0.2, "net_arch": [128, 128]}
    ]

    print(f"{'Run':<4} | {'LR':<7} | {'Gam':<5} | {'Buf':<6} | {'Bat':<4} | {'Exp':<4} | {'Arch':<10} | {'Mean Reward':<12}")
    print("-" * 85)

    for i, params in enumerate(experiments, 1):
        env = EcoTrackEnv(render_mode=None)
        env = Monitor(env)

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params["lr"],
            gamma=params["gamma"],
            buffer_size=params["buffer"],
            batch_size=params["batch"],
            exploration_fraction=params["explore"],
            policy_kwargs={"net_arch": params["net_arch"]},
            verbose=0,
            tensorboard_log="./logs/dqn/"
        )

        # Train - NOW 50,000 STEPS
        model.learn(total_timesteps=50000, tb_log_name=f"run_{i}")

        # Evaluate
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        # Generate Descriptive Filename
        arch_str = "-".join(map(str, params['net_arch']))
        model_filename = (
            f"models/dqn/dqn_lr{params['lr']}_gam{params['gamma']}_"
            f"buf{params['buffer']}_bat{params['batch']}_"
            f"exp{params['explore']}_arch{arch_str}"
        )
        
        model.save(model_filename)

        # Report
        print(f"{i:<4} | {params['lr']:<7} | {params['gamma']:<5} | {params['buffer']:<6} | {params['batch']:<4} | {params['explore']:<4} | {str(params['net_arch']):<10} | {mean_reward:.2f}")

        env.close()

if __name__ == "__main__":
    train_dqn()