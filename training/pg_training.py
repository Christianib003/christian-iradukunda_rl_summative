"""
Policy Gradient training script for the EcoTrack environment.

Card 12 – Implement Policy Gradient Training Script

Supports:
- PPO  (Stable-Baselines3)
- A2C  (Stable-Baselines3)
- REINFORCE (custom implementation)

Features:
- Algorithm chosen via --algo (ppo, a2c, reinforce)
- Hyperparameter presets chosen via --config-id (PPO-01..10, A2C-01..10, REIN-01..10)
- Training metrics logged via Monitor into CSV
- Models saved under models/pg/ with algorithm + key hyperparameters in filenames (no timestamps)
- Short evaluation run after training
"""

import os
import argparse
import os
import csv
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import EcoTrackEnv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Predefined Hyperparameter Configs (aligned with hyperparameter_plan.md)
# ---------------------------------------------------------------------------

PPO_PRESETS: Dict[str, Dict[str, Any]] = {
    "PPO-01": dict(
        learning_rate=3e-4,
        gamma=0.99,
        net_arch=[128, 128],
        n_steps=2048,
        batch_size=64,
        clip_range=0.20,
        ent_coef=0.0,
        n_epochs=10,
    ),
    "PPO-02": dict(
        learning_rate=1e-4,
        gamma=0.99,
        net_arch=[128, 128],
        n_steps=2048,
        batch_size=64,
        clip_range=0.20,
        ent_coef=0.01,
        n_epochs=10,
    ),
    "PPO-03": dict(
        learning_rate=1e-3,
        gamma=0.99,
        net_arch=[128, 128],
        n_steps=2048,
        batch_size=64,
        clip_range=0.20,
        ent_coef=0.0,
        n_epochs=10,
    ),
    "PPO-04": dict(
        learning_rate=3e-4,
        gamma=0.95,
        net_arch=[64, 64],
        n_steps=1024,
        batch_size=64,
        clip_range=0.10,
        ent_coef=0.01,
        n_epochs=5,
    ),
    "PPO-05": dict(
        learning_rate=3e-4,
        gamma=0.99,
        net_arch=[256, 256],
        n_steps=4096,
        batch_size=128,
        clip_range=0.20,
        ent_coef=0.0,
        n_epochs=10,
    ),
    "PPO-06": dict(
        learning_rate=3e-4,
        gamma=0.99,
        net_arch=[256, 256],
        n_steps=2048,
        batch_size=256,
        clip_range=0.30,
        ent_coef=0.02,
        n_epochs=10,
    ),
    "PPO-07": dict(
        learning_rate=1e-4,
        gamma=0.99,
        net_arch=[64, 64],
        n_steps=2048,
        batch_size=128,
        clip_range=0.20,
        ent_coef=0.0,
        n_epochs=5,
    ),
    "PPO-08": dict(
        learning_rate=3e-4,
        gamma=0.95,
        net_arch=[128, 128],
        n_steps=1024,
        batch_size=256,
        clip_range=0.20,
        ent_coef=0.01,
        n_epochs=10,
    ),
    "PPO-09": dict(
        learning_rate=1e-3,
        gamma=0.99,
        net_arch=[256, 256],
        n_steps=1024,
        batch_size=64,
        clip_range=0.10,
        ent_coef=0.0,
        n_epochs=5,
    ),
    "PPO-10": dict(
        learning_rate=3e-4,
        gamma=0.99,
        net_arch=[128, 128],
        n_steps=4096,
        batch_size=128,
        clip_range=0.30,
        ent_coef=0.01,
        n_epochs=10,
    ),
}

A2C_PRESETS: Dict[str, Dict[str, Any]] = {
    "A2C-01": dict(
        learning_rate=7e-4,
        gamma=0.99,
        net_arch=[128, 128],
        n_steps=128,
        ent_coef=0.0,
        vf_coef=0.5,
    ),
    "A2C-02": dict(
        learning_rate=3e-4,
        gamma=0.99,
        net_arch=[128, 128],
        n_steps=128,
        ent_coef=0.01,
        vf_coef=0.5,
    ),
    "A2C-03": dict(
        learning_rate=1e-3,
        gamma=0.99,
        net_arch=[128, 128],
        n_steps=128,
        ent_coef=0.0,
        vf_coef=0.5,
    ),
    "A2C-04": dict(
        learning_rate=7e-4,
        gamma=0.95,
        net_arch=[64, 64],
        n_steps=64,
        ent_coef=0.0,
        vf_coef=0.5,
    ),
    "A2C-05": dict(
        learning_rate=7e-4,
        gamma=0.99,
        net_arch=[256, 256],
        n_steps=256,
        ent_coef=0.0,
        vf_coef=0.5,
    ),
    "A2C-06": dict(
        learning_rate=3e-4,
        gamma=0.95,
        net_arch=[256, 256],
        n_steps=256,
        ent_coef=0.02,
        vf_coef=0.5,
    ),
    "A2C-07": dict(
        learning_rate=7e-4,
        gamma=0.99,
        net_arch=[128, 128],
        n_steps=64,
        ent_coef=0.01,
        vf_coef=0.7,
    ),
    "A2C-08": dict(
        learning_rate=1e-3,
        gamma=0.95,
        net_arch=[64, 64],
        n_steps=128,
        ent_coef=0.02,
        vf_coef=0.5,
    ),
    "A2C-09": dict(
        learning_rate=3e-4,
        gamma=0.99,
        net_arch=[256, 256],
        n_steps=64,
        ent_coef=0.0,
        vf_coef=0.7,
    ),
    "A2C-10": dict(
        learning_rate=7e-4,
        gamma=0.95,
        net_arch=[128, 128],
        n_steps=256,
        ent_coef=0.01,
        vf_coef=0.5,
    ),
}

REIN_PRESETS: Dict[str, Dict[str, Any]] = {
    # REINFORCE presets based on the earlier grid:
    "REIN-01": dict(
        learning_rate=1e-3,
        gamma=0.99,
        net_arch=[128, 128],
        batch_episodes=10,
        use_baseline=True,
        ent_coef=0.0,
    ),
    "REIN-02": dict(
        learning_rate=3e-4,
        gamma=0.99,
        net_arch=[128, 128],
        batch_episodes=10,
        use_baseline=True,
        ent_coef=0.01,
    ),
    "REIN-03": dict(
        learning_rate=3e-3,
        gamma=0.99,
        net_arch=[128, 128],
        batch_episodes=10,
        use_baseline=True,
        ent_coef=0.0,
    ),
    "REIN-04": dict(
        learning_rate=1e-3,
        gamma=0.95,
        net_arch=[64, 64],
        batch_episodes=5,
        use_baseline=True,
        ent_coef=0.0,
    ),
    "REIN-05": dict(
        learning_rate=1e-3,
        gamma=0.99,
        net_arch=[256, 256],
        batch_episodes=20,
        use_baseline=True,
        ent_coef=0.0,
    ),
    "REIN-06": dict(
        learning_rate=3e-4,
        gamma=0.95,
        net_arch=[128, 128],
        batch_episodes=20,
        use_baseline=True,
        ent_coef=0.01,
    ),
    "REIN-07": dict(
        learning_rate=1e-3,
        gamma=0.99,
        net_arch=[128, 128],
        batch_episodes=5,
        use_baseline=False,
        ent_coef=0.0,
    ),
    "REIN-08": dict(
        learning_rate=3e-3,
        gamma=0.95,
        net_arch=[64, 64],
        batch_episodes=10,
        use_baseline=False,
        ent_coef=0.01,
    ),
    "REIN-09": dict(
        learning_rate=3e-4,
        gamma=0.99,
        net_arch=[256, 256],
        batch_episodes=10,
        use_baseline=False,
        ent_coef=0.0,
    ),
    "REIN-10": dict(
        learning_rate=1e-3,
        gamma=0.95,
        net_arch=[256, 256],
        batch_episodes=20,
        use_baseline=True,
        ent_coef=0.01,
    ),
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_ecotrack_env(seed: Optional[int] = None, monitor_file: Optional[str] = None) -> Monitor:
    """
    Create a single EcoTrackEnv wrapped in a Monitor for logging.
    """
    env = EcoTrackEnv(
        grid_width=10,
        grid_height=10,
        max_steps=200,
        max_capacity=100.0,
        render_mode=None,
    )
    if seed is not None:
        env.reset(seed=seed)

    return Monitor(env, filename=monitor_file)


# ---------------------------------------------------------------------------
# Preset application
# ---------------------------------------------------------------------------

def apply_preset_to_args(args: argparse.Namespace) -> None:
    """
    If args.config_id is set, override hyperparameters in args using the
    corresponding preset for the chosen algorithm.
    """
    if not args.config_id:
        return

    algo = args.algo.lower()
    config_id = args.config_id.upper()

    if algo == "ppo":
        presets = PPO_PRESETS
    elif algo == "a2c":
        presets = A2C_PRESETS
    elif algo == "reinforce":
        presets = REIN_PRESETS
    else:
        raise ValueError(f"Unsupported algo '{args.algo}' for presets.")

    if config_id not in presets:
        raise ValueError(
            f"Unknown config_id '{args.config_id}' for algo '{args.algo}'. "
            f"Available: {', '.join(presets.keys())}"
        )

    preset = presets[config_id]
    print(f"[PG] Applying preset '{config_id}' for algo {algo.upper()}: {preset}")

    args.learning_rate = preset["learning_rate"]
    args.gamma = preset["gamma"]

    if algo in {"ppo", "a2c"}:
        args.n_steps = preset["n_steps"]
        args.ent_coef = preset["ent_coef"]
    if algo == "ppo":
        args.batch_size = preset["batch_size"]
        args.clip_range = preset["clip_range"]
        args.n_epochs = preset["n_epochs"]
    if algo == "a2c":
        args.vf_coef = preset["vf_coef"]
    if algo == "reinforce":
        args.batch_episodes = preset["batch_episodes"]
        args.use_baseline = preset["use_baseline"]
        args.ent_coef = preset["ent_coef"]

    args.net_arch = preset["net_arch"]
    args.config_id = config_id  # normalized


# ---------------------------------------------------------------------------
# SB3 PPO / A2C training & evaluation
# ---------------------------------------------------------------------------

def train_sb3_pg(args: argparse.Namespace) -> str:
    """
    Train PPO or A2C using Stable-Baselines3.
    """
    algo = args.algo.lower()
    if algo not in {"ppo", "a2c"}:
        raise ValueError("train_sb3_pg called with non-SB3 algo")

    # Timestamp only for log folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = args.run_name or f"{algo}_run"
    if args.config_id:
        base_run_name = f"{base_run_name}_{args.config_id}"
    run_id = f"{base_run_name}_{timestamp}"

    # Logs
    base_log_dir = args.log_dir  # e.g. logs/pg
    run_log_dir = os.path.join(base_log_dir, algo, run_id)
    os.makedirs(run_log_dir, exist_ok=True)

    # Models
    model_dir = args.model_dir or os.path.join("models", "pg")
    os.makedirs(model_dir, exist_ok=True)

    monitor_file = os.path.join(run_log_dir, "monitor.csv")

    def _init_env():
        return make_ecotrack_env(seed=args.seed, monitor_file=monitor_file)

    env = DummyVecEnv([_init_env])

    common_kwargs = dict(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        tensorboard_log=run_log_dir,
        verbose=1,
    )

    policy_kwargs = {}
    if args.net_arch is not None:
        policy_kwargs["net_arch"] = args.net_arch

    if algo == "ppo":
        ModelClass = PPO
        model_kwargs = dict(
            **common_kwargs,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
        )
    else:
        ModelClass = A2C
        model_kwargs = common_kwargs

    print(f"[PG] Starting {algo.upper()} training with parameters:")
    print(f"    run_id: {run_id}")
    if args.config_id:
        print(f"    config_id: {args.config_id}")
    for k, v in model_kwargs.items():
        print(f"    {k}: {v}")
    if policy_kwargs:
        print(f"    policy_kwargs: {policy_kwargs}")

    model = ModelClass(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs or None,
        **model_kwargs,
    )

    print(f"[PG] Training {algo.upper()} for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=10,
        progress_bar=True,
    )

    cfg_tag = args.config_id or "custom"
    if algo == "ppo":
        model_filename = (
            f"ppo_ecotrack_{cfg_tag}"
            f"_lr{args.learning_rate}"
            f"_g{args.gamma}"
            f"_ns{args.n_steps}"
            f"_bs{args.batch_size}"
            f"_clip{args.clip_range}"
            f"_ent{args.ent_coef}"
            f"_ep{args.n_epochs}.zip"
        )
    else:
        model_filename = (
            f"a2c_ecotrack_{cfg_tag}"
            f"_lr{args.learning_rate}"
            f"_g{args.gamma}"
            f"_ns{args.n_steps}"
            f"_ent{args.ent_coef}"
            f"_vf{args.vf_coef}.zip"
        )

    model_path = os.path.join(model_dir, model_filename)
    model.save(model_path)
    print(f"[PG] Saved {algo.upper()} model to: {model_path}")
    print(f"[PG] Episode metrics CSV (Monitor): {monitor_file}")

    env.close()
    return model_path, run_log_dir



def evaluate_sb3_pg(model_path: str, algo: str, n_episodes: int = 5, seed: Optional[int] = 123) -> dict:
    """
    Evaluate a PPO/A2C model for a few episodes and return summary metrics.
    """
    algo = algo.lower()
    if algo == "ppo":
        ModelClass = PPO
    elif algo == "a2c":
        ModelClass = A2C
    else:
        raise ValueError("evaluate_sb3_pg called with non-SB3 algo")

    print(f"[Eval-{algo.upper()}] Loading model from: {model_path}")
    model = ModelClass.load(model_path)

    rewards = []
    lengths = []

    for ep in range(n_episodes):
        env = EcoTrackEnv(
            grid_width=10,
            grid_height=10,
            max_steps=200,
            max_capacity=100.0,
            render_mode=None,
        )
        if seed is not None:
            env.reset(seed=seed + ep)

        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            ep_len += 1

        env.close()
        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(f"[Eval-{algo.upper()}] Episode {ep + 1}: reward={ep_reward:.2f}, length={ep_len}")

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))

    print(
        f"[Eval-{algo.upper()}] Mean reward over {n_episodes} episodes: "
        f"{mean_reward:.2f} ± {std_reward:.2f}"
    )
    print(
        f"[Eval-{algo.upper()}] Mean episode length: "
        f"{mean_length:.1f} ± {std_length:.1f}"
    )

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "std_length": std_length,
    }




# ---------------------------------------------------------------------------
# REINFORCE implementation
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs: np.ndarray):
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), log_prob.squeeze(), entropy.squeeze()


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """
    Compute discounted returns for a single episode.
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train_reinforce(args: argparse.Namespace) -> str:
    """
    Custom REINFORCE training loop.

    total_episodes controls how many episodes to collect.
    batch_episodes controls how many episodes per policy update.
    """
    # Timestamp only for log folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = args.run_name or "reinforce_run"
    if args.config_id:
        base_run_name = f"{base_run_name}_{args.config_id}"
    run_id = f"{base_run_name}_{timestamp}"

    # Logs
    base_log_dir = args.log_dir
    run_log_dir = os.path.join(base_log_dir, "reinforce", run_id)
    os.makedirs(run_log_dir, exist_ok=True)

    model_dir = args.model_dir or os.path.join("models", "pg")
    os.makedirs(model_dir, exist_ok=True)

    monitor_file = os.path.join(run_log_dir, "monitor.csv")

    env = make_ecotrack_env(seed=args.seed, monitor_file=monitor_file)
    obs_space = env.observation_space
    act_space = env.action_space

    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(act_space.n)

    hidden_sizes = args.net_arch or [128, 128]
    policy = PolicyNet(obs_dim, act_dim, hidden_sizes)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    gamma = args.gamma
    batch_episodes = args.batch_episodes
    total_episodes = args.total_episodes
    ent_coef = args.ent_coef
    use_baseline = args.use_baseline

    print("[REINFORCE] Starting training with parameters:")
    print(f"    run_id: {run_id}")
    if args.config_id:
        print(f"    config_id: {args.config_id}")
    print(f"    lr: {args.learning_rate}, gamma: {gamma}")
    print(f"    net_arch: {hidden_sizes}")
    print(f"    batch_episodes: {batch_episodes}, total_episodes: {total_episodes}")
    print(f"    use_baseline: {use_baseline}, ent_coef: {ent_coef}")

    all_returns = []

    # buffers for batch
    batch_log_probs: List[torch.Tensor] = []
    batch_entropies: List[torch.Tensor] = []
    batch_returns: List[float] = []

    for ep in range(1, total_episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False

        ep_rewards: List[float] = []
        ep_log_probs: List[torch.Tensor] = []
        ep_entropies: List[torch.Tensor] = []

        while not (done or truncated):
            action, log_prob, entropy = policy.act(obs)
            obs, reward, done, truncated, info = env.step(action)

            ep_rewards.append(float(reward))
            ep_log_probs.append(log_prob)
            ep_entropies.append(entropy)

        # Compute returns for this episode
        ep_returns = compute_returns(ep_rewards, gamma)
        all_returns.append(sum(ep_rewards))

        # Add to batch
        batch_log_probs.extend(ep_log_probs)
        batch_entropies.extend(ep_entropies)
        batch_returns.extend(ep_returns)

        # Logging
        if ep % 10 == 0:
            avg_ret = np.mean(all_returns[-10:])
            print(f"[REINFORCE] Episode {ep}/{total_episodes} | "
                  f"avg return (last 10) = {avg_ret:.2f}")

        # Perform update every batch_episodes
        if ep % batch_episodes == 0:
            log_probs_tensor = torch.stack(batch_log_probs)
            returns_tensor = torch.tensor(batch_returns, dtype=torch.float32)
            entropies_tensor = torch.stack(batch_entropies)

            # Optional baseline: simple mean-return baseline
            if use_baseline:
                baseline = returns_tensor.mean()
                advantages = returns_tensor - baseline
            else:
                advantages = returns_tensor

            # Normalize advantages to improve stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(advantages * log_probs_tensor).mean()
            entropy_bonus = entropies_tensor.mean()

            loss = policy_loss - ent_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clear batch buffers
            batch_log_probs.clear()
            batch_entropies.clear()
            batch_returns.clear()

    # Quick evaluation with the trained policy
    eval_episodes = args.eval_episodes
    rewards = []
    lengths = []
    for ep in range(eval_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        ep_len = 0
        while not (done or truncated):
            # Use greedy action selection (argmax) at eval time
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(x)
            action = torch.argmax(logits, dim=-1).item()
            obs, reward, done, truncated, info = env.step(action)
            ep_rew += float(reward)
            ep_len += 1
        rewards.append(ep_rew)
        lengths.append(ep_len)
        print(f"[Eval-REINFORCE] Episode {ep + 1}: reward={ep_rew:.2f}, length={ep_len}")

    # --- summary metrics ---
    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    mean_l = float(np.mean(lengths))
    std_l = float(np.std(lengths))

    print(
        f"[Eval-REINFORCE] Mean reward over {eval_episodes} episodes: "
        f"{mean_r:.2f} ± {std_r:.2f}"
    )
    print(
        f"[Eval-REINFORCE] Mean episode length: "
        f"{mean_l:.1f} ± {std_l:.1f}"
    )

    metrics = {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "mean_length": mean_l,
        "std_length": std_l,
    }

    env.close()

    # Save model (state_dict) – file name encodes config + key hypers (no timestamp)
    cfg_tag = args.config_id or "custom"
    model_filename = (
        f"reinforce_ecotrack_{cfg_tag}"
        f"_lr{args.learning_rate}"
        f"_g{gamma}"
        f"_be{batch_episodes}"
        f"_bl{int(use_baseline)}"
        f"_ent{ent_coef}.pt"
    )
    model_path = os.path.join(model_dir, model_filename)
    torch.save(policy.state_dict(), model_path)
    print(f"[REINFORCE] Saved policy state_dict to: {model_path}")
    print(f"[REINFORCE] Episode metrics CSV (Monitor): {monitor_file}")

    # --- append results row for Card 14 ---
    results_path = os.path.join("results", "reinforce_results.csv")
    append_reinforce_results_row(
        results_path=results_path,
        cfg_tag=cfg_tag,
        args=args,
        run_log_dir=run_log_dir,
        model_path=model_path,
        metrics=metrics,
    )

    return model_path



def append_reinforce_results_row(
    results_path: str,
    cfg_tag: str,
    args,
    run_log_dir: str,
    model_path: str,
    metrics: dict,
):
    """
    Append a single summary row for a REINFORCE run to results/reinforce_results.csv.
    Creates the file + header if it does not exist.
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    fieldnames = [
        "config_id",
        "learning_rate",
        "gamma",
        "net_arch",
        "batch_episodes",
        "total_episodes",
        "use_baseline",
        "ent_coef",
        "eval_mean_reward",
        "eval_std_reward",
        "eval_mean_length",
        "eval_std_length",
        "model_path",
        "log_dir",
    ]

    file_exists = os.path.isfile(results_path)

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row = {
            "config_id": cfg_tag,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "net_arch": getattr(args, "net_arch", None),
            "batch_episodes": args.batch_episodes,
            "total_episodes": args.total_episodes,
            "use_baseline": getattr(args, "use_baseline", None),
            "ent_coef": getattr(args, "ent_coef", None),
            "eval_mean_reward": metrics["mean_reward"],
            "eval_std_reward": metrics["std_reward"],
            "eval_mean_length": metrics["mean_length"],
            "eval_std_length": metrics["std_length"],
            "model_path": model_path,
            "log_dir": run_log_dir,
        }

        writer.writerow(row)
        print(f"[REINFORCE] Appended summary row to {results_path}")

def append_pg_results_row(
    results_path: str,
    algo: str,
    cfg_tag: str,
    args,
    run_log_dir: str,
    model_path: str,
    metrics: dict,
):
    """
    Append a summary row for a PPO or A2C run to the given CSV.
    Columns mirror the DQN/REINFORCE style: eval_mean_reward, model_path, etc.
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    fieldnames = [
        "config_id",
        "algo",
        "learning_rate",
        "gamma",
        "net_arch",
        "n_steps",
        "batch_size",
        "clip_range",
        "ent_coef",
        "n_epochs",
        "vf_coef",
        "total_timesteps",
        "eval_mean_reward",
        "eval_std_reward",
        "eval_mean_length",
        "eval_std_length",
        "model_path",
        "log_dir",
    ]

    file_exists = os.path.isfile(results_path)

    # Represent net_arch nicely
    net_arch_val = getattr(args, "net_arch", None)
    if isinstance(net_arch_val, (list, tuple)):
        net_arch_str = "-".join(str(h) for h in net_arch_val)
    else:
        net_arch_str = str(net_arch_val) if net_arch_val is not None else None

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row = {
            "config_id": cfg_tag,
            "algo": algo,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "net_arch": net_arch_str,
            "n_steps": getattr(args, "n_steps", None),
            "batch_size": getattr(args, "batch_size", None),
            "clip_range": getattr(args, "clip_range", None),
            "ent_coef": getattr(args, "ent_coef", None),
            "n_epochs": getattr(args, "n_epochs", None),
            "vf_coef": getattr(args, "vf_coef", None),
            "total_timesteps": getattr(args, "total_timesteps", None),
            "eval_mean_reward": metrics["mean_reward"],
            "eval_std_reward": metrics["std_reward"],
            "eval_mean_length": metrics["mean_length"],
            "eval_std_length": metrics["std_length"],
            "model_path": model_path,
            "log_dir": run_log_dir,
        }

        writer.writerow(row)
        print(f"[PG-{algo.upper()}] Appended summary row to {results_path}")



def append_a2c_results_row(
    results_path: str,
    cfg_tag: str,
    args,
    run_log_dir: str,
    model_path: str,
    metrics: dict,
):
    """
    Append a single summary row for an A2C run to results/a2c_results.csv.
    Creates the file + header if it does not exist.
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    fieldnames = [
        "config_id",
        "learning_rate",
        "gamma",
        "net_arch",
        "n_steps",
        "ent_coef",
        "vf_coef",
        "total_timesteps",
        "eval_mean_reward",
        "eval_std_reward",
        "eval_mean_length",
        "eval_std_length",
        "model_path",
        "log_dir",
    ]

    file_exists = os.path.isfile(results_path)

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row = {
            "config_id": cfg_tag,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "net_arch": getattr(args, "net_arch", None),
            "n_steps": args.n_steps,
            "ent_coef": getattr(args, "ent_coef", None),
            "vf_coef": getattr(args, "vf_coef", None),
            "total_timesteps": args.total_timesteps,
            "eval_mean_reward": metrics["mean_reward"],
            "eval_std_reward": metrics["std_reward"],
            "eval_mean_length": metrics["mean_length"],
            "eval_std_length": metrics["std_length"],
            "model_path": model_path,
            "log_dir": run_log_dir,
        }

        writer.writerow(row)
        print(f"[A2C] Appended summary row to {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO / A2C / REINFORCE on the EcoTrack environment."
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "a2c", "reinforce"],
        required=True,
        help="Algorithm to train: 'ppo', 'a2c', or 'reinforce'.",
    )

    parser.add_argument(
        "--config-id",
        type=str,
        default=None,
        help="Optional preset ID (PPO-01..10, A2C-01..10, REIN-01..10).",
    )

    # Shared
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Logical name for this run (used in log directory).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/pg",
        help="Base directory for logs (TensorBoard + Monitor CSV).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/pg",
        help="Directory to save models (all PG algorithms).",
    )

    # SB3 training timesteps
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50_000,
        help="Total timesteps for PPO/A2C.",
    )

    # REINFORCE episodes
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=200,
        help="Total episodes for REINFORCE training.",
    )
    parser.add_argument(
        "--batch-episodes",
        type=int,
        default=10,
        help="Number of episodes per REINFORCE update.",
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="Use a simple mean-return baseline in REINFORCE.",
    )

    # Common hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--net-arch",
        type=str,
        default=None,
        help="Comma-separated hidden layer sizes, e.g. '64,64' or '128,128'.",
    )

    # PPO/A2C specific
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Rollout length per update (PPO/A2C).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy coefficient (PPO/A2C/REINFORCE).",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda (PPO/A2C).",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function coefficient (A2C/PPO).",
    )

    # PPO-specific
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO batch size.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clipping range.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO epochs per update.",
    )

    # Evaluation
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes after training.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="If set, skip evaluation (mainly for speed).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse net_arch string into list[int], if provided
    if args.net_arch is not None:
        try:
            args.net_arch = [int(x.strip()) for x in args.net_arch.split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid --net-arch value: {args.net_arch}. Use e.g. '64,64'.")

    # Apply preset if requested
    apply_preset_to_args(args)

    if args.algo in {"ppo", "a2c"}:
        # Train
        model_path, run_log_dir = train_sb3_pg(args)

        # Evaluate and log
        if not args.skip_eval:
            metrics = evaluate_sb3_pg(
                model_path,
                algo=args.algo,
                n_episodes=args.eval_episodes,
                seed=args.seed,
            )

            cfg_tag = args.config_id or "custom"
            if args.algo == "ppo":
                results_path = "results/ppo_results.csv"
            else:  # a2c
                results_path = "results/a2c_results.csv"

            append_pg_results_row(
                results_path=results_path,
                algo=args.algo,
                cfg_tag=cfg_tag,
                args=args,
                run_log_dir=run_log_dir,
                model_path=model_path,
                metrics=metrics,
            )

    else:
        # REINFORCE
        model_path = train_reinforce(args)
        # REINFORCE already does its own short eval + logging inside train_reinforce



if __name__ == "__main__":
    main()
