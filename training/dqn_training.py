"""
DQN training script for the EcoTrack environment.

This script:
- Creates a Gymnasium-compatible EcoTrackEnv
- Wraps it for Stable-Baselines3
- Trains a DQN agent
- Saves the trained model to models/dqn/
- Prints a quick evaluation summary

Usage (from project root):

    python -m training.dqn_training \
        --total-timesteps 50000 \
        --run-name dqn_v1

You can tweak hyperparameters via CLI flags; later we'll use this
to run multiple configurations for the assignment.
"""

import os
import argparse
from datetime import datetime
from typing import Optional

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import EcoTrackEnv


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_ecotrack_env(seed: Optional[int] = None) -> Monitor:
    """
    Create a single EcoTrackEnv wrapped in a Monitor for SB3.

    Args:
        seed: optional random seed for reproducibility.

    Returns:
        A Monitor-wrapped Gymnasium environment.
    """
    env = EcoTrackEnv(
        grid_width=10,
        grid_height=10,
        max_steps=200,
        max_capacity=100.0,
        render_mode=None,  # no GUI during training
    )
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


# ---------------------------------------------------------------------------
# Training / evaluation utilities
# ---------------------------------------------------------------------------
def train_dqn(args: argparse.Namespace) -> str:
    """
    Train a DQN agent on EcoTrackEnv using Stable-Baselines3.

    Args:
        args: parsed command-line arguments.

    Returns:
        Path to the saved model file.
    """
    # Ensure output directories exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Build vectorized environment (single env here)
    def _init_env():
        return make_ecotrack_env(seed=args.seed)

    env = DummyVecEnv([_init_env])

    # Compose a run id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_name}_{timestamp}" if args.run_name else timestamp

    # DQN hyperparameters (can be tuned later)
    dqn_kwargs = dict(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        target_update_interval=args.target_update_interval,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        verbose=1,
    )

    print("[DQN] Starting training with parameters:")
    for k, v in dqn_kwargs.items():
        print(f"    {k}: {v}")

    # Create the model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=args.log_dir,
        **dqn_kwargs,
    )

    # Train
    print(f"[DQN] Training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=10,
        progress_bar=True,
    )

    # Save model
    model_path = os.path.join(args.model_dir, f"dqn_ecotrack_{run_id}.zip")
    model.save(model_path)
    print(f"[DQN] Saved model to: {model_path}")

    env.close()
    return model_path


def evaluate_dqn(model_path: str, n_episodes: int = 5, seed: Optional[int] = 123) -> None:
    """
    Quick evaluation of a trained DQN model on EcoTrackEnv.

    Runs a small number of episodes without rendering and prints
    average reward and episode length.

    Args:
        model_path: path to the saved DQN model.
        n_episodes: number of evaluation episodes.
        seed: optional seed for env.
    """
    from stable_baselines3 import DQN  # local import to avoid circular issues

    print(f"[DQN] Loading model from: {model_path}")
    model = DQN.load(model_path)

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
            # SB3 expects obs without batch dim; model.predict adds it internally
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            ep_len += 1

        env.close()
        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(f"[Eval] Episode {ep + 1}: reward={ep_reward:.2f}, length={ep_len}")

    print(
        f"[Eval] Mean reward over {n_episodes} episodes: "
        f"{np.mean(rewards):.2f} ± {np.std(rewards):.2f}"
    )
    print(
        f"[Eval] Mean episode length: "
        f"{np.mean(lengths):.1f} ± {np.std(lengths):.1f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on the EcoTrack environment."
    )

    # Core training settings
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50_000,
        help="Total number of training timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for env and model.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="dqn_run",
        help="Logical name for this run (used in model filename).",
    )

    # Paths
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/dqn",
        help="Directory for tensorboard logs.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/dqn",
        help="Directory to save DQN models.",
    )

    # DQN hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the DQN optimizer.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for Q-network updates.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50_000,
        help="Replay buffer size.",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.2,
        help="Fraction of training during which exploration rate is annealed.",
    )
    parser.add_argument(
        "--exploration-final-eps",
        type=float,
        default=0.05,
        help="Final value of epsilon after exploration annealing.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1000,
        help="Frequency (in steps) for target network updates.",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=4,
        help="Frequency (in steps) of training updates.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Number of gradient steps after each rollout.",
    )

    # Evaluation
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate after training.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="If set, skip evaluation after training.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_path = train_dqn(args)

    if not args.skip_eval:
        evaluate_dqn(model_path, n_episodes=args.eval_episodes, seed=args.seed)


if __name__ == "__main__":
    main()
