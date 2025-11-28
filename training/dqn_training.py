import os
import argparse
import csv
import os
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import EcoTrackEnv



DQN_PRESETS: Dict[str, Dict[str, Any]] = {
    "DQN-01": dict(
        learning_rate=1e-3,
        gamma=0.99,
        net_arch=[128, 128],
        buffer_size=50_000,
        batch_size=64,
        exploration_fraction=0.20,
        exploration_final_eps=0.05,
    ),
    "DQN-02": dict(
        learning_rate=3e-4,
        gamma=0.99,
        net_arch=[128, 128],
        buffer_size=50_000,
        batch_size=64,
        exploration_fraction=0.30,
        exploration_final_eps=0.05,
    ),
    "DQN-03": dict(
        learning_rate=3e-3,
        gamma=0.99,
        net_arch=[128, 128],
        buffer_size=50_000,
        batch_size=64,
        exploration_fraction=0.20,
        exploration_final_eps=0.05,
    ),
    "DQN-04": dict(
        learning_rate=1e-3,
        gamma=0.95,
        net_arch=[64, 64],
        buffer_size=20_000,
        batch_size=32,
        exploration_fraction=0.25,
        exploration_final_eps=0.10,
    ),
    "DQN-05": dict(
        learning_rate=1e-3,
        gamma=0.995,
        net_arch=[256, 256],
        buffer_size=100_000,
        batch_size=128,
        exploration_fraction=0.20,
        exploration_final_eps=0.05,
    ),
    "DQN-06": dict(
        learning_rate=3e-4,
        gamma=0.995,
        net_arch=[256, 256],
        buffer_size=100_000,
        batch_size=64,
        exploration_fraction=0.30,
        exploration_final_eps=0.05,
    ),
    "DQN-07": dict(
        learning_rate=1e-3,
        gamma=0.99,
        net_arch=[64, 64],
        buffer_size=50_000,
        batch_size=32,
        exploration_fraction=0.10,
        exploration_final_eps=0.05,
    ),
    "DQN-08": dict(
        learning_rate=1e-3,
        gamma=0.99,
        net_arch=[256, 256],
        buffer_size=50_000,
        batch_size=128,
        exploration_fraction=0.30,
        exploration_final_eps=0.10,
    ),
    "DQN-09": dict(
        learning_rate=3e-4,
        gamma=0.95,
        net_arch=[128, 128],
        buffer_size=100_000,
        batch_size=64,
        exploration_fraction=0.15,
        exploration_final_eps=0.05,
    ),
    "DQN-10": dict(
        learning_rate=3e-3,
        gamma=0.99,
        net_arch=[64, 64],
        buffer_size=20_000,
        batch_size=128,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
    ),
}



def make_ecotrack_env(seed: Optional[int] = None, monitor_file: Optional[str] = None) -> Monitor:
    env = EcoTrackEnv(
        grid_width=10,
        grid_height=10,
        max_steps=200,
        max_capacity=100.0,
        render_mode=None,  # no GUI during training
    )
    if seed is not None:
        env.reset(seed=seed)

    # Monitor logs: episode reward, length, timesteps → CSV
    return Monitor(env, filename=monitor_file)


def apply_preset_to_args(args: argparse.Namespace) -> None:
    if not args.config_id:
        return

    config_id = args.config_id.upper()
    if config_id not in DQN_PRESETS:
        raise ValueError(
            f"Unknown config_id '{args.config_id}'. "
            f"Available: {', '.join(DQN_PRESETS.keys())}"
        )

    preset = DQN_PRESETS[config_id]
    print(f"[DQN] Applying preset '{config_id}': {preset}")

    args.learning_rate = preset["learning_rate"]
    args.gamma = preset["gamma"]
    args.buffer_size = preset["buffer_size"]
    args.batch_size = preset["batch_size"]
    args.exploration_fraction = preset["exploration_fraction"]
    args.exploration_final_eps = preset["exploration_final_eps"]
    args.net_arch = preset["net_arch"]
    args.config_id = config_id  # normalized


def train_dqn(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_run_name = args.run_name or "dqn_run"
    if args.config_id:
        base_run_name = f"{base_run_name}_{args.config_id}"

    run_id = f"{base_run_name}_{timestamp}"

    base_log_dir = args.log_dir
    run_log_dir = os.path.join(base_log_dir, run_id)
    os.makedirs(run_log_dir, exist_ok=True)

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    monitor_file = os.path.join(run_log_dir, "monitor.csv")

    def _init_env():
        return make_ecotrack_env(seed=args.seed, monitor_file=monitor_file)

    env = DummyVecEnv([_init_env])

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

    policy_kwargs = {}
    if args.net_arch is not None:
        policy_kwargs["net_arch"] = args.net_arch

    print("[DQN] Starting training with parameters:")
    print(f"    run_id: {run_id}")
    if args.config_id:
        print(f"    config_id: {args.config_id}")
    for k, v in dqn_kwargs.items():
        print(f"    {k}: {v}")
    if policy_kwargs:
        print(f"    policy_kwargs: {policy_kwargs}")

    # Create the model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=run_log_dir,
        policy_kwargs=policy_kwargs or None,
        **dqn_kwargs,
    )

    # Train
    print(f"[DQN] Training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=10,
        progress_bar=True,
    )

    cfg_tag = args.config_id or "custom"
    model_filename = (
        f"dqn_ecotrack_{cfg_tag}"
        f"_lr{args.learning_rate}"
        f"_g{args.gamma}"
        f"_bs{args.batch_size}"
        f"_buf{args.buffer_size}.zip"
    )
    model_path = os.path.join(model_dir, model_filename)
    model.save(model_path)
    print(f"[DQN] Saved model to: {model_path}")
    print(f"[DQN] Episode metrics CSV (Monitor): {monitor_file}")

    env.close()
    return model_path, run_log_dir


def evaluate_dqn(model_path: str, n_episodes: int = 5, seed: int = 123):
    print(f"[Eval] Loading model from: {model_path}")
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
        print(f"[Eval] Episode {ep + 1}: reward={ep_reward:.2f}, length={ep_len}")

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    mean_l = float(np.mean(lengths))
    std_l = float(np.std(lengths))

    print(f"[Eval] Mean reward over {n_episodes} episodes: {mean_r:.2f} ± {std_r:.2f}")
    print(f"[Eval] Mean episode length: {mean_l:.1f} ± {std_l:.1f}")

    return {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "mean_length": mean_l,
        "std_length": std_l,
    }
    
    
def append_dqn_results_row(
    results_path: str,
    cfg_tag: str,
    args,
    run_log_dir: str,
    model_path: str,
    metrics: dict,
    ):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    fieldnames = [
        "config_id",
        "learning_rate",
        "gamma",
        "batch_size",
        "buffer_size",
        "target_update_interval",
        "exploration_fraction",
        "exploration_final_eps",
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
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "target_update_interval": args.target_update_interval,
            "exploration_fraction": args.exploration_fraction,
            "exploration_final_eps": args.exploration_final_eps,
            "total_timesteps": args.total_timesteps,
            "eval_mean_reward": metrics["mean_reward"],
            "eval_std_reward": metrics["std_reward"],
            "eval_mean_length": metrics["mean_length"],
            "eval_std_length": metrics["std_length"],
            "model_path": model_path,
            "log_dir": run_log_dir,
        }

        writer.writerow(row)
        print(f"[DQN] Appended summary row to {results_path}")
  


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on the EcoTrack environment."
    )

    parser.add_argument(
        "--config-id",
        type=str,
        default=None,
        help="Predefined DQN config ID (e.g. DQN-01, DQN-02). Overrides matching hyperparameters.",
    )

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
        help="Logical name for this run (used in directory and filenames).",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/dqn",
        help="Base directory for logs (TensorBoard + Monitor CSV).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/dqn",
        help="Directory to save DQN models.",
    )

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

    # Network architecture
    parser.add_argument(
        "--net-arch",
        type=str,
        default=None,
        help="Comma-separated hidden layer sizes for MLP, e.g. '64,64' or '128,128'. "
             "If not set and no preset is used, SB3 defaults are used.",
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

    apply_preset_to_args(args)

    if isinstance(args.net_arch, str) and args.net_arch is not None:
        layers = []
        for h in args.net_arch.split(","):
            h = h.strip()
            if h:
                layers.append(int(h))
        args.net_arch = layers if layers else None

    model_path, run_log_dir = train_dqn(args)

    if not args.skip_eval:
        metrics = evaluate_dqn(
            model_path,
            n_episodes=args.eval_episodes,
            seed=args.seed,
        )

        results_path = os.path.join("results", "dqn_results.csv")
        cfg_tag = args.config_id or "custom"

        append_dqn_results_row(
            results_path=results_path,
            cfg_tag=cfg_tag,
            args=args,
            run_log_dir=run_log_dir,
            model_path=model_path,
            metrics=metrics,
        )




if __name__ == "__main__":
    main()
