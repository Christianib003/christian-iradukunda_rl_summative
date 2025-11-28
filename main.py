import argparse
import time
import os
import ast
import pandas as pd
import numpy as np

from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import EcoTrackEnv
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """
    Same architecture as used in training.pg_training for REINFORCE.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes):
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


def load_best_row_from_final(algo: str, csv_path: str = "results/final_comparison.csv"):
    """
    Load the best row for a given algo ('dqn','reinforce','a2c','ppo') or overall ('auto')
    from results/final_comparison.csv.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run scripts/evaluate_best_models.py first, "
            "or pass --model-path manually."
        )

    df = pd.read_csv(csv_path)

    if algo == "auto":
        if "is_overall_best" in df.columns and df["is_overall_best"].any():
            row = df[df["is_overall_best"]].iloc[0]
        else:
            # Fall back to highest mean_reward across all algos
            row = df.loc[df["mean_reward"].idxmax()]
    else:
        sub = df[df["algo"] == algo]
        if sub.empty:
            raise ValueError(f"No rows for algo='{algo}' found in {csv_path}")
        # Pick config with highest mean_reward for that algo
        row = sub.loc[sub["mean_reward"].idxmax()]

    return row


def load_reinforce_net_arch(config_id: str, csv_path: str = "results/reinforce_results.csv"):
    """
    Given a REINFORCE config_id, look up its net_arch in reinforce_results.csv.
    Returns a list of ints, falling back to [128, 128] if lookup/parsing fails.
    """
    default_arch = [128, 128]
    if not os.path.isfile(csv_path):
        return default_arch

    df = pd.read_csv(csv_path)
    sub = df[df["config_id"] == config_id]
    if sub.empty or "net_arch" not in sub.columns:
        return default_arch

    raw = sub.iloc[0]["net_arch"]
    if isinstance(raw, str):
        try:
            arch = ast.literal_eval(raw)
            if isinstance(arch, (list, tuple)):
                return [int(x) for x in arch]
        except Exception:
            return default_arch
    return default_arch


def make_env(render: bool = True, seed: int | None = None):
    """
    Create an EcoTrackEnv for the demo.

    render = True  -> render_mode='human'
    render = False -> render_mode=None (no GUI)
    """
    render_mode = "human" if render else None
    env = EcoTrackEnv(
        grid_width=10,
        grid_height=10,
        max_steps=200,
        max_capacity=100.0,
        render_mode=render_mode,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env



def run_episodes(algo: str,
                 model,
                 episodes: int = 5,
                 fps: float = 4.0,
                 base_seed: int = 123,
                 policy_net: PolicyNet = None):
    """
    Run several evaluation episodes with rendering.
    For SB3 algos (dqn/a2c/ppo), `model` is the SB3 model.
    For REINFORCE, `model` is None and `policy_net` is used instead.
    """
    step_delay = 1.0 / fps if fps > 0 else 0.0

    algo = algo.lower()
    print(f"[MAIN] Running {algo.upper()} for {episodes} episodes at {fps:.1f} FPS")

    for ep in range(1, episodes + 1):
        seed = base_seed + ep
        env = make_env(render=(fps > 0), seed=seed)

        # reset env (with seed) and render initial state
        obs, info = env.reset(seed=seed)
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        print(f"[MAIN] Episode {ep}/{episodes} (seed={seed}) started...")

        # show initial state if rendering
        if step_delay > 0:
            env.render()
            time.sleep(step_delay)

        while not (done or truncated):
            if algo in {"dqn", "ppo", "a2c"}:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            elif algo == "reinforce":
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy_net(x)
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                raise ValueError(f"Unsupported algo '{algo}' in run_episodes")

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1

            if step_delay > 0:
                env.render()        # 🔹 this actually updates the window every step
                time.sleep(step_delay)

        print(f"[MAIN] Episode {ep} finished | reward={ep_reward:.2f}, steps={steps}")
        env.close()

    print("[MAIN] All episodes completed.")


    print("[MAIN] All episodes completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entry point to visualize the best EcoTrack RL policy."
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["auto", "dqn", "reinforce", "a2c", "ppo"],
        default="auto",
        help="Which algorithm to load. 'auto' picks the overall best from final_comparison.csv.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit model path. If provided, overrides final_comparison.csv.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to render.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Approximate frames per second for rendering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed; each episode adds its index.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results/final_comparison.csv",
        help="Path to final_comparison.csv.",
    )
    parser.add_argument(
        "--reinforce-results",
        type=str,
        default="results/reinforce_results.csv",
        help="Path to reinforce_results.csv (for net_arch lookup).",
    )
    parser.add_argument(
        "--reinforce-net-arch",
        type=str,
        default=None,
        help="Override REINFORCE hidden sizes, e.g. '64,64'. "
             "If not set, will be inferred from reinforce_results.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Decide which model to load
    if args.model_path is not None:
        if args.algo == "auto":
            raise ValueError(
                "If you pass --model-path, you must also specify --algo (not 'auto')."
            )
        algo = args.algo.lower()
        model_path = args.model_path
        config_id = None
        print(f"[MAIN] Using explicit model path for {algo.upper()}: {model_path}")
    else:
        # Use final_comparison.csv to pick best model
        row = load_best_row_from_final(args.algo.lower(), args.results_csv)
        algo = row["algo"]
        model_path = row["model_path"]
        config_id = row.get("config_id", None)
        print(
            f"[MAIN] Selected best {algo.upper()} model from {args.results_csv}: "
            f"config_id={config_id}, path={model_path}"
        )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Check that you ran the training/evaluation scripts and that paths in "
            "results/*.csv are correct."
        )

    # Load the appropriate model
    algo = algo.lower()
    sb3_model = None
    reinforce_policy = None

    if algo == "dqn":
        print(f"[MAIN] Loading DQN model from {model_path}")
        sb3_model = DQN.load(model_path)
    elif algo == "ppo":
        print(f"[MAIN] Loading PPO model from {model_path}")
        sb3_model = PPO.load(model_path)
    elif algo == "a2c":
        print(f"[MAIN] Loading A2C model from {model_path}")
        sb3_model = A2C.load(model_path)
    elif algo == "reinforce":
        # For REINFORCE, recreate PolicyNet and load its state_dict
        if args.reinforce_net_arch is not None:
            hidden_sizes = [int(x.strip()) for x in args.reinforce_net_arch.split(",") if x.strip()]
        else:
            if config_id is None:
                raise ValueError(
                    "REINFORCE selected but config_id is unknown. "
                    "Either run with --algo reinforce and --model-path AND "
                    "--reinforce-net-arch, or use final_comparison.csv generated by "
                    "scripts/evaluate_best_models.py."
                )
            hidden_sizes = load_reinforce_net_arch(
                config_id=config_id,
                csv_path=args.reinforce_results,
            )

        # We need obs_dim/act_dim; create a dummy env for that
        dummy_env = EcoTrackEnv(
            grid_width=10,
            grid_height=10,
            max_steps=200,
            max_capacity=100.0,
            render_mode=None,
        )
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        dummy_env.close()

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(act_space.n)

        print(
            f"[MAIN] Reconstructing REINFORCE policy with obs_dim={obs_dim}, "
            f"act_dim={act_dim}, net_arch={hidden_sizes}"
        )
        reinforce_policy = PolicyNet(obs_dim, act_dim, hidden_sizes)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        reinforce_policy.load_state_dict(state_dict)
        reinforce_policy.eval()
    else:
        raise ValueError(f"Unsupported algo '{algo}'. Must be one of dqn/ppo/a2c/reinforce/auto.")

    # Run episodes with rendering
    run_episodes(
        algo=algo,
        model=sb3_model,
        episodes=args.episodes,
        fps=args.fps,
        base_seed=args.seed,
        policy_net=reinforce_policy,
    )


if __name__ == "__main__":
    main()
