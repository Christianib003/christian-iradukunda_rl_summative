import os
import ast
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import DQN, PPO, A2C

from environment.custom_env import EcoTrackEnv
from training.pg_training import PolicyNet  # reuse the same network as REINFORCE



def make_eval_env(seed: int) -> EcoTrackEnv:
    env = EcoTrackEnv(
        grid_width=10,
        grid_height=10,
        max_steps=200,
        max_capacity=100.0,
        render_mode=None,
    )
    env.reset(seed=seed)
    return env


def get_first_key(info: Dict[str, Any], keys: List[str], default=0) -> Any:
    if info is None:
        return default
    for k in keys:
        if k in info and info[k] is not None:
            return info[k]
    return default


def load_best_row(csv_path: str, algo_name: str) -> Dict[str, Any]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{algo_name}: results file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "eval_mean_reward" not in df.columns:
        raise ValueError(f"{algo_name}: 'eval_mean_reward' column missing in {csv_path}")

    df = df.dropna(subset=["eval_mean_reward"]).copy()
    df["eval_mean_reward"] = df["eval_mean_reward"].astype(float)

    best_idx = df["eval_mean_reward"].idxmax()
    best_row = df.loc[best_idx].to_dict()
    return best_row


def eval_sb3_model(
    model,
    algo_name: str,
    n_episodes: int = 20,
    base_seed: int = 123,
) -> Dict[str, float]:
    rewards = []
    lengths = []
    overflow_counts = []
    high_prio_serviced = []

    for ep in range(n_episodes):
        env = make_eval_env(seed=base_seed + ep)
        obs, info = env.reset()
        done = False
        truncated = False

        ep_rew = 0.0
        ep_len = 0
        last_info = info

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_rew += float(reward)
            ep_len += 1
            last_info = info

        env.close()
        rewards.append(ep_rew)
        lengths.append(ep_len)

        overflow = get_first_key(
            last_info,
            keys=[
                "overflow_count",
                "overflows",
                "overflow_events",
            ],
            default=0,
        )
        high_serv = get_first_key(
            last_info,
            keys=[
                "high_priority_serviced",
                "serviced_high_priority_bins",
            ],
            default=0,
        )

        overflow_counts.append(overflow)
        high_prio_serviced.append(high_serv)

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(lengths)),
        "std_steps": float(np.std(lengths)),
        "mean_overflow_count": float(np.mean(overflow_counts)),
        "mean_high_prio_serviced": float(np.mean(high_prio_serviced)),
    }

    print(
        f"[Eval-{algo_name.upper()}] "
        f"mean_reward={metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}, "
        f"mean_steps={metrics['mean_steps']:.1f} ± {metrics['std_steps']:.1f}, "
        f"mean_overflow={metrics['mean_overflow_count']:.2f}, "
        f"mean_high_prio_serviced={metrics['mean_high_prio_serviced']:.2f}"
    )

    return metrics


def eval_reinforce_model(
    row: Dict[str, Any],
    n_episodes: int = 20,
    base_seed: int = 321,
) -> Dict[str, float]:
    """
    Evaluate REINFORCE policy stored as a PolicyNet state_dict.

    'row' is the best-row dict from reinforce_results.csv,
    and must contain 'model_path' and 'net_arch'.
    """
    model_path = row["model_path"]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"REINFORCE model_path does not exist: {model_path}")

    # Parse hidden sizes from the stored net_arch string
    hidden_sizes = [128, 128]
    if "net_arch" in row and isinstance(row["net_arch"], str):
        try:
            hidden_sizes = ast.literal_eval(row["net_arch"])
        except Exception:
            # fallback if parsing fails
            pass

    # Create a temp env to infer obs_dim & act_dim
    tmp_env = make_eval_env(seed=base_seed)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(act_space.n)
    tmp_env.close()

    policy = PolicyNet(obs_dim, act_dim, hidden_sizes)
    state_dict = torch.load(model_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()

    rewards = []
    lengths = []
    overflow_counts = []
    high_prio_serviced = []

    for ep in range(n_episodes):
        env = make_eval_env(seed=base_seed + ep)
        obs, info = env.reset()
        done = False
        truncated = False

        ep_rew = 0.0
        ep_len = 0
        last_info = info

        while not (done or truncated):
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = policy(x)
                action = torch.argmax(logits, dim=-1).item()

            obs, reward, done, truncated, info = env.step(int(action))
            ep_rew += float(reward)
            ep_len += 1
            last_info = info

        env.close()
        rewards.append(ep_rew)
        lengths.append(ep_len)

        overflow = get_first_key(
            last_info,
            keys=[
                "overflow_count",
                "overflows",
                "overflow_events",
            ],
            default=0,
        )
        high_serv = get_first_key(
            last_info,
            keys=[
                "high_priority_serviced",
                "serviced_high_priority_bins",
            ],
            default=0,
        )

        overflow_counts.append(overflow)
        high_prio_serviced.append(high_serv)

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(lengths)),
        "std_steps": float(np.std(lengths)),
        "mean_overflow_count": float(np.mean(overflow_counts)),
        "mean_high_prio_serviced": float(np.mean(high_prio_serviced)),
    }

    print(
        f"[Eval-REINFORCE] "
        f"mean_reward={metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}, "
        f"mean_steps={metrics['mean_steps']:.1f} ± {metrics['std_steps']:.1f}, "
        f"mean_overflow={metrics['mean_overflow_count']:.2f}, "
        f"mean_high_prio_serviced={metrics['mean_high_prio_serviced']:.2f}"
    )

    return metrics



def main():
    os.makedirs("results", exist_ok=True)

    # Load best rows from each results CSV
    dqn_row = load_best_row("results/dqn_results.csv", "dqn")
    rein_row = load_best_row("results/reinforce_results.csv", "reinforce")
    a2c_row = load_best_row("results/a2c_results.csv", "a2c")
    ppo_row = load_best_row("results/ppo_results.csv", "ppo")

    # Load models
    # DQN
    dqn_model_path = dqn_row["model_path"]
    if not os.path.isfile(dqn_model_path):
        raise FileNotFoundError(f"DQN model_path does not exist: {dqn_model_path}")
    dqn_model = DQN.load(dqn_model_path)

    # A2C
    a2c_model_path = a2c_row["model_path"]
    if not os.path.isfile(a2c_model_path):
        raise FileNotFoundError(f"A2C model_path does not exist: {a2c_model_path}")
    a2c_model = A2C.load(a2c_model_path)

    # PPO
    ppo_model_path = ppo_row["model_path"]
    if not os.path.isfile(ppo_model_path):
        raise FileNotFoundError(f"PPO model_path does not exist: {ppo_model_path}")
    ppo_model = PPO.load(ppo_model_path)

    # Evaluate each algorithm
    N_EPISODES = 20

    dqn_metrics = eval_sb3_model(dqn_model, "dqn", n_episodes=N_EPISODES, base_seed=1000)
    a2c_metrics = eval_sb3_model(a2c_model, "a2c", n_episodes=N_EPISODES, base_seed=2000)
    ppo_metrics = eval_sb3_model(ppo_model, "ppo", n_episodes=N_EPISODES, base_seed=3000)
    rein_metrics = eval_reinforce_model(rein_row, n_episodes=N_EPISODES, base_seed=4000)

    # Build comparison table
    rows = []

    def add_row(algo_name: str, row_dict: Dict[str, Any], metrics: Dict[str, float]):
        rows.append({
            "algo": algo_name,
            "config_id": row_dict.get("config_id", None),
            "model_path": row_dict.get("model_path", None),
            "eval_episodes": N_EPISODES,
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "mean_steps": metrics["mean_steps"],
            "std_steps": metrics["std_steps"],
            "mean_overflow_count": metrics["mean_overflow_count"],
            "mean_high_prio_serviced": metrics["mean_high_prio_serviced"],
        })

    add_row("dqn", dqn_row, dqn_metrics)
    add_row("reinforce", rein_row, rein_metrics)
    add_row("a2c", a2c_row, a2c_metrics)
    add_row("ppo", ppo_row, ppo_metrics)

    df = pd.DataFrame(rows)

    # Pick overall best (highest mean_reward)
    best_idx = df["mean_reward"].idxmax()
    df["is_overall_best"] = False
    df.loc[best_idx, "is_overall_best"] = True

    best_row = df.loc[best_idx]
    print("\n=== Final Comparison Summary ===")
    print(df)
    print("\n=== Overall Best Model ===")
    print(
        f"Algo: {best_row['algo']}, config: {best_row['config_id']}, "
        f"mean_reward={best_row['mean_reward']:.2f}, "
        f"mean_overflow={best_row['mean_overflow_count']:.2f}, "
        f"mean_high_prio_serviced={best_row['mean_high_prio_serviced']:.2f}"
    )

    # Save to CSV
    out_path = "results/final_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Final] Wrote comparison table to: {out_path}")


if __name__ == "__main__":
    main()
