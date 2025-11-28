"""
Generate plots for EcoTrack RL experiments.

Card 18 – Generate Plots for Results & Stability

This script:
- Loads per-algorithm results CSVs:
    results/dqn_results.csv
    results/reinforce_results.csv
    results/a2c_results.csv
    results/ppo_results.csv
- Picks the best configuration for each algorithm (by eval_mean_reward).
- Uses the corresponding log_dir to load Monitor logs (monitor.csv) and
  build reward curves + stability metrics.
- Computes an approximate "episodes to converge" metric.
- Loads results/final_comparison.csv for generalization bar plot.

Outputs:
- figures/reward_curves_best_configs.png
- figures/reward_stability_best_configs.png
- figures/episodes_to_converge.png
- figures/final_generalization_bar.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

ALGO_CONFIG = {
    "dqn": {
        "label": "DQN",
        "results_csv": "results/dqn_results.csv",
    },
    "reinforce": {
        "label": "REINFORCE",
        "results_csv": "results/reinforce_results.csv",
    },
    "a2c": {
        "label": "A2C",
        "results_csv": "results/a2c_results.csv",
    },
    "ppo": {
        "label": "PPO",
        "results_csv": "results/ppo_results.csv",
    },
}

FINAL_COMPARISON_CSV = "results/final_comparison.csv"
FIG_DIR = "figures"
MONITOR_FILENAME = "monitor.csv"

os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_best_row(results_csv: str) -> pd.Series:
    """
    Load a results CSV and return the row with the best eval_mean_reward.
    Assumes the CSV has a column 'eval_mean_reward' and 'log_dir'.
    """
    df = pd.read_csv(results_csv)
    if "eval_mean_reward" not in df.columns:
        raise ValueError(f"{results_csv} has no 'eval_mean_reward' column.")
    if "log_dir" not in df.columns:
        raise ValueError(f"{results_csv} has no 'log_dir' column.")

    best = df.sort_values("eval_mean_reward", ascending=False).iloc[0]
    return best


def load_monitor_rewards(log_dir: str) -> np.ndarray:
    """
    Given a run log_dir, load its monitor.csv and return per-episode rewards.

    SB3 Monitor CSV has columns r (reward), l (length), t (time),
    with a commented first line.
    """
    monitor_path = os.path.join(log_dir, MONITOR_FILENAME)
    if not os.path.exists(monitor_path):
        raise FileNotFoundError(f"Monitor file not found: {monitor_path}")

    df = pd.read_csv(monitor_path, comment="#")
    if "r" not in df.columns:
        raise ValueError(f"Monitor CSV {monitor_path} has no 'r' column.")
    rewards = df["r"].to_numpy()
    return rewards


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def moving_std(x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling standard deviation using a simple sliding window.
    """
    if len(x) < window:
        return np.zeros_like(x, dtype=float)
    stds = []
    for i in range(len(x) - window + 1):
        stds.append(np.std(x[i:i + window]))
    return np.array(stds)


def episodes_to_converge(rewards: np.ndarray,
                         window: int = 50,
                         frac_of_max: float = 0.9) -> int:
    """
    Approximate convergence episode:
    - Compute moving average with given window.
    - Let max_avg = max(moving_avg).
    - Convergence episode = first episode where moving_avg >= frac_of_max * max_avg.
    If not reached, return total number of episodes.

    Returns episode index in 1-based indexing (human-friendly).
    """
    if len(rewards) < window:
        return len(rewards)

    ma = moving_average(rewards, window=window)
    max_avg = ma.max()
    target = frac_of_max * max_avg

    indices = np.where(ma >= target)[0]
    if len(indices) == 0:
        return len(rewards)
    # ma[0] corresponds to episodes [1..window]
    first_idx = indices[0]
    # approximate "episode number" when this moving average window ends
    return first_idx + window  # 1-based-ish


# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------

def plot_reward_curves(best_runs: dict, window: int = 20) -> None:
    """
    Plot smoothed reward curves over episodes for each algorithm's best config.
    """
    plt.figure(figsize=(8, 5))

    for algo_key, info in best_runs.items():
        label = info["label"]
        rewards = info["rewards"]
        ma = moving_average(rewards, window=window)
        episodes = np.arange(1, len(ma) + 1)

        plt.plot(episodes, ma, label=label)

    plt.xlabel(f"Episode (smoothed with window={window})")
    plt.ylabel("Episode reward (moving average)")
    plt.title("Training reward curves – best config per algorithm")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(FIG_DIR, "reward_curves_best_configs.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved reward curves to {out_path}")


def plot_reward_stability(best_runs: dict, window: int = 20) -> None:
    """
    Plot rolling std of episode reward as a simple stability indicator.
    Lower std over time ≈ more stable training.
    """
    plt.figure(figsize=(8, 5))

    for algo_key, info in best_runs.items():
        label = info["label"]
        rewards = info["rewards"]
        std_curve = moving_std(rewards, window=window)
        episodes = np.arange(1, len(std_curve) + 1)

        plt.plot(episodes, std_curve, label=label)

    plt.xlabel(f"Episode (window={window})")
    plt.ylabel("Rolling std of episode reward")
    plt.title("Training stability – rolling reward variability")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(FIG_DIR, "reward_stability_best_configs.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved stability plot to {out_path}")


def plot_episodes_to_converge(best_runs: dict,
                              window: int = 50,
                              frac_of_max: float = 0.9) -> None:
    """
    Bar plot of approximate episodes-to-converge per algorithm.
    """
    algos = []
    conv_eps = []

    for algo_key, info in best_runs.items():
        label = info["label"]
        rewards = info["rewards"]
        ep_conv = episodes_to_converge(rewards, window=window, frac_of_max=frac_of_max)
        algos.append(label)
        conv_eps.append(ep_conv)

    plt.figure(figsize=(6, 4))
    x = np.arange(len(algos))
    plt.bar(x, conv_eps)
    plt.xticks(x, algos)
    plt.ylabel("Episodes to converge")
    plt.title(f"Episodes to converge (window={window}, frac={frac_of_max:.2f})")
    plt.grid(axis="y", alpha=0.3)

    out_path = os.path.join(FIG_DIR, "episodes_to_converge.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved episodes-to-converge plot to {out_path}")


def plot_final_generalization_bar(final_csv: str = FINAL_COMPARISON_CSV) -> None:
    """
    Use results/final_comparison.csv (from evaluate_best_models) to plot
    a bar chart of mean reward per algorithm on the evaluation scenarios.
    """
    if not os.path.exists(final_csv):
        print(f"[PLOTS] final comparison CSV not found: {final_csv} – skipping bar chart.")
        return

    df = pd.read_csv(final_csv)
    if "algo" not in df.columns or "mean_reward" not in df.columns:
        print(f"[PLOTS] final_comparison.csv missing 'algo' or 'mean_reward' – skipping bar chart.")
        return

    # If multiple rows per algo exist, average them (should normally be 1 per algo)
    df_grouped = df.groupby("algo", as_index=False)["mean_reward"].mean()

    plt.figure(figsize=(6, 4))
    x = np.arange(len(df_grouped))
    plt.bar(x, df_grouped["mean_reward"])
    plt.xticks(x, df_grouped["algo"])
    plt.ylabel("Mean reward (eval)")
    plt.title("Final generalization performance – mean reward per algorithm")
    plt.grid(axis="y", alpha=0.3)

    out_path = os.path.join(FIG_DIR, "final_generalization_bar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved final generalization bar chart to {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # 1) Load best runs and their monitor rewards
    best_runs = {}

    for algo_key, cfg in ALGO_CONFIG.items():
        label = cfg["label"]
        results_csv = cfg["results_csv"]

        if not os.path.exists(results_csv):
            print(f"[WARN] Results CSV not found for {label}: {results_csv} – skipping.")
            continue

        best_row = load_best_row(results_csv)
        log_dir = best_row["log_dir"]

        try:
            rewards = load_monitor_rewards(log_dir)
        except Exception as e:
            print(f"[WARN] Could not load monitor rewards for {label} "
                  f"(log_dir={log_dir}): {e}")
            continue

        best_runs[algo_key] = {
            "label": label,
            "best_row": best_row,
            "rewards": rewards,
        }

        print(f"[INFO] Best {label} config: config_id={best_row['config_id']}, "
              f"eval_mean_reward={best_row['eval_mean_reward']:.2f}, "
              f"log_dir={log_dir}")

    if not best_runs:
        print("[ERROR] No best runs loaded – nothing to plot.")
        return

    # 2) Reward curves (learning curves)
    plot_reward_curves(best_runs, window=20)

    # 3) Stability (rolling std of rewards)
    plot_reward_stability(best_runs, window=20)

    # 4) Episodes to converge
    plot_episodes_to_converge(best_runs, window=50, frac_of_max=0.9)

    # 5) Final generalization bar chart (from evaluate_best_models)
    plot_final_generalization_bar(FINAL_COMPARISON_CSV)


if __name__ == "__main__":
    main()
