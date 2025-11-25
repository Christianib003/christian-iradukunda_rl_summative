# scripts/dqn_agent_ui.py

import argparse
import time
import os

from stable_baselines3 import DQN
from environment.custom_env import EcoTrackEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained DQN agent in the EcoTrack UI (Pygame)."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained DQN .zip model file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Number of agent decisions per second (lower = slower, more visible).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed for env resets.",
    )
    return parser.parse_args()


def run_ui(model_path: str, episodes: int, fps: float, seed: int):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[UI] Loading model from: {model_path}")
    model = DQN.load(model_path)

    # Step delay based on desired FPS
    step_delay = 1.0 / fps if fps > 0 else 0.0

    for ep in range(episodes):
        print(f"\n[UI] Starting episode {ep + 1}/{episodes}")

        # IMPORTANT: use render_mode="human" for Pygame window
        env = EcoTrackEnv(
            grid_width=10,
            grid_height=10,
            max_steps=200,
            max_capacity=100.0,
            render_mode="human",
        )

        # Gymnasium-style reset
        obs, info = env.reset(seed=seed + ep)

        done = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0

        while not (done or truncated):
            # Let the model choose an action
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            ep_len += 1

            # Render one frame
            env.render()

            # Slow down so you can see what’s happening
            if step_delay > 0:
                time.sleep(step_delay)

        print(f"[UI] Episode {ep + 1} finished. Reward={ep_reward:.2f}, length={ep_len}")
        env.close()

    print("[UI] All episodes finished.")


def main():
    args = parse_args()
    run_ui(
        model_path=args.model_path,
        episodes=args.episodes,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
