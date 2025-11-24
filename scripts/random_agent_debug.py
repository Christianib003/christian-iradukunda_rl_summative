"""
Random agent debug script for the EcoTrack environment.

This script:
- Runs a number of episodes with a random policy
- Prints step-by-step information about actions, rewards, and key state stats
- Helps verify that environment dynamics and reward structure behave as expected
"""

from environment.custom_env import EcoTrackEnv


def run_random_episodes(
    n_episodes: int = 3,
    max_steps: int = 50,
    render: bool = True,
):
    env = EcoTrackEnv(max_steps=max_steps, render_mode="human" if render else None)

    for ep in range(n_episodes):
        print("\n" + "=" * 60)
        print(f"Starting episode {ep + 1}/{n_episodes}")
        print("=" * 60)

        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step_idx = 0

        while not (done or truncated):
            action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Extract detailed reward components if available
            step_components = info.get("step_reward_components", {})
            collect = step_components.get("collect", 0.0)
            serviced = step_components.get("serviced_bonus", 0.0)
            move = step_components.get("move", 0.0)
            wait = step_components.get("wait", 0.0)
            invalid = step_components.get("invalid", 0.0)
            overflow = step_components.get("overflow", 0.0)
            terminal = step_components.get("terminal", 0.0)

            print(
                f"[Ep {ep + 1} | Step {step_idx:02d}] "
                f"Action={action} | "
                f"Reward={reward:+.3f} "
                f"(collect={collect:+.3f}, serviced={serviced:+.3f}, "
                f"move={move:+.3f}, wait={wait:+.3f}, invalid={invalid:+.3f}, "
                f"overflow={overflow:+.3f}, terminal={terminal:+.3f}) | "
                f"Overflows={info.get('overflow_count', 0)} | "
                f"Serviced={info.get('serviced_bins_count', 0)} "
                f"(high-prio={info.get('serviced_high_priority_count', 0)})"
            )

            if render:
                env.render()

            step_idx += 1

        print(
            f"Episode {ep + 1} finished. "
            f"Total reward = {total_reward:.3f}, "
            f"steps = {step_idx}, "
            f"done={done}, truncated={truncated}"
        )

    env.close()


if __name__ == "__main__":
    # You can adjust these values as needed during debugging.
    run_random_episodes(n_episodes=3, max_steps=50, render=True)
