"""
Random agent video generator for the EcoTrack environment.

This script:
- Runs a single episode with a random policy
- Uses the pygame-based GUI for visualization
- Captures each frame from the window
- Saves a video file (MP4 or GIF) for the assignment

Usage:
    python -m scripts.random_agent_video

The output file will be stored in the `figures/` folder.
"""

import os
from typing import List

import numpy as np
import imageio.v2 as imageio
import pygame

from environment.custom_env import EcoTrackEnv


def record_random_episode(
    max_steps: int = 100,
    fps: int = 10,
    out_dir: str = "figures",
    out_filename: str = "random_agent_demo.mp4",
    use_gif: bool = False,
) -> str:
    """
    Run a single episode with random actions and record frames.

    Args:
        max_steps: maximum number of environment steps to record
        fps: frames per second for the output video
        out_dir: directory to save the video file
        out_filename: name of the output file (mp4 or gif)
        use_gif: if True, saves a GIF instead of MP4

    Returns:
        Full path to the saved video file.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Ensure correct extension if using GIF
    if use_gif and not out_filename.lower().endswith(".gif"):
        out_filename = os.path.splitext(out_filename)[0] + ".gif"
    if not use_gif and not out_filename.lower().endswith(".mp4"):
        out_filename = os.path.splitext(out_filename)[0] + ".mp4"

    out_path = os.path.join(out_dir, out_filename)

    env = EcoTrackEnv(max_steps=max_steps, render_mode="human")

    frames: List[np.ndarray] = []

    obs, info = env.reset()
    done = False
    truncated = False
    step_idx = 0

    print(f"[EcoTrack] Recording random episode to: {out_path}")

    while step_idx < max_steps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Render the current frame in pygame
        env.render()

        # Capture the window surface as an RGB array
        surface = pygame.display.get_surface()
        if surface is None:
            raise RuntimeError("No pygame display surface found for capture.")

        frame = pygame.surfarray.array3d(surface)
        # pygame.surfarray returns (width, height, 3); imageio expects (height, width, 3)
        frame = np.transpose(frame, (1, 0, 2))

        frames.append(frame)

        step_idx += 1

    env.close()

    # Save video file
    if use_gif:
        # GIF: easier to guarantee without ffmpeg
        imageio.mimsave(out_path, frames, fps=fps)
    else:
        # MP4: requires imageio-ffmpeg / ffmpeg installed
        imageio.mimsave(out_path, frames, fps=fps)

    print(f"[EcoTrack] Saved video with {len(frames)} frames at {fps} FPS.")
    return out_path


if __name__ == "__main__":
    # Choose whether to make MP4 or GIF here
    # Set use_gif=True if ffmpeg isn't available on the grading machine.
    record_random_episode(
        max_steps=200,
        fps=10,
        out_dir="figures",
        out_filename="random_agent_demo.mp4",
        use_gif=False,  # change to True if you prefer GIF
    )
