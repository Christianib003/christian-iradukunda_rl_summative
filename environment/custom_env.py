import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EcoTrackEnv(gym.Env):
    """
    Custom environment for the EcoTrack smart waste collection problem.

    This is a placeholder skeleton; full dynamics and reward logic
    will be implemented in later tasks.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        # TODO: replace with actual dimensions and observation layout
        self.grid_width = 10
        self.grid_height = 10

        # Discrete actions: up, down, left, right, pick_up, unload, wait
        self.action_space = spaces.Discrete(7)

        # Placeholder observation space: 10-dim vector
        # This will be updated to the true state representation.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.state = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # TODO: initialize truck position, bin states, time, etc.
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return self.state, info

    def step(self, action: int):
        # TODO: implement environment dynamics and reward logic
        assert self.action_space.contains(action), "Invalid action"

        obs = self.state
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        # TODO: delegate to rendering.py or simple text rendering
        if self.render_mode == "human":
            print("EcoTrackEnv state:", self.state)

    def close(self):
        pass
