import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EcoTrackEnv(gym.Env):
    """
    Custom Environment for EcoTrack: Smart Waste Collection Routing.
    The agent (truck) must collect waste from bins and avoid overflows.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=10, max_steps=100, n_bins=15, n_high_priority=5):
        super(EcoTrackEnv, self).__init__()

        # --- Configuration ---
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_bins = n_bins
        self.n_high_priority = n_high_priority
        self.render_mode = render_mode
        self.k_nearest = 5  # Number of nearest bins to include in observation

        # --- Action Space ---
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Pick Up, 5: Unload, 6: Wait
        self.action_space = spaces.Discrete(7)

        # --- Observation Space ---
        # Vector size calculation:
        # Agent: [x, y, load_ratio, time_ratio] = 4 values
        # Depot: [dist_to_depot] = 1 value
        # K Nearest Bins: K * [dx, dy, fill_level, is_priority] = 5 * 4 = 20 values
        # Global: [overflow_ratio, high_prio_serviced_ratio] = 2 values
        # Total = 27 values
        obs_size = 4 + 1 + (self.k_nearest * 4) + 2
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # --- State Variables (Initialized in reset) ---
        self.agent_pos = None
        self.agent_load = 0
        self.current_step = 0
        self.bins = []  # Will store bin objects or dicts
        self.depot_pos = None
        
        # Rendering tools
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed)
        
        # TODO: Implement state initialization (Card 4)
        
        # Placeholder observation
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        
        return observation, info

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        # TODO: Implement logic (Card 6, 7, 8, 9)
        
        # Placeholder returns
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Visualizes the environment.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # TODO: Implement Pygame rendering (Card 10)
        pass

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()