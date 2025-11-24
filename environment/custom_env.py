import math
from typing import Optional, Tuple, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EcoTrackEnv(gym.Env):
    """
    EcoTrack – Smart Waste Collection Environment

    A custom Gymnasium environment where an autonomous garbage truck operates
    on a 2D grid city, collects trash from bins (normal + high-priority),
    unloads at a depot, and tries to avoid overflow events while working
    within a fixed time horizon.

    This file implements the *skeleton*:
    - Action and observation spaces
    - State representation
    - Basic step/reset structure

    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    # Number of nearest bins to encode in the observation
    K_NEAREST_BINS = 5

    def __init__(
        self,
        grid_width: int = 10,
        grid_height: int = 10,
        max_steps: int = 200,
        max_capacity: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # --- Core environment parameters ---
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_steps = max_steps
        self.max_capacity = max_capacity
        self.render_mode = render_mode

        # Will be set by _build_map()
        self.depot_pos: Tuple[int, int] = (0, 0)
        self.bin_positions: List[Tuple[int, int]] = []
        self.bin_priority: np.ndarray | None = None  # shape (num_bins,)
        self.total_bins: int = 0
        self.total_high_priority_bins: int = 0

        # --- Internal state variables (episode-specific) ---
        self.truck_pos: Tuple[int, int] = (0, 0)
        self.truck_load: float = 0.0
        self.t_step: int = 0

        self.bin_fill: np.ndarray | None = None  # shape (num_bins,)
        self.overflow_count: int = 0
        self.serviced_bins_count: int = 0
        self.serviced_high_priority_count: int = 0

        # RNG will be set by Gymnasium's seeding in reset()
        self.np_random: np.random.Generator | None = None

        # --- Define action space ---
        # 0: up, 1: down, 2: left, 3: right, 4: pick_up, 5: unload, 6: wait
        self.action_space = spaces.Discrete(7)

        # --- Define observation space ---
        # As per design: 9 global features + 5 * 4 bin features = 29
        obs_dim = 9 + self.K_NEAREST_BINS * 4
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Build a simple default map layout (bins + depot positions)
        self._build_map()

    # ----------------------------------------------------------------------
    # Map / layout helpers
    # ----------------------------------------------------------------------
    def _build_map(self) -> None:
        """
        Define the static layout of the grid:
        - Depot position
        - Locations of bins
        - Which bins are high-priority

        For now we use a simple hard-coded layout.
        Later, this can be randomized or made more complex if desired.
        """
        # Example: depot at (0, 0)
        self.depot_pos = (0, 0)

        # Example bin placement:
        # A few normal bins and a few high-priority bins scattered in the grid.
        bin_positions: List[Tuple[int, int]] = [
            (2, 2),
            (3, 5),
            (5, 3),
            (7, 7),
            (8, 2),
            (1, 8),
            (4, 6),
        ]
        # Mark some as high-priority (e.g. markets/hospitals)
        # 1 = high-priority, 0 = normal
        bin_priority_flags = np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int32)

        assert len(bin_positions) == bin_priority_flags.shape[0], (
            "bin_positions and bin_priority size mismatch"
        )

        self.bin_positions = bin_positions
        self.bin_priority = bin_priority_flags
        self.total_bins = len(self.bin_positions)
        self.total_high_priority_bins = int(self.bin_priority.sum())

    # ----------------------------------------------------------------------
    # Gymnasium API: reset and step
    # ----------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Reset the environment to an initial state.

        Returns:
            observation (np.ndarray): initial observation vector
            info (dict): additional metadata (empty for now)
        """
        super().reset(seed=seed)
        # Gymnasium's seeding mechanism sets self.np_random
        if self.np_random is None:
            # Fallback if not set; this should not usually be necessary
            self.np_random = np.random.default_rng(seed)

        # --- Reset episode state ---
        self.truck_pos = self.depot_pos
        self.truck_load = 0.0
        self.t_step = 0

        # Initialize bin fills with some random values (placeholder)
        # Actual ranges / logic will be refined in the dynamics card.
        low_init = 0.2 * self.max_capacity
        high_init = 0.6 * self.max_capacity
        self.bin_fill = self.np_random.uniform(
            low=low_init, high=high_init, size=(self.total_bins,)
        )

        self.overflow_count = 0
        self.serviced_bins_count = 0
        self.serviced_high_priority_count = 0

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        """
        Apply one environment step given the selected action.

        This skeleton version:
        - Validates the action,
        - Increments time,
        - Returns the current observation with zero reward,
        - Ends the episode when max_steps is reached.

        Full movement, bin dynamics, rewards, and overflow logic will be
        implemented in the next task.
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # TODO: Implement full movement / pickup / unload / wait behaviour here.
        # For now, we just advance time and keep the state fixed.

        self.t_step += 1

        # Placeholder reward and termination logic
        reward = 0.0
        terminated = False
        truncated = False

        if self.t_step >= self.max_steps:
            # End of the day
            terminated = True

        obs = self._get_obs()

        info: Dict[str, Any] = {}
        return obs, float(reward), terminated, truncated, info

    # ----------------------------------------------------------------------
    # Observation construction
    # ----------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """
        Construct the 29-dimensional observation vector:

        - 9 global features
        - 5 nearest bins × (dx, dy, fill, priority) = 20
        """
        if self.bin_fill is None:
            # Safety default; should not happen if reset() was called properly
            self.bin_fill = np.zeros(self.total_bins, dtype=np.float32)

        # --- Global / agent-centric features (9) ---
        x, y = self.truck_pos

        truck_x_norm = x / (self.grid_width - 1)
        truck_y_norm = y / (self.grid_height - 1)
        load_ratio = self.truck_load / self.max_capacity
        time_ratio = self.t_step / self.max_steps

        depot_x, depot_y = self.depot_pos
        depot_dx = depot_x - x
        depot_dy = depot_y - y

        # Normalize distances to [-1, 1] (max distance is grid_width/height)
        depot_dx_norm = np.clip(depot_dx / (self.grid_width - 1), -1.0, 1.0)
        depot_dy_norm = np.clip(depot_dy / (self.grid_height - 1), -1.0, 1.0)

        overflow_ratio = (
            self.overflow_count / self.total_bins if self.total_bins > 0 else 0.0
        )
        high_prio_serviced_ratio = (
            self.serviced_high_priority_count / self.total_high_priority_bins
            if self.total_high_priority_bins > 0
            else 0.0
        )
        serviced_bins_ratio = (
            self.serviced_bins_count / self.total_bins if self.total_bins > 0 else 0.0
        )

        global_feats = np.array(
            [
                truck_x_norm,
                truck_y_norm,
                load_ratio,
                time_ratio,
                depot_dx_norm,
                depot_dy_norm,
                overflow_ratio,
                high_prio_serviced_ratio,
                serviced_bins_ratio,
            ],
            dtype=np.float32,
        )

        # --- Local bin features for K nearest bins ---
        # Compute distances from truck to each bin
        bin_features: List[float] = []

        distances = []
        for idx, (bx, by) in enumerate(self.bin_positions):
            dist = abs(bx - x) + abs(by - y)  # Manhattan distance
            distances.append((dist, idx))

        # Sort by distance and take K nearest
        distances.sort(key=lambda t: t[0])
        k = min(self.K_NEAREST_BINS, len(distances))
        selected_indices = [distances[i][1] for i in range(k)]

        for idx in selected_indices:
            bx, by = self.bin_positions[idx]
            fill = float(self.bin_fill[idx])
            prio = float(self.bin_priority[idx])

            dx = bx - x
            dy = by - y
            # Normalize dx, dy to [-1, 1]
            dx_norm = np.clip(dx / (self.grid_width - 1), -1.0, 1.0)
            dy_norm = np.clip(dy / (self.grid_height - 1), -1.0, 1.0)
            fill_norm = np.clip(fill / self.max_capacity, 0.0, 1.0)

            bin_features.extend([dx_norm, dy_norm, fill_norm, prio])

        # If fewer than K bins, pad with zeros
        missing_bins = self.K_NEAREST_BINS - k
        if missing_bins > 0:
            bin_features.extend([0.0] * missing_bins * 4)

        bin_feats_arr = np.array(bin_features, dtype=np.float32)

        obs = np.concatenate([global_feats, bin_feats_arr], axis=0)

        # Safety check – should always be the correct dimension
        assert obs.shape == self.observation_space.shape

        return obs

    # ----------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------
    def render(self):
        """
        Placeholder render method.

        Later, this will delegate to a dedicated renderer in `rendering.py`
        for a 2D visualization (pygame/OpenGL).

        For now, it prints a minimal text representation if render_mode='human'.
        """
        if self.render_mode == "human":
            print(
                f"Step: {self.t_step} | Pos: {self.truck_pos} | "
                f"Load: {self.truck_load:.1f} | Overflows: {self.overflow_count}"
            )

    def close(self):
        """Clean up any rendering resources (if used)."""
        # Will be used when we add a graphical renderer
        pass
