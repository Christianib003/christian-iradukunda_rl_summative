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

        # Reward & penalty coefficients (can be tuned)
        self.alpha_collect = 0.05          # reward per unit of collected waste
        self.R_serviced_normal = 2.0       # bonus when normal bin is serviced
        self.R_serviced_high = 5.0         # bonus when high-priority bin is serviced
        self.R_all_high_prio = 20.0        # bonus if all high-priority bins serviced

        self.C_move = 0.05                 # cost per movement
        self.C_wait = 0.10                 # cost for WAIT action
        self.C_invalid = 0.50              # cost for invalid/unproductive actions
        self.C_overflow_normal = 5.0       # penalty for normal bin overflow
        self.C_overflow_high = 10.0        # penalty for high-priority bin overflow

        self.beta_terminal = 5.0           # terminal shaping coef * serviced_bins_ratio

        # Overflow & bin thresholds
        self.overflow_threshold = self.max_capacity  # >100% considered overflow
        self.max_overflows = 5                       # truncate episode if exceeded
        self.serviced_threshold_ratio = 0.1          # <= 10% considered serviced

        # Bin fill dynamics (per-step increments)
        self.normal_fill_range = (1.0, 3.0)          # units per step for normal bins
        self.high_fill_range = (2.0, 5.0)            # units per step for high-priority bins
        self.max_bin_fill = self.max_capacity * 1.5  # clamp to avoid runaway values

        # Map-related attributes (set by _build_map)
        self.depot_pos: Tuple[int, int] = (0, 0)
        self.bin_positions: List[Tuple[int, int]] = []
        self.bin_priority: np.ndarray | None = None  # shape (num_bins,)
        self.total_bins: int = 0
        self.total_high_priority_bins: int = 0
        self.blocked_cells: set[Tuple[int, int]] = set()  # optional

        # --- Internal state variables (episode-specific) ---
        self.truck_pos: Tuple[int, int] = (0, 0)
        self.truck_load: float = 0.0
        self.t_step: int = 0

        self.bin_fill: np.ndarray | None = None  # shape (num_bins,)
        self.overflow_count: int = 0
        self.serviced_bins_count: int = 0
        self.serviced_high_priority_count: int = 0

        # Per-bin flags
        self.serviced_bins_mask: np.ndarray | None = None  # shape (num_bins,), bool
        self.overflowed_mask: np.ndarray | None = None     # shape (num_bins,), bool

        # RNG (set by Gymnasium's seeding logic in reset)
        self.np_random: np.random.Generator | None = None

        # --- Define action space ---
        # 0: up, 1: down, 2: left, 3: right, 4: pick_up, 5: unload, 6: wait
        self.action_space = spaces.Discrete(7)

        # --- Define observation space ---
        # 9 global features + 5 * 4 bin features = 29
        obs_dim = 9 + self.K_NEAREST_BINS * 4
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Build static map (bins + depot positions)
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
        """
        # Depot at (0, 0)
        self.depot_pos = (0, 0)

        # Example bin placement
        bin_positions: List[Tuple[int, int]] = [
            (2, 2),
            (3, 5),
            (5, 3),
            (7, 7),
            (8, 2),
            (1, 8),
            (4, 6),
        ]
        # 1 = high-priority, 0 = normal
        bin_priority_flags = np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int32)

        assert len(bin_positions) == bin_priority_flags.shape[0], (
            "bin_positions and bin_priority size mismatch"
        )

        self.bin_positions = bin_positions
        self.bin_priority = bin_priority_flags
        self.total_bins = len(self.bin_positions)
        self.total_high_priority_bins = int(self.bin_priority.sum())

        # No blocked cells for now (can be extended later)
        self.blocked_cells = set()

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
            info (dict): additional metadata
        """
        super().reset(seed=seed)

        # Ensure we have a RNG
        if self.np_random is None:
            # Gymnasium registers its own RNG; fallback for safety
            self.np_random = np.random.default_rng(seed)

        # --- Reset episode state ---
        self.truck_pos = self.depot_pos
        self.truck_load = 0.0
        self.t_step = 0

        # Initialize bin fills in [20%, 60%] of capacity
        low_init = 0.2 * self.max_capacity
        high_init = 0.6 * self.max_capacity
        self.bin_fill = self.np_random.uniform(
            low=low_init, high=high_init, size=(self.total_bins,)
        )

        self.overflow_count = 0
        self.serviced_bins_count = 0
        self.serviced_high_priority_count = 0

        self.serviced_bins_mask = np.zeros(self.total_bins, dtype=bool)
        self.overflowed_mask = np.zeros(self.total_bins, dtype=bool)

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        """
        Apply one environment step given the selected action.

        Implements:
        - Movement and invalid moves
        - PICK_UP, UNLOAD, WAIT actions
        - Bin fill dynamics & overflow penalties
        - Reward calculation and termination conditions
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if self.bin_fill is None:
            raise RuntimeError("Environment must be reset() before step().")

        # Track reward components for debugging
        r_collect = 0.0
        r_serviced_bonus = 0.0
        r_move = 0.0
        r_wait = 0.0
        r_invalid = 0.0
        r_overflow = 0.0
        r_terminal = 0.0

        # ------------------------------------------------------------------
        # 1. Apply chosen action
        # ------------------------------------------------------------------
        x, y = self.truck_pos
        new_x, new_y = x, y

        # Movement actions
        if action in [0, 1, 2, 3]:
            if action == 0:       # MOVE_UP
                new_y = y - 1
            elif action == 1:     # MOVE_DOWN
                new_y = y + 1
            elif action == 2:     # MOVE_LEFT
                new_x = x - 1
            elif action == 3:     # MOVE_RIGHT
                new_x = x + 1

            # Check if move is valid (within grid & not blocked)
            if (
                0 <= new_x < self.grid_width
                and 0 <= new_y < self.grid_height
                and (new_x, new_y) not in self.blocked_cells
            ):
                # Valid move
                self.truck_pos = (new_x, new_y)
                r_move -= self.C_move
            else:
                # Invalid move (hit boundary or blocked cell)
                r_invalid -= self.C_invalid

        elif action == 4:
            # PICK_UP
            bin_idx = self._bin_index_at_position(self.truck_pos)

            if bin_idx is None:
                # No bin here
                r_invalid -= self.C_invalid
            elif self.truck_load >= self.max_capacity:
                # No capacity
                r_invalid -= self.C_invalid
            else:
                # Valid pickup
                remaining_capacity = self.max_capacity - self.truck_load
                current_fill = self.bin_fill[bin_idx]
                collectible = min(remaining_capacity, current_fill)
                if collectible <= 0:
                    # Nothing useful to collect
                    r_invalid -= self.C_invalid
                else:
                    # Apply collection
                    self.truck_load += collectible
                    self.bin_fill[bin_idx] = max(0.0, current_fill - collectible)
                    r_collect += self.alpha_collect * collectible

                    # Check if bin now considered serviced (below threshold)
                    serviced_threshold = self.serviced_threshold_ratio * self.max_capacity
                    if (
                        self.bin_fill[bin_idx] <= serviced_threshold
                        and not self.serviced_bins_mask[bin_idx]
                    ):
                        self.serviced_bins_mask[bin_idx] = True
                        self.serviced_bins_count += 1
                        if self.bin_priority[bin_idx] == 1:
                            self.serviced_high_priority_count += 1
                            r_serviced_bonus += self.R_serviced_high
                        else:
                            r_serviced_bonus += self.R_serviced_normal

        elif action == 5:
            # UNLOAD
            if self.truck_pos == self.depot_pos:
                # Valid unload
                self.truck_load = 0.0
                # Optional: small positive reward could be added here,
                # but for now we keep it neutral. The main benefit is
                # freeing capacity to collect more.
            else:
                # Invalid unload attempt
                r_invalid -= self.C_invalid

        elif action == 6:
            # WAIT
            r_wait -= self.C_wait

        # ------------------------------------------------------------------
        # 2. Advance time
        # ------------------------------------------------------------------
        self.t_step += 1

        # ------------------------------------------------------------------
        # 3. Update bin fills & check overflow
        # ------------------------------------------------------------------
        for i in range(self.total_bins):
            # Increase bin fill based on type
            if self.bin_priority[i] == 1:
                low, high = self.high_fill_range
            else:
                low, high = self.normal_fill_range

            increment = float(self.np_random.uniform(low=low, high=high))
            self.bin_fill[i] += increment
            # Clamp to max_bin_fill for numeric stability
            if self.bin_fill[i] > self.max_bin_fill:
                self.bin_fill[i] = self.max_bin_fill

            # Overflow check
            if (
                self.bin_fill[i] > self.overflow_threshold
                and not self.overflowed_mask[i]
            ):
                self.overflowed_mask[i] = True
                self.overflow_count += 1

                if self.bin_priority[i] == 1:
                    r_overflow -= self.C_overflow_high
                else:
                    r_overflow -= self.C_overflow_normal

        # ------------------------------------------------------------------
        # 4. Check termination / truncation
        # ------------------------------------------------------------------
        terminated = False
        truncated = False

        # End of "day"
        if self.t_step >= self.max_steps:
            terminated = True

        # Catastrophic overflow condition
        if self.overflow_count > self.max_overflows:
            # Treat this as truncation (episode ended early for bad performance)
            truncated = True

        # Terminal shaping reward
        if terminated or truncated:
            serviced_bins_ratio = (
                self.serviced_bins_count / self.total_bins
                if self.total_bins > 0
                else 0.0
            )
            r_terminal += self.beta_terminal * serviced_bins_ratio

            # Bonus if all high-priority bins have been serviced
            if (
                self.total_high_priority_bins > 0
                and self.serviced_high_priority_count == self.total_high_priority_bins
            ):
                r_terminal += self.R_all_high_prio

        # ------------------------------------------------------------------
        # 5. Build observation & total reward
        # ------------------------------------------------------------------
        obs = self._get_obs()
        total_reward = (
            r_collect
            + r_serviced_bonus
            + r_move
            + r_wait
            + r_invalid
            + r_overflow
            + r_terminal
        )

        info: Dict[str, Any] = {
            "step_reward_components": {
                "collect": r_collect,
                "serviced_bonus": r_serviced_bonus,
                "move": r_move,
                "wait": r_wait,
                "invalid": r_invalid,
                "overflow": r_overflow,
                "terminal": r_terminal,
            },
            "overflow_count": self.overflow_count,
            "serviced_bins_count": self.serviced_bins_count,
            "serviced_high_priority_count": self.serviced_high_priority_count,
        }

        return obs, float(total_reward), terminated, truncated, info

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _bin_index_at_position(self, pos: Tuple[int, int]) -> Optional[int]:
        """Return the index of the bin at the given position, or None."""
        for idx, (bx, by) in enumerate(self.bin_positions):
            if (bx, by) == pos:
                return idx
        return None

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

        # Normalize distances to [-1, 1]
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
        bin_features: List[float] = []

        distances = []
        for idx, (bx, by) in enumerate(self.bin_positions):
            dist = abs(bx - x) + abs(by - y)  # Manhattan distance
            distances.append((dist, idx))

        distances.sort(key=lambda t: t[0])
        k = min(self.K_NEAREST_BINS, len(distances))
        selected_indices = [distances[i][1] for i in range(k)]

        for idx in selected_indices:
            bx, by = self.bin_positions[idx]
            fill = float(self.bin_fill[idx])
            prio = float(self.bin_priority[idx])

            dx = bx - x
            dy = by - y
            dx_norm = np.clip(dx / (self.grid_width - 1), -1.0, 1.0)
            dy_norm = np.clip(dy / (self.grid_height - 1), -1.0, 1.0)
            fill_norm = np.clip(fill / self.max_capacity, 0.0, 1.0)

            bin_features.extend([dx_norm, dy_norm, fill_norm, prio])

        # Zero padding if fewer than K bins
        missing_bins = self.K_NEAREST_BINS - k
        if missing_bins > 0:
            bin_features.extend([0.0] * missing_bins * 4)

        bin_feats_arr = np.array(bin_features, dtype=np.float32)

        obs = np.concatenate([global_feats, bin_feats_arr], axis=0)
        assert obs.shape == self.observation_space.shape

        return obs

    # ----------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------
    def render(self):
        """
        Placeholder render method.

        Later, this will delegate to a dedicated renderer in `rendering.py`
        for 2D visualization (pygame/OpenGL).

        For now, it prints a minimal text representation if render_mode='human'.
        """
        if self.render_mode == "human":
            print(
                f"[EcoTrackEnv] Step: {self.t_step} | Pos: {self.truck_pos} | "
                f"Load: {self.truck_load:.1f} / {self.max_capacity:.1f} | "
                f"Overflows: {self.overflow_count} | "
                f"Serviced bins: {self.serviced_bins_count} "
                f"(high-prio: {self.serviced_high_priority_count})"
            )


    def close(self):
        """Clean up any rendering resources (if used)."""
        # Will be used when we add a graphical renderer
        pass
