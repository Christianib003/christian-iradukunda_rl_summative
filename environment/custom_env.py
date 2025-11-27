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
        self.truck_capacity = 100.0
        self.bin_capacity = 10.0  # Max fill for a bin before overflow
        self.fill_rate_normal = 0.1   # Amount per step
        self.fill_rate_priority = 0.3 # High priority fills 3x faster

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
        super().reset(seed=seed)
        
        # Initialize internal state
        self.current_step = 0
        self.agent_load = 0.0
        
        # Generate new map layout
        self._generate_state()
        
        # Get initial observation
        observation = self._get_obs()
        info = {}
        
        return observation, info


    def step(self, action):
        """
        Executes one time step within the environment.
        """
        # --- 1. Movement Logic (Actions 0-3) ---
        # 0: Up, 1: Down, 2: Left, 3: Right
        direction = np.array([0, 0])
        
        if action == 0:   # Up
            direction = np.array([0, -1])
        elif action == 1: # Down
            direction = np.array([0, 1])
        elif action == 2: # Left
            direction = np.array([-1, 0])
        elif action == 3: # Right
            direction = np.array([1, 0])
            
        # Apply movement if it is a move action
        if action in [0, 1, 2, 3]:
            new_pos = self.agent_pos + direction
            
            # Boundary Check: Ensure new position is within grid limits
            if (0 <= new_pos[0] < self.grid_size) and (0 <= new_pos[1] < self.grid_size):
                self.agent_pos = new_pos
            else:
                # Wall hit: Agent stays in current position
                pass
        
        # Action 6 (WAIT) is implicitly handled here by doing nothing to position
        
        # --- 2. Interaction Logic (Placeholder for Card 7) ---
        # --- 2. Interaction Logic ---
        reward = 0  # Initialize reward for this step
        
        # Action 4: PICK_UP
        if action == 4:
            # Check if there is a bin at the current location
            bin_at_loc = None
            for b in self.bins:
                if np.array_equal(b["pos"], self.agent_pos):
                    bin_at_loc = b
                    break
            
            if bin_at_loc:
                # Calculate how much space we have left in the truck
                space_left = self.truck_capacity - self.agent_load
                
                # We can take min(space_left, amount_in_bin)
                amount_to_take = min(space_left, bin_at_loc["fill"])
                
                if amount_to_take > 0:
                    # Perform transfer
                    self.agent_load += amount_to_take
                    bin_at_loc["fill"] -= amount_to_take
                    
                    # Reward for collection (will be tuned in Card 9)
                    # For now, just tracking state changes
                else:
                    # Bin empty or Truck full - Invalid action penalty later
                    pass
            else:
                # No bin here - Invalid action penalty later
                pass

        # Action 5: UNLOAD
        elif action == 5:
            # Check if at Depot
            if np.array_equal(self.agent_pos, self.depot_pos):
                if self.agent_load > 0:
                    # Successful unload
                    self.agent_load = 0.0
                else:
                    # Already empty - minor waste of time
                    pass
            else:
                # Not at depot - Invalid action penalty later
                pass
        
        # --- 3. Update Environment State ---
        self._update_bins()
        
        # --- 4. Calculate Reward (Placeholder for Card 9) ---
        reward = 0
        
        # --- 5. Check Termination ---
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False  # Will be True if task completed (added later)
        
        # Get new observation
        observation = self._get_obs()
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
            
    def _generate_state(self):
        """
        Randomly places the depot and bins on the grid.
        """
        # 1. Place Depot (randomly)
        self.depot_pos = np.random.randint(0, self.grid_size, size=2)
        
        # 2. Place Agent (at Depot initially)
        self.agent_pos = self.depot_pos.copy()
        
        # 3. Place Bins
        # We want unique locations for bins, avoiding the depot
        available_locs = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if not np.array_equal([x, y], self.depot_pos):
                    available_locs.append([x, y])
        
        # Select random locations for bins
        bin_indices = np.random.choice(len(available_locs), self.n_bins, replace=False)
        bin_locs = [available_locs[i] for i in bin_indices]
        
        self.bins = []
        for i, loc in enumerate(bin_locs):
            is_priority = 1 if i < self.n_high_priority else 0
            # Initial random fill between 0.1 and 0.5 (10% to 50%)
            initial_fill = np.random.uniform(0.1, 0.5) * self.bin_capacity
            
            self.bins.append({
                "pos": np.array(loc),
                "fill": initial_fill,
                "is_priority": is_priority,
                "capacity": self.bin_capacity
            })
    
    
    def _update_bins(self):
        """
        Simulates waste generation.
        """
        for b in self.bins:
            # Base fill rate based on priority
            rate = self.fill_rate_priority if b["is_priority"] else self.fill_rate_normal
            
            # Add randomness (e.g., +/- 20% variability)
            actual_fill = rate * np.random.uniform(0.8, 1.2)
            
            # Update bin
            b["fill"] += actual_fill
            
            # Note: We do NOT cap the fill at capacity here.
            # We allow it to go above capacity so we can detect "Overflow" 
            # in the reward function (next card).
    
    
    def _get_obs(self):
        # 1. Agent State
        # Normalize position to [0, 1]
        agent_x_norm = self.agent_pos[0] / self.grid_size
        agent_y_norm = self.agent_pos[1] / self.grid_size
        load_norm = self.agent_load / self.truck_capacity
        time_norm = self.current_step / self.max_steps
        
        agent_state = np.array([agent_x_norm, agent_y_norm, load_norm, time_norm], dtype=np.float32)

        # 2. Depot State
        # Normalized Manhattan distance to depot
        dist_depot = np.sum(np.abs(self.agent_pos - self.depot_pos)) / self.grid_size
        depot_state = np.array([dist_depot], dtype=np.float32)

        # 3. Nearest Bins State (The complex part)
        # Calculate distances to all bins
        bin_distances = []
        for i, b in enumerate(self.bins):
            dist = np.sum(np.abs(self.agent_pos - b["pos"]))
            bin_distances.append((dist, i))
        
        # Sort by distance and take top K
        bin_distances.sort(key=lambda x: x[0])
        nearest_indices = [x[1] for x in bin_distances[:self.k_nearest]]
        
        nearest_bins_data = []
        for idx in nearest_indices:
            b = self.bins[idx]
            # Relative position (dx, dy) normalized
            dx = (b["pos"][0] - self.agent_pos[0]) / self.grid_size
            dy = (b["pos"][1] - self.agent_pos[1]) / self.grid_size
            fill_ratio = b["fill"] / b["capacity"]
            is_prio = float(b["is_priority"])
            nearest_bins_data.extend([dx, dy, fill_ratio, is_prio])
            
        # If we have fewer than K bins (edge case), pad with zeros
        while len(nearest_bins_data) < self.k_nearest * 4:
            nearest_bins_data.extend([0, 0, 0, 0])
            
        bins_state = np.array(nearest_bins_data, dtype=np.float32)

        # 4. Global State
        # Fraction of bins overflowing
        overflow_count = sum(1 for b in self.bins if b["fill"] > b["capacity"])
        overflow_ratio = overflow_count / self.n_bins
        
        # Fraction of high priority bins that are currently empty (serviced)
        high_prio_bins = [b for b in self.bins if b["is_priority"]]
        serviced_high_prio = sum(1 for b in high_prio_bins if b["fill"] <= 0)
        prio_clean_ratio = serviced_high_prio / len(high_prio_bins) if high_prio_bins else 1.0
        
        global_state = np.array([overflow_ratio, prio_clean_ratio], dtype=np.float32)

        # Concatenate all
        return np.concatenate([agent_state, depot_state, bins_state, global_state])


    def _render_frame(self):
        # TODO: Implement Pygame rendering (Card 10)
        pass

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()