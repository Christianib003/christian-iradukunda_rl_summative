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
        # Adjusted for better pacing
        self.fill_rate_normal = 0.02    # Was 0.1 (Now much slower)
        self.fill_rate_priority = 0.08   # Was 0.3 (Still 5x faster than normal, but manageable)
        
        
        # --- Reward Configuration ---
        self.reward_collect_scale = 1.0   # Reward per unit of trash collected
        self.reward_prio_bonus = 5.0      # Bonus for servicing a high-priority bin
        self.reward_mission_complete = 20.0 # Bonus for finishing the day with all high-prio bins empty
        self.penalty_move = -0.05         # Cost per movement step
        self.penalty_wait = -0.1          # Cost per wait step
        self.penalty_invalid = -0.5       # Cost for illegal actions
        self.penalty_overflow = -2.0      # Penalty PER overflowing bin PER step

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
        
        # --- Rendering Configuration ---
        self.cell_size = 64  # Pixels per grid cell
        self.window_size = self.grid_size * self.cell_size
        
        # Colors (R, G, B)
        self.COLOR_BG = (255, 255, 255)      # White
        self.COLOR_GRID = (200, 200, 200)    # Light Grey
        self.COLOR_DEPOT = (50, 50, 255)     # Blue
        self.COLOR_BIN_LOW = (0, 255, 0)     # Green
        self.COLOR_BIN_MED = (255, 255, 0)   # Yellow
        self.COLOR_BIN_HIGH = (255, 0, 0)    # Red
        self.COLOR_AGENT = (50, 50, 50)      # Dark Grey Truck

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
        # Initialize variables for reward calculation
        step_reward = 0.0
        amount_collected = 0.0
        prio_bonus = 0.0
        is_invalid_action = False
        
        # --- 1. Movement Logic (Actions 0-3) ---
        if action in [0, 1, 2, 3]:
            direction = np.array([0, 0])
            if action == 0: direction = np.array([0, -1])   # Up
            elif action == 1: direction = np.array([0, 1])  # Down
            elif action == 2: direction = np.array([-1, 0]) # Left
            elif action == 3: direction = np.array([1, 0])  # Right
            
            new_pos = self.agent_pos + direction
            
            # Boundary Check
            if (0 <= new_pos[0] < self.grid_size) and (0 <= new_pos[1] < self.grid_size):
                self.agent_pos = new_pos
                step_reward += self.penalty_move
            else:
                # Hit wall
                is_invalid_action = True # Or just treat as wasted move
                step_reward += self.penalty_move # Still pay fuel cost
        
        # --- 2. Interaction Logic ---
        elif action == 4: # PICK_UP
            # Find bin at current location
            bin_at_loc = None
            for b in self.bins:
                if np.array_equal(b["pos"], self.agent_pos):
                    bin_at_loc = b
                    break
            
            if bin_at_loc and bin_at_loc["fill"] > 0 and self.agent_load < self.truck_capacity:
                space_left = self.truck_capacity - self.agent_load
                amount_to_take = min(space_left, bin_at_loc["fill"])
                
                self.agent_load += amount_to_take
                bin_at_loc["fill"] -= amount_to_take
                amount_collected = amount_to_take
                
                # Bonus if we serviced a high priority bin
                if bin_at_loc["is_priority"] and amount_to_take > 0:
                    prio_bonus = self.reward_prio_bonus
            else:
                is_invalid_action = True

        elif action == 5: # UNLOAD
            if np.array_equal(self.agent_pos, self.depot_pos):
                if self.agent_load > 0:
                    self.agent_load = 0.0
                else:
                    is_invalid_action = True
            else:
                is_invalid_action = True

        elif action == 6: # WAIT
            step_reward += self.penalty_wait

        # --- 3. Update Environment State ---
        self._update_bins()
        
        # --- 4. Calculate Reward ---
        # Add collection rewards
        step_reward += (amount_collected * self.reward_collect_scale)
        step_reward += prio_bonus
        
        # Subtract invalid action penalty
        if is_invalid_action:
            step_reward += self.penalty_invalid
            
        # Subtract Overflow Penalty (Critical!)
        overflow_count = sum(1 for b in self.bins if b["fill"] > b["capacity"])
        step_reward += (overflow_count * self.penalty_overflow)

        # --- 5. Check Termination ---
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # Check Success Condition: All High Priority bins are empty?
        # Note: This is hard! Maybe just define success as "Survival" or "High Score".
        # But for the rubric, let's keep it simple: no early termination for success, 
        # just try to get max score until time runs out.
        terminated = False 

        # Get new observation
        observation = self._get_obs()
        info = {
            "overflow_count": overflow_count,
            "load": self.agent_load
        }
        
        return observation, step_reward, terminated, truncated, info
    
    
    def render(self):
        """
        Public method to trigger rendering.
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
        """
        Internal method to do the actual drawing.
        Combines Map, Truck, and HUD.
        """
        import pygame
        
        # 1. Initialize Pygame & Font (Ensure this happens in ALL modes)
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        
        # Initialize font if it doesn't exist yet (works for rgb_array too)
        if not hasattr(self, 'font'):
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 20)
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # 2. Create a canvas to draw on
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.COLOR_BG)
        
        # 3. Draw Depot
        depot_rect = pygame.Rect(
            self.depot_pos[0] * self.cell_size,
            self.depot_pos[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(canvas, self.COLOR_DEPOT, depot_rect)
        
        # 4. Draw Bins
        for b in self.bins:
            x, y = b["pos"]
            
            # Determine color based on fill level
            fill_ratio = b["fill"] / b["capacity"]
            if fill_ratio < 0.5:
                color = self.COLOR_BIN_LOW
            elif fill_ratio < 1.0:
                color = self.COLOR_BIN_MED
            else:
                color = self.COLOR_BIN_HIGH # Overflowing!
            
            # Draw bin
            bin_size = int(self.cell_size * 0.6)
            offset = int((self.cell_size - bin_size) / 2)
            
            bin_rect = pygame.Rect(
                x * self.cell_size + offset,
                y * self.cell_size + offset,
                bin_size,
                bin_size
            )
            pygame.draw.rect(canvas, color, bin_rect)
            
            # If High Priority, add border
            if b["is_priority"]:
                pygame.draw.rect(canvas, (0,0,0), bin_rect, 3)

        # 5. Draw Grid Lines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, 
                self.COLOR_GRID, 
                (0, x * self.cell_size), 
                (self.window_size, x * self.cell_size)
            )
            pygame.draw.line(
                canvas, 
                self.COLOR_GRID, 
                (x * self.cell_size, 0), 
                (x * self.cell_size, self.window_size)
            )

        # 6. Draw Agent (Truck)
        center_x = int(self.agent_pos[0] * self.cell_size + self.cell_size / 2)
        center_y = int(self.agent_pos[1] * self.cell_size + self.cell_size / 2)
        radius = int(self.cell_size * 0.35)
        
        pygame.draw.circle(canvas, self.COLOR_AGENT, (center_x, center_y), radius)
        
        if self.agent_load > 0:
            load_radius = int(radius * 0.5)
            pygame.draw.circle(canvas, (200, 200, 200), (center_x, center_y), load_radius)

        # 7. Draw HUD (Heads Up Display) - NOW OUTSIDE THE 'HUMAN' CHECK
        # Create a semi-transparent background for text
        overlay = pygame.Surface((self.window_size, 60)) # 60px high
        overlay.set_alpha(200) # Transparency
        overlay.fill((0, 0, 0)) # Black background
        canvas.blit(overlay, (0, 0)) # Draw at top

        # Prepare text surfaces
        text_color = (255, 255, 255)
        
        # Stats Text
        overflow_count = sum(1 for b in self.bins if b["fill"] > b["capacity"])
        stats_text = f"Step: {self.current_step} | Load: {int(self.agent_load)} | Overflows: {overflow_count}"
        
        # Render Text
        label = self.font.render(stats_text, True, text_color)
        canvas.blit(label, (10, 10))
        
        help_text = "Red=Overflow  Yellow=Full  Blue=Depot"
        label2 = self.font.render(help_text, True, (200, 200, 200))
        canvas.blit(label2, (10, 35))

        # 8. Output to Screen (Only if human)
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
        # 9. Return Array (For video recording)
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2))


    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()