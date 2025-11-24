# EcoTrack Environment Specification

## 1. Overview

**Environment name:** `EcoTrackEnv`  
**Domain:** Smart waste collection in a small city  
**Agent:** Autonomous garbage truck  
**Objective:** Maximize total collected waste and service of high-priority bins while:
- Minimizing travel and idle time,
- Avoiding bin overflow events,
- Operating within a fixed “day length” (episode horizon).

The environment is implemented as a custom Gymnasium environment compatible with Stable-Baselines3 and will be used to train and compare four RL algorithms:

- DQN (value-based),
- REINFORCE (policy gradient),
- A2C (actor-critic),
- PPO (policy gradient with clipping).



## 2. Environment Layout

### 2.1 Grid

- The city is represented as a **10 × 10** grid.
- Coordinate system:
  - `x ∈ {0, …, 9}` horizontally,
  - `y ∈ {0, …, 9}` vertically.

### 2.2 Cell Types

Each cell belongs to exactly one of the following types:

1. **Road cell**  
   - Traversable by the truck.
2. **Bin cell (normal)**  
   - Contains a trash bin with a fill level that increases over time.
3. **High-priority bin cell**  
   - Same as bin cell, but associated with a high-priority location (e.g., hospital/market).
   - Fill rate is higher and overflow is more costly.
4. **Depot cell**  
   - Starting position of the truck.
   - Only cell where the agent can unload collected waste.
5. **Blocked / non-road cell (optional)**  
   - Non-traversable (truck cannot enter).
   - Included to increase routing complexity (e.g., buildings).

The exact layout (positions of bins, high-priority bins, depot, and blocked cells) is defined in the environment initialization and can be fixed or sampled from a small set of predefined maps.


## 3. Agent Definition

- **Single agent:** a garbage truck.
- Internal state variables maintained by the environment:
  - `truck_pos = (x, y)` – current location on the grid.
  - `truck_load ∈ [0, max_capacity]` – current load of collected trash.
  - `t_step ∈ [0, max_steps]` – current time step in the episode.
  - `bin_fill[i] ∈ [0, max_bin_fill]` – fill level of each bin `i`.
  - `bin_priority[i] ∈ {0, 1}` – 1 if bin is high-priority, else 0.
  - `overflow_count` – number of bins that have overflowed during the episode.
  - `serviced_bins_count` – number of bins fully serviced.
  - `serviced_high_priority_count` – number of high-priority bins fully serviced.



## 4. Action Space

The action space is **discrete** with 7 actions:

0. `MOVE_UP`  
1. `MOVE_DOWN`  
2. `MOVE_LEFT`  
3. `MOVE_RIGHT`  
4. `PICK_UP`  
5. `UNLOAD`  
6. `WAIT`

### 4.1 Action Semantics

- **MOVE_* (0–3)**  
  - Attempt to move the truck one cell in the given direction.
  - If the target cell is outside the grid or blocked:
    - Position remains unchanged,
    - The step still consumes time,
    - A small penalty is applied for ineffective movement.

- **PICK_UP (4)**  
  - If the current cell contains a bin and the truck has free capacity:
    - Transfer as much trash as possible from the bin to the truck, up to `max_capacity`.
    - Bin fill level is reduced accordingly.
    - Positive reward is awarded per unit collected.
    - If the bin’s fill level falls below a “serviced” threshold (e.g., 10% of capacity), the bin is considered serviced, and an extra bonus may be given (higher for high-priority bins).
  - If the current cell does not contain a bin, or truck is already full:
    - No collection occurs,
    - A small penalty is applied for an invalid/unproductive action.

- **UNLOAD (5)**  
  - If the truck is at the depot cell:
    - `truck_load` is reset to 0 (all collected waste is unloaded),
    - No additional reward (or a small positive reward) is given directly; the main benefit is freeing capacity for future collections.
  - If the truck is not at the depot:
    - No effect on `truck_load`,
    - A small penalty is applied for invalid unload attempts.

- **WAIT (6)**  
  - The truck remains in the same cell.
  - Time advances by one step.
  - Bin fill levels continue to increase.
  - A small penalty is applied to model the opportunity cost of idling.
  - In principle, `WAIT` allows the agent to learn that idleness is generally suboptimal, unless there is a strategic reason (e.g., timing a pickup near a high-priority bin’s fill level).

### 4.2 Edge Cases

- Attempting to move out of bounds or into a blocked cell:
  - No position change, small negative reward.
- `PICK_UP` at a non-bin cell or with no remaining capacity:
  - No state improvement, penalty applied.
- `UNLOAD` away from depot:
  - No unloading, penalty applied.

These edge cases are included deliberately to encourage learning robust policies and to ensure the action space is exhaustive and realistic.



## 5. Observation Space

The agent receives a **vector observation** encoding its own state and information about nearby bins and global context.

The observation is a 29-dimensional vector:

1. **Global / agent-centric features (9 values)**  
   - `truck_x_norm` – truck x-coordinate normalized to `[0, 1]`.
   - `truck_y_norm` – truck y-coordinate normalized to `[0, 1]`.
   - `load_ratio = truck_load / max_capacity` (in `[0, 1]`).
   - `time_ratio = t_step / max_steps` (in `[0, 1]`).
   - `depot_dx_norm` – normalized horizontal distance from truck to depot (e.g., in `[-1, 1]`).
   - `depot_dy_norm` – normalized vertical distance from truck to depot (e.g., in `[-1, 1]`).
   - `overflow_ratio = overflow_count / total_bins` (in `[0, 1]`).
   - `high_prio_serviced_ratio = serviced_high_priority_count / total_high_priority_bins` (in `[0, 1]`).
   - `serviced_bins_ratio = serviced_bins_count / total_bins` (in `[0, 1]`).

2. **Local bin features for K nearest bins (here K = 5 → 20 values)**  
   For each of the 5 nearest bins (based on Manhattan distance from truck):

   - `bin_k_dx_norm` – normalized horizontal offset from truck to bin k (e.g., in `[-1, 1]`).
   - `bin_k_dy_norm` – normalized vertical offset from truck to bin k (e.g., in `[-1, 1]`).
   - `bin_k_fill_norm` – current fill level of bin k normalized to `[0, 1]`.
   - `bin_k_priority` – 1 if high-priority bin, else 0.

If there are fewer than 5 bins (theoretically), remaining slots are padded with zeros.

### 5.1 Observation Space in Gymnasium

- Represented as `spaces.Box(low=-1, high=1, shape=(29,), dtype=np.float32)`.
- All features are normalized to the range `[-1, 1]` or `[0, 1]` and then fit into the Box bounds.



## 6. Environment Dynamics

At each time step:

1. **Agent Action**  
   - The agent selects an action `a_t` from the discrete action space.
   - The environment applies the corresponding transition (movement, pickup, unload, or wait).

2. **Bin Fill Updates**  
   - Every bin `i` updates its fill level:
     - For normal bins:
       - `bin_fill[i] += U(normal_min, normal_max)` (e.g., 1–3 units per step).
     - For high-priority bins:
       - `bin_fill[i] += U(high_min, high_max)` (e.g., 2–5 units per step).
   - Fill levels are capped at `max_bin_fill` for tracking, but overflow penalties trigger when fill exceeds `overflow_threshold` (e.g., 100% of capacity).

3. **Overflow Check**  
   - If `bin_fill[i] > overflow_threshold` and the bin is not yet marked as overflowed:
     - `overflow_count += 1`,
     - A penalty is applied (larger for high-priority bins),
     - The bin may be clamped to a maximum value.

4. **Time Update**  
   - `t_step += 1`.

5. **Termination Check**  
   - `terminated = True` if:
     - `t_step >= max_steps` (end of “day”),
   - `truncated = True` optionally if:
     - `overflow_count` exceeds a maximum tolerance (e.g., > 5), meaning the day is considered a failure.

6. **Reward Calculation**  
   - See Section 7 for detailed reward structure.

7. **Info Dict**  
   - `info` may include:
     - `{"overflow_count": ..., "serviced_bins": ..., "serviced_high_priority": ..., "step_reward_components": {...}}`
   - Used mainly for logging and analysis, not by the agent.



## 7. Reward Structure

The reward at each step is built from several components:

### 7.1 Positive Rewards

- **Collection reward**  
  - When `PICK_UP` successfully collects `Δw` units of waste from a bin:
    - `r_collect = alpha * Δw`  
      Example: `alpha = 0.05` → +0.05 reward per unit.

- **Serviced bin bonus**  
  - When a bin’s fill level drops below a “serviced” threshold after pickup:
    - Normal bin: `+R_serviced_normal` (e.g., +2.0).
    - High-priority bin: `+R_serviced_high` (e.g., +5.0).

- **All high-priority serviced (terminal bonus)**  
  - At the end of the episode, if **all high-priority bins** have been serviced at least once:
    - `+R_all_high_prio` (e.g., +20.0).

### 7.2 Negative Rewards (Costs & Penalties)

- **Movement cost**  
  - For any `MOVE_*` action:
    - `r_move = -C_move` (e.g., `-0.05`) to discourage unnecessary wandering.

- **Wait cost**  
  - For `WAIT` action:
    - `r_wait = -C_wait` (e.g., `-0.1`) to model the opportunity cost of idling.

- **Invalid action cost**  
  - For `PICK_UP` at non-bin cell or when truck is full; `UNLOAD` away from depot; movement into blocked/out-of-bound cell:
    - `r_invalid = -C_invalid` (e.g., `-0.5`).

- **Overflow penalty**  
  - When a bin overflows:
    - Normal bin overflow: `r_overflow_normal = -C_overflow_normal` (e.g., `-5.0`).
    - High-priority bin overflow: `r_overflow_high = -C_overflow_high` (e.g., `-10.0`).

### 7.3 Terminal Shaping (Optional)

At the end of the episode:

- Additional shaping term based on overall performance, e.g.:

```text
r_terminal = beta * serviced_bins_ratio
```

Where `beta` is a small positive scalar (e.g., `+5.0`).

### 7.4 Total Reward

At each step:

```text
r_t = r_collect
    + r_serviced_bonus
    + r_move
    + r_wait
    + r_invalid
    + r_overflow
    + (r_terminal if episode ends)
```

Hyperparameters (`alpha`, `C_move`, `C_wait`, etc.) will be tuned initially to obtain reasonable learning signals and will be documented in the report.



## 8. Episode Initialization and Termination

### 8.1 Initialization (`reset()`)

* Set `t_step = 0`.
* Place the truck at the depot cell.
* Initialize bin fill levels:

  * Sample initial fills from a range, e.g. 20–60% of capacity for variability.
* Reset counters:

  * `truck_load = 0`,
  * `overflow_count = 0`,
  * `serviced_bins_count = 0`,
  * `serviced_high_priority_count = 0`.
* Randomly vary initial bin fills (and optionally map variants) based on environment seed to support generalization.
* Return the initial observation and an empty `info` dict.

### 8.2 Termination Conditions (`step()`)

* **Terminated** (`terminated = True`) if:

  * `t_step >= max_steps` (end of day).
* **Truncated** (`truncated = True`) if:

  * `overflow_count > max_overflows` (environment decides the day is catastrophically bad).
* When either flag is true, the episode ends and `reset()` must be called before the next episode.



## 9. Stochasticity and Random Seeds

* Bin fill increments are stochastic (uniform distributions with different ranges for normal vs high-priority bins).
* Initial bin fill levels are randomly sampled.
* Optionally, multiple map layouts can be sampled from a small predefined set.
* `seed` passed to `reset()` and the environment’s RNG will make the stochastic dynamics reproducible for evaluation.



## 10. Gymnasium & Stable-Baselines3 Compatibility

* **Action space:** `gymnasium.spaces.Discrete(7)`
* **Observation space:** `gymnasium.spaces.Box(low=-1, high=1, shape=(29,), dtype=np.float32)`
* `reset()` returns `(obs, info)` as per Gymnasium API.
* `step(action)` returns `(obs, reward, terminated, truncated, info)`.

The environment will be wrapped in a vectorized wrapper (`DummyVecEnv` or `make_vec_env`) when used with Stable-Baselines3.



## 11. Future Extensions 

* Multiple trucks (multi-agent).
* Dynamic traffic / road closures.
* Fuel costs and refuelling actions.
* More complex bin generation patterns (time-of-day behaviour).

