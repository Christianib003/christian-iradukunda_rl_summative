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
   - Currently not used in the implementation (no blocked cells), but supported by the design.

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
  - `serviced_bins_count` – number of bins fully serviced (emptied below threshold).
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
    - If the bin’s fill level falls below a “serviced” threshold (10% of capacity), the bin is considered serviced, and an extra bonus may be given (higher for high-priority bins).
  - If the current cell does not contain a bin, or truck is already full:
    - No collection occurs,
    - A small penalty is applied for an invalid/unproductive action.

- **UNLOAD (5)**  
  - If the truck is at the depot cell:
    - `truck_load` is reset to 0 (all collected waste is unloaded),
    - Optional small reward; main benefit is freeing capacity.
  - If the truck is not at the depot:
    - No effect on `truck_load`,
    - A small penalty is applied for invalid unload attempts.

- **WAIT (6)**  
  - The truck remains in the same cell.
  - Time advances by one step.
  - Bin fill levels continue to increase.
  - A small penalty is applied to model the opportunity cost of idling.

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
   - `depot_dx_norm` – normalized horizontal distance from truck to depot (in `[-1, 1]`).
   - `depot_dy_norm` – normalized vertical distance from truck to depot (in `[-1, 1]`).
   - `overflow_ratio = overflow_count / total_bins` (in `[0, 1]`).
   - `high_prio_serviced_ratio = serviced_high_priority_count / total_high_priority_bins` (in `[0, 1]`).
   - `serviced_bins_ratio = serviced_bins_count / total_bins` (in `[0, 1]`).

2. **Local bin features for K nearest bins (K = 5 → 20 values)**  
   For each of the 5 nearest bins (by Manhattan distance):

   - `bin_k_dx_norm` – normalized horizontal offset from truck to bin k (in `[-1, 1]`).
   - `bin_k_dy_norm` – normalized vertical offset from truck to bin k (in `[-1, 1]`).
   - `bin_k_fill_norm` – current fill level of bin k normalized to `[0, 1]`.
   - `bin_k_priority` – 1 if high-priority bin, else 0.

If there are fewer than 5 bins, remaining slots are padded with zeros.

### 5.1 Observation Space in Gymnasium

- Represented as `spaces.Box(low=-1, high=1, shape=(29,), dtype=np.float32)`.
- All features are normalized to the range `[-1, 1]` or `[0, 1]` and then fit into the Box bounds.



## 6. Environment Dynamics

At each time step:

1. **Agent Action**  
   - The agent selects an action `a_t` from the discrete action space.
   - The environment applies the corresponding transition (movement, pickup, unload, or wait).

2. **Bin Fill Updates (Updated Tuning)**  

   To avoid unrealistically aggressive overflow while still making high-priority bins more urgent, bin fill increments are *moderate* and different for normal vs high-priority bins:

   - For **normal bins**:
     - `bin_fill[i] += U(0.3, 0.8)` units per step.
   - For **high-priority bins**:
     - `bin_fill[i] += U(0.6, 1.2)` units per step.

   This means high-priority bins fill roughly twice as fast on average, creating genuine urgency without making the task impossible.

   Fill levels are clamped at:

   - `max_bin_fill = 1.5 * max_capacity`

   to prevent runaway values during long episodes.

3. **Overflow Check**

   - Overflow threshold:
     - `overflow_threshold = max_capacity` (i.e., > 100% of capacity).
   - If `bin_fill[i] > overflow_threshold` and the bin has not overflowed previously in the episode:
     - `overflow_count += 1`,
     - A penalty is applied (larger for high-priority bins),
     - The bin is marked as overflowed (to avoid repeated penalties for the same bin).

4. **Time Update**

   - `t_step += 1`.

5. **Termination Check**

   - `terminated = True` if:
     - `t_step >= max_steps` (end of “day”).
   - `truncated = True` if:
     - `overflow_count` exceeds a maximum tolerance (e.g., > 5), meaning the day is considered a failure.

6. **Reward Calculation**

   - See Section 7 for detailed reward structure.

7. **Info Dict**

   - `info` includes metrics useful for analysis/training logs, e.g.:
     - `overflow_count`
     - `serviced_bins_count`
     - `serviced_high_priority_count`
     - `step_reward_components` (per-component breakdown of the reward).



## 7. Reward Structure

The reward at each step is composed of several components.

### 7.1 Positive Rewards

- **Collection reward**  
  - When `PICK_UP` successfully collects `Δw` units of waste from a bin:
    - `r_collect = alpha * Δw`  
      with `alpha = 0.05` in the current implementation.

- **Serviced bin bonus**  
  - When a bin’s fill level drops below a “serviced” threshold after pickup:
    - Serviced threshold:
      - `serviced_threshold_ratio = 0.1`  
        (i.e., bin fill ≤ 10% of capacity).
    - Normal bin: `+R_serviced_normal` (e.g., +2.0).
    - High-priority bin: `+R_serviced_high` (e.g., +5.0).

- **All high-priority serviced (terminal bonus)**  
  - At the end of the episode, if **all high-priority bins** have been serviced at least once:
    - `+R_all_high_prio` (e.g., +20.0).

### 7.2 Negative Rewards (Costs & Penalties)

- **Movement cost**  
  - For any `MOVE_*` action:
    - `r_move = -C_move` (e.g., `-0.05`).

- **Wait cost**  
  - For `WAIT` action:
    - `r_wait = -C_wait` (e.g., `-0.10`).

- **Invalid action cost**  
  - For:
    - `PICK_UP` at non-bin cell or when truck is full,
    - `UNLOAD` away from depot,
    - Movement into blocked/out-of-bound cell:
    - `r_invalid = -C_invalid` (e.g., `-0.50`).

- **Overflow penalty**  
  - When a bin overflows:
    - Normal bin overflow: `r_overflow_normal = -C_overflow_normal` (e.g., `-5.0`).
    - High-priority bin overflow: `r_overflow_high = -C_overflow_high` (e.g., `-10.0`).

### 7.3 Terminal Shaping

At the end of the episode (either normal termination or truncation):

- A shaping term based on overall performance:

```text
r_terminal = beta_terminal * serviced_bins_ratio
```

where:

* `serviced_bins_ratio = serviced_bins_count / total_bins`,

* `beta_terminal` is a small positive scalar (e.g., `+5.0`).

* Additional bonus if all high-priority bins have been serviced (as above).

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



## 8. Episode Initialization and Termination

### 8.1 Initialization (`reset()`)

At the beginning of each episode:

* `t_step = 0`.
* Truck is placed at the depot cell:

  * `truck_pos = depot_pos`.
* `truck_load = 0.0`.
* Bin fill levels are initialized to a **low but non-trivial** range:

```text
low_init  = 0.1 * max_capacity   # 10% of capacity
high_init = 0.2 * max_capacity   # 20% of capacity
bin_fill[i] ~ U(low_init, high_init)
```

This tuning was chosen after initial experiments showed that starting at 20–60% plus fast fill rates caused bins to overflow too quickly, leaving little room for exploration and learning. The new setting:

* Gives the agent more breathing room early in the episode,

* Still allows overflows to happen later if bins are neglected.

* Counters are reset:

  * `overflow_count = 0`,
  * `serviced_bins_count = 0`,
  * `serviced_high_priority_count = 0`.

* Boolean masks track per-bin status:

  * `serviced_bins_mask[i]` – whether bin i has been serviced (≤ 10%).
  * `overflowed_mask[i]` – whether bin i has overflowed.

* A PRNG seed can be passed to make initialization and bin dynamics reproducible.

### 8.2 Termination Conditions (`step()`)

* **Terminated** (`terminated = True`) if:

  * `t_step >= max_steps` (end of day).
* **Truncated** (`truncated = True`) if:

  * `overflow_count > max_overflows` (e.g., > 5), indicating catastrophic performance.

When either flag is true, the episode ends and `reset()` must be called before the next episode.



## 9. Stochasticity and Random Seeds

* Bin fill increments are stochastic, drawn from the ranges:

  * `U(0.3, 0.8)` for normal bins,
  * `U(0.6, 1.2)` for high-priority bins.
* Bin initial fills are drawn from `U(0.1 * max_capacity, 0.2 * max_capacity)`.
* The environment uses Gymnasium’s seeding API and an internal RNG (`np.random.Generator`) to ensure reproducibility.



## 10. Gymnasium & Stable-Baselines3 Compatibility

* **Action space:** `gymnasium.spaces.Discrete(7)`
* **Observation space:** `gymnasium.spaces.Box(low=-1, high=1, shape=(29,), dtype=np.float32)`
* `reset()` returns `(obs, info)` as per Gymnasium API.
* `step(action)` returns `(obs, reward, terminated, truncated, info)`.

For training with Stable-Baselines3, the environment will typically be wrapped in a vectorized wrapper (e.g., `DummyVecEnv` or `make_vec_env`).



## 11. Design & Tuning Notes (For Report Discussion)

* Initial configuration with faster bin fill rates and higher starting fills led to **very early overflows**, even for exploratory policies, making the environment too unforgiving.
* Through random-policy diagnostics and manual inspection, bin dynamics were tuned to:

  * Start fills at **10–20%** of capacity,
  * Use moderate increments of **0.3–0.8** (normal) and **0.6–1.2** (high-priority),
  * Keep high-priority bins clearly more urgent but still reachable.
* This tuning improved:

  * Learning stability for RL agents,
  * Realism (bins don’t instantly overflow),
  * The balance between exploration and task difficulty.



## 12. Future Extensions (Optional, Non-graded)

* Multiple trucks (multi-agent).
* Dynamic traffic / road closures.
* Fuel costs and refuelling actions.
* Time-of-day patterns for bin generation.

