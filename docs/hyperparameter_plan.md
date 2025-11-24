# Hyperparameter Search Plan – EcoTrack RL Summative

This document defines structured hyperparameter grids for the four RL algorithms used in the EcoTrack environment:

- DQN (value-based)
- REINFORCE (policy gradient)
- A2C (actor-critic)
- PPO (clipped policy gradient)

For each algorithm, we define at least **10 configurations** varying:

- Learning rate (`lr`)
- Discount factor (`gamma`)
- Network size (`net_arch`)
- And algorithm-specific choices such as:
  - Replay buffer size, batch size, epsilon schedule (DQN)
  - Batch episodes and baseline usage (REINFORCE)
  - `n_steps`, entropy coefficient, etc. (A2C, PPO)
  - `clip_range`, epochs, batch size (PPO)

Each configuration has a short ID so that results can be easily referenced in the report and plots.

> **Implementation note:**  
> - Network size will be set via `policy_kwargs={"net_arch": [hidden1, hidden2]}`.  
> - DQN epsilon decay will be controlled via `exploration_fraction` and `exploration_final_eps`.  
> - REINFORCE will be implemented as a custom training loop; `batch_episodes`, `use_baseline`, and `entropy_coef` will be explicit arguments there.

---

## 1. DQN Hyperparameter Grid (Value-based)

We will vary:

- **Learning rate**: `{3e-4, 1e-3, 3e-3}`
- **Gamma**: `{0.95, 0.99, 0.995}`
- **Network sizes**:  
  - S: `[64, 64]`  
  - M: `[128, 128]`  
  - L: `[256, 256]`
- **Replay buffer size**: `{20_000, 50_000, 100_000}`
- **Batch size**: `{32, 64, 128}`
- **Epsilon schedule** (via `exploration_fraction`, `exploration_final_eps`):
  - Slow decay vs faster decay
  - Final epsilon `{0.1, 0.05}`

We select 10 concrete combinations:

### 1.1 DQN Configurations

| ID        | lr      | gamma | net_arch   | buffer_size | batch_size | exploration_fraction | exploration_final_eps | Notes                                                |
|-----------|---------|-------|-----------|-------------|------------|----------------------|-----------------------|------------------------------------------------------|
| DQN-01    | 1e-3    | 0.99  | [128,128] | 50_000      | 64         | 0.20                 | 0.05                  | **Baseline** (matches current script defaults).      |
| DQN-02    | 3e-4    | 0.99  | [128,128] | 50_000      | 64         | 0.30                 | 0.05                  | Lower lr, slower decay to test more stable learning. |
| DQN-03    | 3e-3    | 0.99  | [128,128] | 50_000      | 64         | 0.20                 | 0.05                  | Higher lr, tests faster but riskier learning.        |
| DQN-04    | 1e-3    | 0.95  | [64,64]   | 20_000      | 32         | 0.25                 | 0.10                  | Smaller net & buffer, more myopic discount.          |
| DQN-05    | 1e-3    | 0.995 | [256,256] | 100_000     | 128        | 0.20                 | 0.05                  | Larger net, longer horizon, large buffer.            |
| DQN-06    | 3e-4    | 0.995 | [256,256] | 100_000     | 64         | 0.30                 | 0.05                  | Very conservative lr + big net + long horizon.       |
| DQN-07    | 1e-3    | 0.99  | [64,64]   | 50_000      | 32         | 0.10                 | 0.05                  | Faster early learning (short decay), smaller net.    |
| DQN-08    | 1e-3    | 0.99  | [256,256] | 50_000      | 128        | 0.30                 | 0.10                  | Large net, more stochastic exploration.              |
| DQN-09    | 3e-4    | 0.95  | [128,128] | 100_000     | 64         | 0.15                 | 0.05                  | Myopic gamma with large buffer & conservative lr.    |
| DQN-10    | 3e-3    | 0.99  | [64,64]   | 20_000      | 128        | 0.25                 | 0.05                  | Aggressive lr, small net, high batch size.           |

> **Implementation tasks later:**
> - Add CLI options for `net_arch`, `exploration_fraction`, `exploration_final_eps`, and pass them to `DQN` via `policy_kwargs` and constructor args.  
> - Log `ID` (e.g. `DQN-03`) in model filename and TensorBoard run name.

---

## 2. REINFORCE Hyperparameter Grid (Policy Gradient)

REINFORCE will be implemented as a custom episodic policy gradient:

- **Learning rate**: `{1e-3, 3e-4, 3e-3}`
- **Gamma**: `{0.95, 0.99}`
- **Network sizes**: `[64,64]`, `[128,128]`, `[256,256]`
- **Batch episodes** (episodes per policy update): `{5, 10, 20}`
- **Baseline usage**:
  - `use_baseline = {True, False}` (value function baseline to reduce variance)
- **Entropy coefficient**: `{0.0, 0.01}` (encourage exploration)

We pick 10 concrete combinations:

### 2.1 REINFORCE Configurations

| ID        | lr      | gamma | net_arch   | batch_episodes | use_baseline | entropy_coef | Notes                                                     |
|-----------|---------|-------|-----------|----------------|--------------|--------------|-----------------------------------------------------------|
| REIN-01   | 1e-3    | 0.99  | [128,128] | 10             | True         | 0.00         | **Baseline config** – moderate batch, baseline on.        |
| REIN-02   | 3e-4    | 0.99  | [128,128] | 10             | True         | 0.01         | Lower lr, entropy regularization for more exploration.    |
| REIN-03   | 3e-3    | 0.99  | [128,128] | 10             | True         | 0.00         | Higher lr, tests faster learning, baseline still on.      |
| REIN-04   | 1e-3    | 0.95  | [64,64]   | 5              | True         | 0.00         | Smaller net, more myopic gamma, frequent updates.         |
| REIN-05   | 1e-3    | 0.99  | [256,256] | 20             | True         | 0.00         | Larger net, longer batch, more stable gradient estimate.  |
| REIN-06   | 3e-4    | 0.95  | [128,128] | 20             | True         | 0.01         | Very stable lr + myopic discount + entropy.               |
| REIN-07   | 1e-3    | 0.99  | [128,128] | 5              | False        | 0.00         | No baseline, tests impact of higher variance.             |
| REIN-08   | 3e-3    | 0.95  | [64,64]   | 10             | False        | 0.01         | Aggressive update, small net, no baseline, extra entropy. |
| REIN-09   | 3e-4    | 0.99  | [256,256] | 10             | False        | 0.00         | Large net, low lr, no baseline.                           |
| REIN-10   | 1e-3    | 0.95  | [256,256] | 20             | True         | 0.01         | Big net, long batch, baseline + entropy.                  |

> **Implementation tasks later:**
> - Implement REINFORCE training loop with arguments matching the table.  
> - Use `baseline` as either:
>   - Learned value network, or  
>   - Moving average of returns (simpler variant).  
> - Log ID (e.g. `REIN-07`) in run name.

---

## 3. A2C Hyperparameter Grid (Actor-Critic)

We vary:

- **Learning rate**: `{7e-4, 3e-4, 1e-3}`
- **Gamma**: `{0.95, 0.99}`
- **n_steps**: `{64, 128, 256}` (rollout length)
- **Entropy coefficient**: `{0.0, 0.01, 0.02}`
- **Network sizes**: `[64,64]`, `[128,128]`, `[256,256]`
- **Value function coefficient (vf_coef)**: `{0.5, 0.7}`

We choose 10 configs:

### 3.1 A2C Configurations

| ID        | lr      | gamma | net_arch   | n_steps | ent_coef | vf_coef | Notes                                                        |
|-----------|---------|-------|-----------|---------|----------|---------|--------------------------------------------------------------|
| A2C-01    | 7e-4    | 0.99  | [128,128] | 128     | 0.00     | 0.50    | **Baseline A2C** – moderate rollout and network.             |
| A2C-02    | 3e-4    | 0.99  | [128,128] | 128     | 0.01     | 0.50    | Lower lr, small entropy regularization.                      |
| A2C-03    | 1e-3    | 0.99  | [128,128] | 128     | 0.00     | 0.50    | Higher lr to test faster adaptation.                         |
| A2C-04    | 7e-4    | 0.95  | [64,64]   | 64      | 0.00     | 0.50    | Shorter horizon, smaller net, more frequent updates.         |
| A2C-05    | 7e-4    | 0.99  | [256,256] | 256     | 0.00     | 0.50    | Larger net + longer rollout – more stable but heavier.       |
| A2C-06    | 3e-4    | 0.95  | [256,256] | 256     | 0.02     | 0.50    | Stronger entropy with big net and myopic gamma.              |
| A2C-07    | 7e-4    | 0.99  | [128,128] | 64      | 0.01     | 0.70    | More weight on value loss, shorter rollout.                  |
| A2C-08    | 1e-3    | 0.95  | [64,64]   | 128     | 0.02     | 0.50    | Aggressive lr + higher entropy on smaller net.               |
| A2C-09    | 3e-4    | 0.99  | [256,256] | 64      | 0.00     | 0.70    | Large net, short rollout, higher vf_coef.                    |
| A2C-10    | 7e-4    | 0.95  | [128,128] | 256     | 0.01     | 0.50    | Longer rollout with mild entropy & myopic discount.          |

> **Implementation tasks later:**
> - Update `training/pg_training.py` so A2C receives `policy_kwargs={"net_arch":[...]}`.  
> - Expose `ent_coef`, `vf_coef`, and `n_steps` via CLI for A2C.  
> - Use configuration IDs (e.g. `A2C-05`) in run names.

---

## 4. PPO Hyperparameter Grid (Clipped Policy Gradient)

We vary:

- **Learning rate**: `{3e-4, 1e-4, 1e-3}`
- **n_steps**: `{128, 256, 512}`
- **Batch size**: `{64, 128, 256}`
- **Clip range**: `{0.1, 0.2, 0.3}`
- **Entropy coefficient**: `{0.0, 0.01, 0.02}`
- **Epochs**: `{5, 10}` (per update)
- **Network sizes**: `[64,64]`, `[128,128]`, `[256,256]`

We define 10 concrete configs:

### 4.1 PPO Configurations

| ID        | lr      | gamma | net_arch   | n_steps | batch_size | clip_range | ent_coef | n_epochs | Notes                                                         |
|-----------|---------|-------|-----------|---------|------------|------------|----------|----------|---------------------------------------------------------------|
| PPO-01    | 3e-4    | 0.99  | [128,128] | 2048    | 64         | 0.20       | 0.00     | 10       | **Baseline PPO** (close to SB3 defaults).                     |
| PPO-02    | 1e-4    | 0.99  | [128,128] | 2048    | 64         | 0.20       | 0.01     | 10       | Lower lr, small entropy bonus.                               |
| PPO-03    | 1e-3    | 0.99  | [128,128] | 2048    | 64         | 0.20       | 0.00     | 10       | Higher lr, tests faster but riskier learning.                 |
| PPO-04    | 3e-4    | 0.95  | [64,64]   | 1024    | 64         | 0.10       | 0.01     | 5        | Smaller net, shorter horizon, tighter clipping.               |
| PPO-05    | 3e-4    | 0.99  | [256,256] | 4096    | 128        | 0.20       | 0.00     | 10       | Larger net, longer rollout, moderate clip.                    |
| PPO-06    | 3e-4    | 0.99  | [256,256] | 2048    | 256        | 0.30       | 0.02     | 10       | Wide batch, looser clip, stronger entropy regularization.     |
| PPO-07    | 1e-4    | 0.99  | [64,64]   | 2048    | 128        | 0.20       | 0.00     | 5        | Small net, lower lr, fewer epochs.                            |
| PPO-08    | 3e-4    | 0.95  | [128,128] | 1024    | 256        | 0.20       | 0.01     | 10       | Myopic gamma, shorter rollout, larger batch.                  |
| PPO-09    | 1e-3    | 0.99  | [256,256] | 1024    | 64         | 0.10       | 0.00     | 5        | Aggressive lr, big net, tighter clip, fewer epochs.           |
| PPO-10    | 3e-4    | 0.99  | [128,128] | 4096    | 128        | 0.30       | 0.01     | 10       | Long rollout, wider clip, moderate entropy.                   |

> **Implementation tasks later:**
> - In `training/pg_training.py`, pass `policy_kwargs={"net_arch":[...]}` into PPO and expose `clip_range`, `batch_size`, `n_epochs`, `ent_coef`, `n_steps` via CLI.  
> - Use configuration IDs (e.g. `PPO-06`) as part of `run-name` and log folder names.

---

## 5. Execution & Logging Plan

For each algorithm:

1. **Select 10 configs** from the tables above (all are already valid).
2. For each config:
   - Run training for a fixed number of timesteps (e.g. 50k–100k depending on algorithm).
   - Save the model under `models/<algo>/<ID>_...zip`.
   - Log training curves to TensorBoard under `logs/<algo>/<ID>`.
3. For each algorithm:
   - Compare:
     - Mean episode reward vs timesteps,
     - Mean episode length,
     - Overflow count per episode (if logged),
     - Serviced bins ratios.
   - Select 1–2 **best** configs per algorithm for:
     - Final video demo,
     - Detailed analysis in the report.

In the report, we will:

- Present **summary tables** like: best DQN vs best PPO vs best A2C vs best REINFORCE.  
- Provide **learning curves** for a subset of configurations to illustrate:
  - Sensitivity to lr, gamma, net size, etc.
  - Stability vs instability (e.g. too high lr or too loose clip range).

