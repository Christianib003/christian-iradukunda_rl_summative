
# EcoTrack: Reinforcement Learning for Smart Waste Collection

## 1. Project Overview

EcoTrack is a custom reinforcement learning project that simulates a smart waste-collection system on a two-dimensional grid. A single collection agent moves around a city-like map, services bins before they overflow, and must manage its limited capacity efficiently.

The goal is to learn control policies that minimize bin overflows and unnecessary movement while maximizing timely servicing. The project compares value-based and policy-gradient methods, including Deep Q-Networks (DQN), REINFORCE, A2C, and PPO, using a consistent environment and reward structure.

The repository includes the custom Gym-style environment, training pipelines for each algorithm, hyperparameter experiments, result aggregation, plotting utilities, and an entry-point demo script that visualizes the best-performing policy.


## 2. Environment: EcoTrackEnv

The environment is implemented in:

`environment/custom_env.py`

### 2.1 Agent

The agent represents an autonomous waste-collection vehicle operating in a grid-based city:

* Lives on a 10 × 10 grid.
* Starts with a fixed maximum capacity (e.g. 100 units).
* Moves between cells and services bins located on the grid.
* Has a limited number of steps per episode (episode horizon).

The agent’s objective is to service bins before they overflow while minimizing wasted movement and respecting capacity constraints.

### 2.2 Action Space

The action space is discrete and typically includes:

* 0: Move up
* 1: Move down
* 2: Move left
* 3: Move right
* 4: Service current bin

The exact mapping is defined in `EcoTrackEnv` and is used consistently by all algorithms.

### 2.3 Observation Space

The environment provides a vector-based observation that encodes:

* Agent position on the grid.
* Remaining capacity.
* Information about bin fill levels (e.g. local or global summary of bins).
* Progress within the episode (e.g. step index / max steps).

The observation is a numeric array (suitable for MLP policies) and is flattened before being fed into the neural networks used by DQN and policy-gradient methods.

### 2.4 Reward Structure

The reward function is designed to encourage timely servicing and overflow prevention while discouraging inefficient movement.

Typical components are:

* Positive reward for successfully servicing a bin, proportional to the amount collected.
* Large negative penalty when any bin overflows.
* Small per-step negative penalty to discourage wandering or unnecessarily long routes.
* Optional penalties for servicing when there is nothing useful to collect, or for hitting invalid actions.

In abstract form, per-step reward can be seen as:

`reward = r_service − λ_overflow * 1[overflow] − λ_step`

where:

* `r_service` is positive when trash is collected.
* `λ_overflow` is a large penalty coefficient for overflow events.
* `λ_step` is a small penalty per step.

This reward policy is shared across all algorithms to make comparison fair and meaningful.


## 3. Repository Structure

A typical structure for this project is:

```text
.
├── environment/
│   └── custom_env.py          # EcoTrackEnv definition
├── training/
│   ├── dqn_training.py        # DQN training script (DQN-01..10 presets)
│   └── pg_training.py         # Policy gradient training for REINFORCE, A2C, PPO
├── scripts/
│   ├── evaluate_best_models.py # Compare best DQN / REINFORCE / A2C / PPO
│   └── make_plots.py          # Load results/*.csv and generate figures
├── results/
│   ├── dqn_results.csv        # Hyperparameter sweep results for DQN
│   ├── reinforce_results.csv  # Hyperparameter sweep results for REINFORCE
│   ├── a2c_results.csv        # Hyperparameter sweep results for A2C
│   ├── ppo_results.csv        # Hyperparameter sweep results for PPO
│   └── final_comparison.csv   # Unified comparison of best models per algorithm
├── figures/
│   ├── cumulative_rewards.png
│   ├── training_stability.png
│   ├── episodes_to_converge.png
│   └── generalization_comparison.png
├── models/
│   ├── dqn/                   # Saved DQN models
│   └── pg/                    # Saved REINFORCE / A2C / PPO models
├── main.py                    # Entry point to load best model and visualize
├── requirements.txt
└── README.md
```


## 4. Setup and Installation

### 4.1 Prerequisites

* Python 3.10+ (3.11 is recommended)
* Git
* A virtual environment tool (e.g. `venv` or `conda`)

The project uses:

* `stable-baselines3` for DQN, PPO, and A2C
* `torch` for the custom REINFORCE implementation
* Standard scientific Python stack (`numpy`, `pandas`, `matplotlib`)

### 4.2 Clone the Repository

```bash
git clone https://github.com/Christianib003/christian-iradukunda_rl_summative.git
cd christian-iradukunda_rl_summative
```

### 4.3 Create and Activate Virtual Environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 4.4 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## 5. Running the Environment

You can interact with the environment directly for quick sanity checks.

Example (minimal):

```python
from environment.custom_env import EcoTrackEnv

env = EcoTrackEnv(
    grid_width=10,
    grid_height=10,
    max_steps=200,
    max_capacity=100.0,
    render_mode="human",   # or None for no rendering
)

obs, info = env.reset(seed=42)
done = False
truncated = False

while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

env.close()
```

This is useful to confirm that rendering works and that the environment behaves as expected before training RL agents.


## 6. Training

### 6.1 Deep Q-Network (DQN)

DQN training is implemented in:

`training/dqn_training.py`

The script supports:

* A set of predefined hyperparameter presets `DQN-01` .. `DQN-10`.
* Logging via Monitor CSV and model saving under `models/dqn/`.

Basic usage (single configuration):

```bash
python -m training.dqn_training \
  --config-id DQN-06 \
  --total-timesteps 500000
```

This will:

* Build the EcoTrack environment.
* Apply the chosen preset (`DQN-06`) with its learning rate, gamma, batch size, buffer size, etc.
* Train the DQN agent.
* Save the trained model to `models/dqn/` with a descriptive filename.
* Log episodic rewards to a monitor CSV and summary to `results/dqn_results.csv` (depending on the final implementation of the script).

Example hyperparameter sweep (all DQN presets):

```bash
for cfg in DQN-01 DQN-02 DQN-03 DQN-04 DQN-05 DQN-06 DQN-07 DQN-08 DQN-09 DQN-10; do
  echo "=== Running $cfg ==="
  python -m training.dqn_training \
    --config-id $cfg \
    --total-timesteps 500000
done
```

After this loop, `results/dqn_results.csv` will contain summary metrics for all 10 configurations.

### 6.2 Policy Gradient Methods (REINFORCE, A2C, PPO)

Policy-gradient training is implemented in:

`training/pg_training.py`

This single script supports:

* REINFORCE (custom PyTorch implementation)
* A2C (Stable-Baselines3)
* PPO (Stable-Baselines3)

Each algorithm has 10 predefined presets:

* `REIN-01` .. `REIN-10`
* `A2C-01` .. `A2C-10`
* `PPO-01` .. `PPO-10`

#### 6.2.1 REINFORCE

Example single configuration:

```bash
python -m training.pg_training \
  --algo reinforce \
  --config-id REIN-01 \
  --run-name REIN-01 \
  --total-episodes 400 \
  --eval-episodes 10
```

Hyperparameter sweep:

```bash
for cfg in REIN-01 REIN-02 REIN-03 REIN-04 REIN-05 REIN-06 REIN-07 REIN-08 REIN-09 REIN-10; do
  echo "=== Running $cfg ==="
  python -m training.pg_training \
    --algo reinforce \
    --config-id $cfg \
    --run-name $cfg \
    --total-episodes 400 \
    --eval-episodes 10
done
```

This produces:

* Trained REINFORCE models in `models/pg/`.
* Episode logs in `logs/pg/reinforce/...`.
* Summary metrics in `results/reinforce_results.csv`.

#### 6.2.2 A2C

Example sweep for A2C:

```bash
for cfg in A2C-01 A2C-02 A2C-03 A2C-04 A2C-05 A2C-06 A2C-07 A2C-08 A2C-09 A2C-10; do
  echo "=== Running $cfg ==="
  python -m training.pg_training \
    --algo a2c \
    --config-id $cfg \
    --run-name $cfg \
    --total-timesteps 50000 \
    --eval-episodes 10
done
```

This will train A2C agents with different configurations and store results in `results/a2c_results.csv` and models in `models/pg/`.

#### 6.2.3 PPO

Example sweep for PPO:

```bash
for cfg in PPO-01 PPO-02 PPO-03 PPO-04 PPO-05 PPO-06 PPO-07 PPO-08 PPO-09 PPO-10; do
  echo "=== Running $cfg ==="
  python -m training.pg_training \
    --algo ppo \
    --config-id $cfg \
    --run-name $cfg \
    --total-timesteps 50000 \
    --eval-episodes 10
done
```

Results are aggregated in `results/ppo_results.csv`, with models saved under `models/pg/`.


## 7. Results and Comparison

After running the hyperparameter sweeps for all algorithms, the project aggregates the best configurations and compares them on a shared evaluation protocol.

### 7.1 Per-Algorithm Results

The following CSV files are generated by the training pipeline:

* `results/dqn_results.csv`
* `results/reinforce_results.csv`
* `results/a2c_results.csv`
* `results/ppo_results.csv`

Each file typically contains, per configuration:

* `config_id`
* Main hyperparameters (e.g. learning rate, gamma, architecture)
* Evaluation metrics (mean reward, std reward, mean episode length, etc.)
* Model path and log directory

These CSVs are used both for analysis and for selecting the “best” configuration per algorithm.

### 7.2 Final Comparison

The script:

`scripts/evaluate_best_models.py`

uses the per-algorithm CSVs to:

* Identify the best configuration for DQN, REINFORCE, A2C, and PPO.
* Evaluate each best model on a common evaluation setup (new seeds / initial bin states).
* Collect metrics such as mean reward, overflow counts, and serviced bins.
* Write a summary table to:

`results/final_comparison.csv`

This file includes a column indicating the overall best model (e.g. via an `is_overall_best` flag) and is used by `main.py` to select the demo model.

Run:

```bash
python -m scripts.evaluate_best_models
```


## 8. Plotting and Analysis

Visual analysis of training and evaluation is handled by:

`scripts/make_plots.py`

Typical plots include:

* Cumulative reward curves over episodes for each algorithm’s best configuration.
* Training stability indicators (e.g. loss curves for DQN, entropy trends for policy-gradient methods).
* Episodes-to-converge or similar convergence proxies per algorithm.
* Generalization performance (e.g. bar chart of mean reward on test seeds).

Run:

```bash
python -m scripts.make_plots
```

All figures are saved into the `figures/` directory with descriptive filenames such as:

* `figures/cumulative_rewards.png`
* `figures/training_stability.png`
* `figures/episodes_to_converge.png`
* `figures/generalization_comparison.png`

These plots are used directly in the written report and for interpretation of results.


## 9. Demo: Visualizing the Best Model

The main entry point for demonstrating the learned policy is:

`main.py`

It:

* Parses command-line arguments specifying which algorithm to use.
* Optionally reads `results/final_comparison.csv` to automatically select the best-overall model.
* Loads the corresponding trained model.
* Creates an EcoTrack environment with `render_mode="human"`.
* Runs several episodes while rendering and printing episode statistics.

### 9.1 Basic Usage

To automatically pick the best model (based on `final_comparison.csv`) and run a short demo:

```bash
python main.py
```

This will:

* Load the row with the best performance (e.g. best DQN configuration).
* Load the corresponding model file from `models/dqn/` or `models/pg/`.
* Run and render several episodes.

### 9.2 Manual Selection

You can override the automatic selection and manually specify the algorithm and model path:

```bash
python main.py \
  --algo dqn \
  --model-path models/dqn/dqn_ecotrack_DQN-06_lr0.0003_g0.995_bs64_buf100000.zip \
  --episodes 5 \
  --fps 4.0
```

For REINFORCE, you can additionally override the policy network architecture if needed:

```bash
python main.py \
  --algo reinforce \
  --model-path models/pg/reinforce_ecotrack_REIN-08_lr0.003_g0.95_be10_bl0_ent0.01.pt \
  --reinforce-net-arch 64,64 \
  --episodes 5 \
  --fps 4.0
```


## 10. Reproducibility Notes

* Each training script accepts a `--seed` argument to control randomness.
* Hyperparameter presets (e.g. `DQN-06`, `REIN-08`, `A2C-03`, `PPO-04`) are defined in code so experiments can be reproduced by referencing `config_id`.
* Evaluation scripts use fixed sets of seeds or random initialization strategies, which are documented in the code.
* All key configuration and result files (`results/*.csv`, `models/*`, `figures/*`) are intended to support reproducibility and comparison across runs.


## 11. Contact

For questions, suggestions, or extensions of this project (e.g. multi-agent variants, more complex reward shaping, or deployment ideas), please use the issue tracker on the repository or reach out to the project author directly.
