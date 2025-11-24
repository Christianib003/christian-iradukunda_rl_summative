# EcoTrack – Smart Waste Collection RL

This repository contains the implementation of my Reinforcement Learning summative assignment.

## Project Overview

EcoTrack is a custom reinforcement learning environment where an autonomous garbage truck operates in a small smart city. Trash bins across the city fill over time, some being high-priority (e.g. markets, hospitals). The agent must decide:

- Where to move,
- When to collect trash,
- When to unload at the depot,

to maximize collected waste and service high-priority bins while avoiding overflow and minimizing travel/time costs.

The project compares:

- **DQN** (value-based),
- **REINFORCE** (policy gradient),
- **A2C** (actor-critic),
- **PPO** (policy gradient with clipping),

all trained on the same custom environment using Stable-Baselines3.

## Repository Structure

```text
environment/
  custom_env.py     # Custom Gymnasium environment (EcoTrackEnv)
  rendering.py      # 2D visualization / GUI using pygame or OpenGL

training/
  dqn_training.py   # Training script for DQN
  pg_training.py    # Training script for REINFORCE, A2C, PPO

models/
  dqn/              # Saved DQN models
  pg/               # Saved policy gradient models (REINFORCE, A2C, PPO)

notebooks/          # Analysis / plotting notebooks
results/            # CSV logs of runs, hyperparameter results
figures/            # Plots for the report (rewards, stability, etc.)
reports/            # Report files (template-filled, exported PDF)
scripts/            # Helper scripts (random agent demo, evaluation, etc.)

main.py             # Entry point to run best-performing agent with GUI
requirements.txt    # Python dependencies
README.md           # Project documentation
.gitignore          # Git ignored files
```