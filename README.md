# 2048 Heuristic-Guided Decision-Making Project

This repository contains the current core codebase for our CCS final project on **heuristic-guided decision-making in 2048**.

## Overview

The goal of this project is **not** to build the strongest possible 2048 AI.  
Instead, we aim to build a clean experimental framework for studying how a useful heuristic behaves under different decision strategies.

At the current stage, the codebase is designed to compare three types of agents:

- **RandomAgent**: chooses randomly among legal actions
- **RigidCornerAgent**: always follows a fixed corner-based heuristic
- **RandomDeviationAgent**: usually follows the corner heuristic, but occasionally deviates from it at random

This setup allows us to examine two main questions:

1. Does a strong heuristic improve performance in 2048?
2. Does occasionally breaking the heuristic help or hurt performance?

## Project Structure

### `logic.py`
Implements the low-level mechanics of 2048, including:

- board creation
- random tile spawning
- move and merge logic
- reward calculation from merges
- game-over and win detection

This file serves as the **rule engine** of the game.

### `env.py`
Defines the agent-facing game environment.

Main interface methods include:

- `reset()`
- `step(action)`
- `get_legal_actions()`
- `simulate_action()`
- `get_highest_tile()`
- `render()`
- `clone()`

This is the main **environment wrapper** used by agents and experiments.

### `heuristics.py`
Contains board evaluation functions used by heuristic-based agents, including:

- corner score
- monotonicity score
- empty-cell count
- merge potential
- weighted total board score

The current target corner is fixed to **bottom-right**.

### `agents.py`
Implements the agents currently used in the project:

- `RandomAgent`
- `RigidCornerAgent`
- `RandomDeviationAgent`

This file defines the **decision policies** for each agent.

### `runner.py`
Provides reusable experiment utilities, including:

- running a single full game
- running many games in batch
- summarizing performance results
- optionally logging trajectories

This file acts as the **backend for experiments**.

### `run_experiments.py`
The main script for running the current batch of experiments.

At the moment, it compares:

- `RandomAgent`
- `RigidCornerAgent`
- `RandomDeviationAgent` with different deviation probabilities

This is the main script for generating the current experimental results.

### `demo_agents.py`
A lightweight sanity-check script for quickly verifying that the agents and runner are working correctly.

## Current Status

The current codebase already supports:

- running the 2048 environment
- evaluating heuristic-based agents
- comparing baseline and control agents
- collecting summary metrics such as:
  - final score
  - number of steps
  - highest tile reached
  - win rate
  - corner stability

## Notes

- In the current setup, reaching **2048 (`WON`)** is treated as a terminal state.
- The codebase is modular, with each file serving a separate role.
- The purpose of this version is to support **baseline and control experiments**, rather than **reinforcement learning** or **DQN training**.

## Future Directions

Possible future extensions include:

- testing additional heuristic strategies
- changing the target corner or making it adaptive
- analyzing trajectory-level behavior in more detail
- extending the framework to RL-based agents for comparison

## How to Run

Make sure you have Python installed.

To run the main batch experiment:

```bash
python run_experiments.py

## Authors

CCS Final Project Team