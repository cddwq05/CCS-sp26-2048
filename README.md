# Heuristic-Guided Decision Making in 2048

This project explores how different balances of **exploration** and **exploitation** affect agent performance in the game 2048.

The central question is:

> Should a 2048 agent rigidly follow a strong corner-based heuristic, or can performance improve when the agent occasionally deviates from that heuristic?

The project compares several agents, ranging from a fully random baseline to heuristic-based agents and controlled-deviation agents.

---

## Project Overview

2048 is a useful environment for studying heuristic-guided decision-making because each move has both immediate and long-term consequences. A common human strategy is to keep the largest tile in one corner while maintaining an ordered board structure. However, following this rule too rigidly may become harmful when the board is crowded or when the agent needs short-term flexibility to survive.

This project tests whether limited and state-dependent deviation from a strong heuristic can improve long-term game performance.

---

## Research Question

How does the balance between rigid heuristic following and controlled deviation affect performance in 2048?

More specifically, this project asks:

1. Does a rigid corner-based heuristic outperform random action selection?
2. Do random deviations from the heuristic help or hurt performance?
3. Can controlled, state-dependent deviations improve performance compared with rigid or randomly exploratory agents?
4. How do different deviation settings affect score, highest tile, win rate, and board stability?

---

## Agents

The project currently includes the following agents:

### RandomAgent

Chooses uniformly at random among all legal actions.

This serves as the baseline agent.

### RigidCornerAgent

Chooses the legal action with the highest one-step heuristic score.

The heuristic rewards boards that:

- keep the highest tile in the bottom-right corner,
- maintain monotonic ordering toward the bottom-right corner,
- preserve empty cells,
- preserve immediate merge opportunities.

### RandomDeviationAgent

Mostly follows the same heuristic as `RigidCornerAgent`, but with probability `p`, it randomly chooses a different legal action.

This agent tests whether simple random exploration improves or disrupts heuristic play.

### SoftmaxAgent

Uses a softmax distribution over heuristic action scores.

A lower temperature makes the agent behave more like the rigid heuristic agent, while a higher temperature makes it more exploratory.

### ControlledDeviationAgent

Follows the rigid heuristic most of the time, but enters a short flexibility window when the board is considered risky.

Risk can be defined by:

- having too few empty cells,
- having no immediate merge opportunities,
- showing a recent drop in heuristic board score.

During the flexibility window, the agent still chooses actions deterministically, but uses a modified heuristic with lower corner pressure and stronger survival-related terms.

---

## File Structure

```text
.
├── agents.py              # Agent implementations
├── env.py                 # Agent-facing 2048 environment
├── heuristics.py          # Board evaluation and heuristic feature functions
├── logic.py               # Low-level 2048 game mechanics
├── runner.py              # Functions for running and summarizing games
├── run_experiments.py     # Main experiment script
└── results/               # Generated CSV outputs after running experiments
```

---

## Code Description

### `logic.py`

Contains the low-level 2048 mechanics:

- board creation,
- tile spawning,
- movement in four directions,
- merging logic,
- game-state checking.

The board is a 4 × 4 grid, and the win condition is reaching the 2048 tile.

### `env.py`

Wraps the low-level game logic into an agent-friendly environment.

The main class is:

```python
Game2048Env
```

It supports:

- `reset()`
- `step(action)`
- `get_legal_actions()`
- `simulate_action(action)`
- `get_state()`
- `get_highest_tile()`
- `render()`
- `clone()`

The `step()` method returns:

```python
next_state, reward, done, info
```

### `heuristics.py`

Defines the board-evaluation function used by heuristic agents.

The current heuristic combines four normalized features:

```python
score =
    corner_weight * corner_score
  + monotonicity_weight * monotonicity_score
  + empty_weight * empty_cell_score
  + merge_weight * merge_score
```

The default weights are:

```python
DEFAULT_WEIGHTS = {
    "corner": 3.0,
    "monotonicity": 2.5,
    "empty": 2.0,
    "merge": 1.2,
}
```

The current implementation supports a fixed target corner:

```python
target_corner = "bottom_right"
```

### `agents.py`

Contains all agent implementations:

- `RandomAgent`
- `RigidCornerAgent`
- `RandomDeviationAgent`
- `SoftmaxAgent`
- `ControlledDeviationAgent`

Most agents implement:

```python
select_action(env)
```

and diagnostics-aware agents also implement:

```python
select_action_with_info(env)
```

The diagnostic information is used by the experiment runner to track deviation steps, risky states, selected actions, and action scores.

### `runner.py`

Contains functions for running games and computing summary statistics.

Main functions:

```python
play_one_game(agent, seed=None)
run_many_games(agent, n_episodes, base_seed=0)
summarize_results(results)
```

Tracked metrics include:

- final score,
- highest tile,
- number of steps,
- win rate,
- corner stability,
- score efficiency,
- deviation step rate,
- risky step rate,
- highest-tile milestone rates.

### `run_experiments.py`

Runs the main agent comparison.

By default, it uses:

```python
N_EPISODES = 500
BASE_SEED = 1000
```

The script compares several agents and saves the results as CSV files.

---

## Running the Project

This project only uses the Python standard library. No third-party packages are required.

To run the full experiment:

```bash
python run_experiments.py
```

The script will print summary statistics for each agent in the terminal.

---

## Output Files

After running:

```bash
python run_experiments.py
```

the script saves two CSV files in the `results/` folder:

```text
results/episode_raw_data.csv
results/agent_summary.csv
```

### `episode_raw_data.csv`

Contains one row per game episode.

Important columns include:

- `agent`
- `episode`
- `seed`
- `final_score`
- `highest_tile`
- `steps`
- `status`
- `won`
- `corner_stability`
- `score_efficiency`
- `deviation_steps`
- `deviation_step_rate`
- `risky_steps`
- `risky_step_rate`

This file is useful for plotting distributions, error bars, and per-agent comparisons.

### `agent_summary.csv`

Contains one summary row per agent.

Important columns include:

- `games`
- `mean_final_score`
- `median_final_score`
- `std_final_score`
- `mean_highest_tile`
- `max_highest_tile`
- `mean_steps`
- `mean_score_efficiency`
- `win_rate`
- `corner_stability_mean`
- `deviation_step_rate`
- `risky_step_rate`
- milestone counts and rates for highest tiles

Milestones currently include:

```text
>=128, >=256, >=512, >=1024, >=2048
```

---

## Current Experiment Settings

The main experiment currently uses conservative heuristic weights:

```python
CONSERVATIVE_WEIGHTS = {
    "corner": 3.5,
    "monotonicity": 2.8,
    "empty": 1.8,
    "merge": 1.0,
}
```

The controlled-deviation agent uses softer deviation weights:

```python
SOFT_DEVIATION_WEIGHTS = {
    "corner": 2.7,
    "monotonicity": 2.3,
    "empty": 2.3,
    "merge": 1.4,
}
```

The final controlled agent is configured as:

```python
ControlledDeviationAgent(
    rigid_weights=CONSERVATIVE_WEIGHTS,
    deviation_weights=SOFT_DEVIATION_WEIGHTS,
    empty_cell_threshold=1,
    use_merge_risk=False,
    use_score_drop_risk=False,
    flexibility_window=1,
)
```

This means the agent temporarily switches to a more flexible heuristic only when the number of empty cells is very low.

---

## Example Usage

Run one full comparison:

```bash
python run_experiments.py
```

Use an agent manually:

```python
from env import Game2048Env
from agents import RigidCornerAgent

env = Game2048Env(seed=0)
agent = RigidCornerAgent()

env.reset()

while not env.done:
    action = agent.select_action(env)
    if action is None:
        break
    env.step(action)

print(env.score)
print(env.get_highest_tile())
```

---

## Reproducibility

The experiment script uses fixed seeds to make results reproducible.

Each episode uses:

```python
seed = BASE_SEED + episode_index
```

The default base seed is:

```python
BASE_SEED = 1000
```

Randomized agents, such as `RandomAgent`, `RandomDeviationAgent`, and `SoftmaxAgent`, also use internal random seeds.

---

## Main Metrics

### Final Score

The final game score accumulated through tile merges.

### Highest Tile

The largest tile reached during a game.

### Win Rate

The fraction of games that reach the 2048 tile.

### Corner Stability

The fraction of steps in which the highest tile is located in the bottom-right corner.

### Score Efficiency

The final score divided by the number of steps.

### Deviation Step Rate

The fraction of steps in which an agent deviates from the rigid heuristic.

### Risky Step Rate

The fraction of steps in which the controlled-deviation agent detects a risky board state.

---

## Project Interpretation

The project is designed to test the idea that good 2048 play requires both stable long-term structure and occasional flexibility.

The rigid corner heuristic provides strong structure, but it may be too inflexible in some constrained board states. Random deviation adds exploration, but can disrupt the board structure. The controlled-deviation approach tries to combine both ideas by allowing flexibility only when the board state indicates risk.

---

## Limitations

This project uses one-step heuristic evaluation. The agents do not perform deep search, expectimax, reinforcement learning, or long-horizon planning.

The current heuristic is also specialized for the bottom-right corner. Other corners would require extending the corner and monotonicity scoring functions.

---

## Possible Future Extensions

Possible future improvements include:

- adding multi-step lookahead,
- implementing expectimax search,
- testing other target corners,
- adding more risk criteria for controlled deviation,
- visualizing full game trajectories,
- comparing heuristic agents with reinforcement-learning agents,
- running larger parameter sweeps over weights, temperatures, and flexibility windows.

---

## Course Project Context

This project was developed as a final project on heuristic-guided decision-making in 2048. It uses the game as a simple computational setting for studying how structured decision rules and controlled flexibility affect long-term performance.
