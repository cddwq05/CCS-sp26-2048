"""Run a terminal-friendly comparison of 2048 agents.

Run:
    python run_experiments.py
"""

from agents import RandomAgent, RandomDeviationAgent, RigidCornerAgent
from runner import run_many_games


N_EPISODES = 100
BASE_SEED = 1000


def main():
    experiments = [
        ("RandomAgent", RandomAgent(seed=0)),
        ("RigidCornerAgent", RigidCornerAgent()),
        ("RandomDeviationAgent(p=0.05)", RandomDeviationAgent(p=0.05, seed=0)),
        ("RandomDeviationAgent(p=0.10)", RandomDeviationAgent(p=0.10, seed=0)),
        ("RandomDeviationAgent(p=0.20)", RandomDeviationAgent(p=0.20, seed=0)),
    ]

    print(f"Running {N_EPISODES} games per agent")

    for name, agent in experiments:
        result = run_many_games(agent, n_episodes=N_EPISODES, base_seed=BASE_SEED)
        print(f"\n{name}")
        print("-" * len(name))
        _print_summary(result["summary"])


def _print_summary(summary):
    print(f"Games: {summary['games']}")
    print(f"Mean final score: {summary['mean_final_score']:.2f}")
    print(f"Median final score: {summary['median_final_score']:.2f}")
    print(f"Std final score: {summary['std_final_score']:.2f}")
    print(f"Mean highest tile: {summary['mean_highest_tile']:.2f}")
    print(f"Max highest tile: {summary['max_highest_tile']}")
    print(f"Mean steps: {summary['mean_steps']:.2f}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Corner stability mean: {summary['corner_stability_mean']:.2%}")
    print("Highest tile milestones:")

    counts = summary["highest_tile_distribution"]
    rates = summary["highest_tile_distribution_rate"]
    for milestone in sorted(counts):
        print(f"  >= {milestone:4d}: {counts[milestone]:4d} ({rates[milestone]:6.2%})")


if __name__ == "__main__":
    main()
