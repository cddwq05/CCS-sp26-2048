"""Run a terminal-friendly comparison of 2048 agents.

Run:
    python run_experiments.py
"""
import csv
import os

from agents import (
    ControlledDeviationAgent,
    RandomAgent,
    RandomDeviationAgent,
    RigidCornerAgent,
    SoftmaxAgent,
)
from runner import run_many_games


N_EPISODES = 500
BASE_SEED = 1000

CONSERVATIVE_WEIGHTS = {
    "corner": 3.5,
    "monotonicity": 2.8,
    "empty": 1.8,
    "merge": 1.0,
}

BALANCED_WEIGHTS = {
    "corner": 3.0,
    "monotonicity": 2.5,
    "empty": 2.0,
    "merge": 1.2,
}

AGGRESSIVE_WEIGHTS = {
    "corner": 2.6,
    "monotonicity": 2.2,
    "empty": 1.8,
    "merge": 1.6,
}

SOFT_DEVIATION_WEIGHTS = {
    "corner": 2.7,
    "monotonicity": 2.3,
    "empty": 2.3,
    "merge": 1.4,
}

def main():
    experiments = [
        (
            "RandomAgent",
            RandomAgent(seed=0),
        ),
        (
            "RigidCornerAgent conservative",
            RigidCornerAgent(weights=CONSERVATIVE_WEIGHTS),
        ),
        (
            "RandomDeviationAgent p=0.05",
            RandomDeviationAgent(
                p=0.05,
                seed=0,
                weights=CONSERVATIVE_WEIGHTS,
            ),
        ),
        (
            "RandomDeviationAgent p=0.10",
            RandomDeviationAgent(
                p=0.10,
                seed=0,
                weights=CONSERVATIVE_WEIGHTS,
            ),
        ),
        (
            "RandomDeviationAgent p=0.20",
            RandomDeviationAgent(
                p=0.20,
                seed=0,
                weights=CONSERVATIVE_WEIGHTS,
            ),
        ),

        (
            "Final controlled agent",
            ControlledDeviationAgent(
                rigid_weights=CONSERVATIVE_WEIGHTS,
                deviation_weights=SOFT_DEVIATION_WEIGHTS,
                empty_cell_threshold=1,
                use_merge_risk=False,
                use_score_drop_risk=False,
                flexibility_window=1,
            ),
        ),
        (
            "Softmax conservative temp=0.10",
            SoftmaxAgent(
                temperature=0.10,
                seed=BASE_SEED,
                weights=CONSERVATIVE_WEIGHTS,
            ),
        ),
        (
            "Softmax conservative temp=0.25",
            SoftmaxAgent(
                temperature=0.25,
                seed=BASE_SEED,
                weights=CONSERVATIVE_WEIGHTS,
            ),
        ),
        (
            "Softmax conservative temp=0.50",
            SoftmaxAgent(
                temperature=0.50,
                seed=BASE_SEED,
                weights=CONSERVATIVE_WEIGHTS,
            ),
        ),
    ]

    print(f"Running {N_EPISODES} games per agent")

    def _save_csv(path, rows):
        """Save a list of dictionaries as a CSV file."""
        if not rows:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        fieldnames = sorted(rows[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    all_episode_rows = []
    summary_rows = []

    for name, agent in experiments:
        result = run_many_games(agent, n_episodes=N_EPISODES, base_seed=BASE_SEED)
        summary = result["summary"]

        print(f"\n{name}")
        print("-" * len(name))
        _print_summary(summary)

        # Save one row per episode for error bars / distribution plots
        for episode_index, episode_result in enumerate(result["results"]):
            row = {
                "agent": name,
                "episode": episode_index,
                "seed": BASE_SEED + episode_index,
                "final_score": episode_result["final_score"],
                "highest_tile": episode_result["highest_tile"],
                "steps": episode_result["steps"],
                "status": episode_result["status"],
                "won": episode_result["won"],
                "corner_stability": episode_result["corner_stability"],
                "deviation_steps": episode_result.get("deviation_steps", 0),
                "risky_steps": episode_result.get("risky_steps", 0),
            }

            # Derived per-episode metrics
            row["score_efficiency"] = (
                episode_result["final_score"] / episode_result["steps"]
                if episode_result["steps"] > 0 else 0
            )

            row["deviation_step_rate"] = (
                episode_result.get("deviation_steps", 0) / episode_result["steps"]
                if episode_result["steps"] > 0 else 0
            )

            row["risky_step_rate"] = (
                episode_result.get("risky_steps", 0) / episode_result["steps"]
                if episode_result["steps"] > 0 else 0
            )

            all_episode_rows.append(row)

        # Save one row per agent summary
        summary_row = {
            "agent": name,
            "games": summary["games"],
            "mean_final_score": summary["mean_final_score"],
            "median_final_score": summary["median_final_score"],
            "std_final_score": summary["std_final_score"],
            "mean_highest_tile": summary["mean_highest_tile"],
            "max_highest_tile": summary["max_highest_tile"],
            "mean_steps": summary["mean_steps"],
            "mean_score_efficiency": summary["mean_score_efficiency"],
            "win_rate": summary["win_rate"],
            "corner_stability_mean": summary["corner_stability_mean"],
            "deviation_step_rate": summary.get("deviation_step_rate", 0),
            "risky_step_rate": summary.get("risky_step_rate", 0),
        }

        for milestone, count in summary["highest_tile_distribution"].items():
            summary_row[f"n_ge_{milestone}"] = count
            summary_row[f"rate_ge_{milestone}"] = summary["highest_tile_distribution_rate"][milestone]

        summary_rows.append(summary_row)

    _save_csv("results/episode_raw_data.csv", all_episode_rows)
    _save_csv("results/agent_summary.csv", summary_rows)

    print("\nSaved raw episode data to results/episode_raw_data.csv")
    print("Saved summary data to results/agent_summary.csv")


def _print_summary(summary):
    print(f"Games: {summary['games']}")
    print(f"Mean final score: {summary['mean_final_score']:.2f}")
    print(f"Median final score: {summary['median_final_score']:.2f}")
    print(f"Std final score: {summary['std_final_score']:.2f}")
    print(f"Mean highest tile: {summary['mean_highest_tile']:.2f}")
    print(f"Max highest tile: {summary['max_highest_tile']}")
    print(f"Mean steps: {summary['mean_steps']:.2f}")
    print(f"Mean score efficiency: {summary['mean_score_efficiency']:.2f}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Corner stability mean: {summary['corner_stability_mean']:.2%}")
    if "deviation_step_rate" in summary:
        print(f"Deviation step rate: {summary['deviation_step_rate']:.2%}")
        print(f"Risky step rate: {summary['risky_step_rate']:.2%}")

        if summary.get("risk_reason_totals"):
            print("Risk reasons:")
            for reason, count in sorted(summary["risk_reason_totals"].items()):
                print(f"  {reason}: {count}")
    print("Highest tile milestones:")

    counts = summary["highest_tile_distribution"]
    rates = summary["highest_tile_distribution_rate"]
    for milestone in sorted(counts):
        print(f"  >= {milestone:4d}: {counts[milestone]:4d} ({rates[milestone]:6.2%})")



if __name__ == "__main__":
    main()
