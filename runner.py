"""Experiment runners for 2048 agents."""

import statistics

from env import Game2048Env
from heuristics import get_highest_tile


HIGHEST_TILE_MILESTONES = (128, 256, 512, 1024, 2048)


def play_one_game(agent, seed=None, record_trajectory=False, start_tiles=2):
    """Play one full game and return final metrics.

    Args:
        agent: Object with select_action(env).
        seed: Environment seed for reproducible tile spawning.
        record_trajectory: Whether to store per-step board transitions.
        start_tiles: Number of initial random tiles on reset.
    """
    env = Game2048Env(seed=seed, start_tiles=start_tiles)
    env.reset()

    trajectory = [] if record_trajectory else None
    steps = 0
    corner_steps = 0

    while not env.done:
        action, decision_info = _select_action_with_optional_info(agent, env)
        if action is None:
            env.is_terminal()
            break

        board_before = env.get_state()
        highest_tile_before = get_highest_tile(board_before)
        max_in_corner_before = _max_in_bottom_right_corner(board_before)

        board_after, reward, done, info = env.step(action)
        highest_tile_after = get_highest_tile(board_after)
        max_in_corner_after = _max_in_bottom_right_corner(board_after)
        steps += 1

        if max_in_corner_after:
            corner_steps += 1

        if record_trajectory:
            trajectory.append(
                {
                    "step": steps,
                    "action": action,
                    "board_before": board_before,
                    "board_after": [row[:] for row in board_after],
                    "highest_tile_before": highest_tile_before,
                    "highest_tile_after": highest_tile_after,
                    "max_in_corner_before": max_in_corner_before,
                    "max_in_corner_after": max_in_corner_after,
                    "is_deviation_step": decision_info.get("is_deviation_step"),
                    "decision_info": decision_info,
                    "reward": reward,
                    "score": info["score"],
                    "done": done,
                    "status": info["status"],
                }
            )

    corner_stability = corner_steps / steps if steps else 0

    return {
        "final_score": env.score,
        "highest_tile": env.get_highest_tile(),
        "steps": steps,
        "status": env.status,
        "won": env.status == "WON",
        "corner_stability": corner_stability,
        "trajectory": trajectory,
    }


def run_many_games(
    agent,
    n_episodes,
    base_seed=0,
    record_trajectory=False,
    start_tiles=2,
):
    """Run several games and return raw results plus summary statistics."""
    results = []

    for episode_index in range(n_episodes):
        result = play_one_game(
            agent,
            seed=base_seed + episode_index,
            record_trajectory=record_trajectory,
            start_tiles=start_tiles,
        )
        results.append(result)

    return {
        "results": results,
        "summary": summarize_results(results),
    }


def summarize_results(results):
    """Compute simple summary metrics for completed game results."""
    if not results:
        return {
            "games": 0,
            "mean_final_score": 0,
            "median_final_score": 0,
            "std_final_score": 0,
            "mean_highest_tile": 0,
            "max_highest_tile": 0,
            "mean_steps": 0,
            "win_rate": 0,
            "highest_tile_distribution": _empty_milestone_dict(),
            "highest_tile_distribution_rate": _empty_milestone_dict(),
            "corner_stability_mean": 0,
        }

    final_scores = [result["final_score"] for result in results]
    highest_tiles = [result["highest_tile"] for result in results]
    steps = [result["steps"] for result in results]
    wins = [result["won"] for result in results]
    corner_stabilities = [result.get("corner_stability", 0) for result in results]
    tile_distribution = _highest_tile_distribution(highest_tiles)

    return {
        "games": len(results),
        "mean_final_score": statistics.mean(final_scores),
        "median_final_score": statistics.median(final_scores),
        "std_final_score": _population_std(final_scores),
        "mean_highest_tile": statistics.mean(highest_tiles),
        "max_highest_tile": max(highest_tiles),
        "mean_steps": statistics.mean(steps),
        "win_rate": sum(wins) / len(wins),
        "highest_tile_distribution": tile_distribution,
        "highest_tile_distribution_rate": {
            milestone: count / len(results)
            for milestone, count in tile_distribution.items()
        },
        "corner_stability_mean": statistics.mean(corner_stabilities),
    }


def _population_std(values):
    if len(values) < 2:
        return 0
    return statistics.pstdev(values)


def _select_action_with_optional_info(agent, env):
    if hasattr(agent, "select_action_with_info"):
        action, info = agent.select_action_with_info(env)
        return action, info or {}

    return agent.select_action(env), {}


def _max_in_bottom_right_corner(board):
    return board[-1][-1] == get_highest_tile(board)


def _highest_tile_distribution(highest_tiles):
    return {
        milestone: sum(1 for tile in highest_tiles if tile >= milestone)
        for milestone in HIGHEST_TILE_MILESTONES
    }


def _empty_milestone_dict():
    return {milestone: 0 for milestone in HIGHEST_TILE_MILESTONES}
