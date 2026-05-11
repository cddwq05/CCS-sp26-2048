"""Simple board-evaluation helpers for 2048 agents.

The current project uses a fixed bottom-right target corner. These utilities
are intentionally small and easy to adjust for course experiments.
"""
import math

DEFAULT_WEIGHTS = {
    "corner": 3.0,
    "monotonicity": 2.5,
    "empty": 2.0,
    "merge": 1.2,
}

FEATURE_RANGES = {
    "corner": (0.0, 10.0),
    "monotonicity": (-21.0, 36.0),
    "empty": (0.0, 14.0),
    "merge": (0.0, 8.0),
}

def normalize_feature(value, feature_name):
    """Normalize one feature value to approximately 0–1 using observed ranges."""
    min_value, max_value = FEATURE_RANGES[feature_name]

    if max_value == min_value:
        return 0

    normalized = (value - min_value) / (max_value - min_value)

    # Keep values in [0, 1] in case future boards slightly exceed observed range.
    return max(0, min(1, normalized))

def get_default_weights():
    """Return a copy of the default heuristic weights."""
    return DEFAULT_WEIGHTS.copy()


def resolve_weights(weights=None):
    """Return default weights updated by any custom values."""
    active_weights = get_default_weights()
    if weights:
        active_weights.update(weights)
    return active_weights


def format_weights(weights=None):
    """Return a compact string for displaying active heuristic weights."""
    active_weights = resolve_weights(weights)
    return ", ".join(
        f"{name}={active_weights[name]}"
        for name in ("corner", "monotonicity", "empty", "merge")
    )


def get_highest_tile(board):
    """Return the maximum tile value on the board."""
    return max(max(row) for row in board)


def count_empty_cells(board):
    """Return the number of empty cells."""
    return sum(1 for row in board for value in row if value == 0)


def corner_score(board, target_corner="bottom_right"):
    """Reward boards whose highest tile is in the target corner."""
    _require_bottom_right(target_corner)

    highest_tile = get_highest_tile(board)
    if board[-1][-1] == highest_tile:
        return math.log2(highest_tile) 
    return 0


def monotonicity_score(board, target_corner="bottom_right"):
    """Score whether tile values increase toward the bottom-right corner.

    For the bottom-right target, rows should generally increase left to right
    and columns should generally increase top to bottom. Equal neighbors are
    treated as acceptable because they preserve merge opportunities.
    """
    _require_bottom_right(target_corner)

    score = 0

    for row in board:
        for left, right in zip(row, row[1:]):
            score += _ordered_pair_score(left, right)

    for col in range(len(board[0])):
        for row in range(len(board) - 1):
            top = board[row][col]
            bottom = board[row + 1][col]
            score += _ordered_pair_score(top, bottom)

    return score


def merge_potential(board):
    """Count neighboring equal non-zero tiles that could merge soon."""
    score = 0

    for row in board:
        for left, right in zip(row, row[1:]):
            if left != 0 and left == right:
                score += 1

    for col in range(len(board[0])):
        for row in range(len(board) - 1):
            if board[row][col] != 0 and board[row][col] == board[row + 1][col]:
                score += 1

    return score


def evaluate_board(board, target_corner="bottom_right", weights=None):
    """Combine normalized board features into a single heuristic score."""
    _require_bottom_right(target_corner)
    weights = resolve_weights(weights)

    raw_corner = corner_score(board, target_corner)
    raw_monotonicity = monotonicity_score(board, target_corner)
    raw_empty = count_empty_cells(board)
    raw_merge = merge_potential(board)

    corner = normalize_feature(raw_corner, "corner")
    monotonicity = normalize_feature(raw_monotonicity, "monotonicity")
    empty = normalize_feature(raw_empty, "empty")
    merge = normalize_feature(raw_merge, "merge")

    return (
        weights.get("corner", 0) * corner
        + weights.get("monotonicity", 0) * monotonicity
        + weights.get("empty", 0) * empty
        + weights.get("merge", 0) * merge
    )


def _ordered_pair_score(first, second):
    """Reward nondecreasing neighbors and penalize decreasing neighbors."""
    if first == 0 or second == 0:
        return 0
    return math.log2(second) - math.log2(first)


def _require_bottom_right(target_corner):
    if target_corner != "bottom_right":
        raise ValueError("Only target_corner='bottom_right' is supported for now.")
