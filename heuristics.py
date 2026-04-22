"""Simple board-evaluation helpers for 2048 agents.

The current project uses a fixed bottom-right target corner. These utilities
are intentionally small and easy to adjust for course experiments.
"""


DEFAULT_WEIGHTS = {
    "corner": 1000.0,
    "monotonicity": 1.0,
    "empty": 50.0,
    "merge": 25.0,
}


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
        return highest_tile
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
    """Combine simple board features into a single heuristic score."""
    _require_bottom_right(target_corner)
    weights = resolve_weights(weights)

    return (
        weights.get("corner", 0) * corner_score(board, target_corner)
        + weights.get("monotonicity", 0) * monotonicity_score(board, target_corner)
        + weights.get("empty", 0) * count_empty_cells(board)
        + weights.get("merge", 0) * merge_potential(board)
    )


def _ordered_pair_score(first, second):
    """Reward nondecreasing neighbors and penalize decreasing neighbors."""
    if first <= second:
        return second - first
    return -(first - second)


def _require_bottom_right(target_corner):
    if target_corner != "bottom_right":
        raise ValueError("Only target_corner='bottom_right' is supported for now.")
