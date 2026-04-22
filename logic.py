"""Low-level 2048 board mechanics.

The environment in env.py uses these functions as the move engine. Move
functions return (new_board, changed, score_gained).
"""

import random

BOARD_SIZE = 4
WIN_TILE = 2048


def new_board():
    """Return a fresh empty 2048 board."""
    return [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]


def copy_board(mat):
    """Return a deep copy of a 2048 board."""
    return [row[:] for row in mat]


def start_game():
    """Create a board for the legacy command-line game."""
    mat = new_board()

    print("Commands are as follows : ")
    print("'W' or 'w' : Move Up")
    print("'S' or 's' : Move Down")
    print("'A' or 'a' : Move Left")
    print("'D' or 'd' : Move Right")

    add_new_tile(mat)
    add_new_tile(mat)
    return mat


def findEmpty(mat):
    """Find the first empty (0) cell in the grid."""
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if mat[i][j] == 0:
                return i, j
    return None, None


def get_empty_cells(mat):
    """Return a list of all empty cell coordinates."""
    empty_cells = []

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if mat[i][j] == 0:
                empty_cells.append((i, j))

    return empty_cells


def add_new_number(mat, value, rng=None):
    """Add value to a random empty cell. Returns True if a tile was added."""
    rng = rng or random
    empty_cells = get_empty_cells(mat)

    if not empty_cells:
        return False

    r, c = rng.choice(empty_cells)
    mat[r][c] = value
    return True


def add_new_2(mat, rng=None):
    """Add a new 2 tile to a random empty cell."""
    return add_new_number(mat, 2, rng)


def add_new_4(mat, rng=None):
    """Add a new 4 tile to a random empty cell."""
    return add_new_number(mat, 4, rng)


def add_new_tile(mat, rng=None):
    """Add a new tile using standard 2048 probabilities: 90% 2, 10% 4."""
    rng = rng or random

    if rng.random() < 0.9:
        return add_new_2(mat, rng)
    return add_new_4(mat, rng)


def get_current_state(mat):
    """Return WON, LOST, or GAME NOT OVER for the given board."""
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if mat[i][j] == WIN_TILE:
                return "WON"

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if mat[i][j] == 0:
                return "GAME NOT OVER"

    for i in range(BOARD_SIZE - 1):
        for j in range(BOARD_SIZE - 1):
            if mat[i][j] == mat[i + 1][j] or mat[i][j] == mat[i][j + 1]:
                return "GAME NOT OVER"

    for j in range(BOARD_SIZE - 1):
        if mat[BOARD_SIZE - 1][j] == mat[BOARD_SIZE - 1][j + 1]:
            return "GAME NOT OVER"

    for i in range(BOARD_SIZE - 1):
        if mat[i][BOARD_SIZE - 1] == mat[i + 1][BOARD_SIZE - 1]:
            return "GAME NOT OVER"

    return "LOST"


def compress(mat):
    """Slide non-zero tiles in each row to the left."""
    changed = False
    new_mat = new_board()

    for i in range(BOARD_SIZE):
        pos = 0
        for j in range(BOARD_SIZE):
            if mat[i][j] != 0:
                new_mat[i][pos] = mat[i][j]

                if j != pos:
                    changed = True
                pos += 1

    return new_mat, changed


def merge(mat):
    """Merge equal neighboring tiles to the left and return merge score."""
    changed = False
    score_gained = 0

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE - 1):
            if mat[i][j] == mat[i][j + 1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j + 1] = 0
                score_gained += mat[i][j]
                changed = True

    return mat, changed, score_gained


def reverse(mat):
    """Reverse every row."""
    new_mat = []
    for i in range(BOARD_SIZE):
        new_mat.append([])
        for j in range(BOARD_SIZE):
            new_mat[i].append(mat[i][BOARD_SIZE - 1 - j])
    return new_mat


def transpose(mat):
    """Transpose rows and columns."""
    new_mat = []
    for i in range(BOARD_SIZE):
        new_mat.append([])
        for j in range(BOARD_SIZE):
            new_mat[i].append(mat[j][i])
    return new_mat


def move_left(grid):
    """Move left and return (new_board, changed, score_gained)."""
    new_grid, changed1 = compress(grid)
    new_grid, changed2, score_gained = merge(new_grid)
    new_grid, _ = compress(new_grid)

    changed = changed1 or changed2
    return new_grid, changed, score_gained


def move_right(grid):
    """Move right and return (new_board, changed, score_gained)."""
    new_grid = reverse(grid)
    new_grid, changed, score_gained = move_left(new_grid)
    new_grid = reverse(new_grid)

    return new_grid, changed, score_gained


def move_up(grid):
    """Move up and return (new_board, changed, score_gained)."""
    new_grid = transpose(grid)
    new_grid, changed, score_gained = move_left(new_grid)
    new_grid = transpose(new_grid)

    return new_grid, changed, score_gained


def move_down(grid):
    """Move down and return (new_board, changed, score_gained)."""
    new_grid = transpose(grid)
    new_grid, changed, score_gained = move_right(new_grid)
    new_grid = transpose(new_grid)

    return new_grid, changed, score_gained
