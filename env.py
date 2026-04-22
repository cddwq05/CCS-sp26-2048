"""Agent-facing 2048 environment.

Example:
    from env import Game2048Env

    env = Game2048Env(seed=0)
    state = env.reset()
    next_state, reward, done, info = env.step("left")
"""

import random

import logic


class Game2048Env:
    """A small, dependency-free 2048 environment for agents and experiments."""

    ACTIONS = ("up", "down", "left", "right")

    def __init__(self, seed=None, start_tiles=2):
        """Create a 2048 environment.

        Args:
            seed: Optional seed for reproducible tile spawning.
            start_tiles: Number of random tiles placed on reset.
        """
        if start_tiles < 0 or start_tiles > logic.BOARD_SIZE * logic.BOARD_SIZE:
            raise ValueError("start_tiles must fit on the board")

        self.seed = seed
        self.start_tiles = start_tiles
        self.rng = random.Random(seed)
        self.board = logic.new_board()
        self.score = 0
        self.done = False
        self.status = "GAME NOT OVER"

    def reset(self):
        """Reset to a fresh board and return the initial state."""
        self.board = logic.new_board()
        self.score = 0
        self.done = False
        self.status = "GAME NOT OVER"

        for _ in range(self.start_tiles):
            logic.add_new_tile(self.board, self.rng)

        self._update_status()
        return self.get_state()

    def step(self, action):
        """Apply one action and return (next_state, reward, done, info)."""
        action = self._normalize_action(action)

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action!r}. Use one of {self.ACTIONS}.")

        if self.done:
            return self.get_state(), 0, True, self._make_info(changed=False)

        new_board, changed, reward = self._apply_action(self.board, action)

        if changed:
            self.board = new_board
            self.score += reward
            self._update_status()

            if not self.done:
                logic.add_new_tile(self.board, self.rng)
                self._update_status()
        else:
            reward = 0
            self._update_status()

        return self.get_state(), reward, self.done, self._make_info(changed)

    def get_legal_actions(self):
        """Return actions that would change the current board."""
        if self.done:
            return []

        legal_actions = []
        for action in self.ACTIONS:
            _, changed, _ = self._apply_action(self.board, action)
            if changed:
                legal_actions.append(action)

        return legal_actions

    def is_terminal(self):
        """Return True when the game has been won or lost."""
        self._update_status()
        return self.done

    def get_highest_tile(self):
        """Return the highest tile currently on the board."""
        return max(max(row) for row in self.board)

    def simulate_action(self, action):
        """Return the board, changed flag, and reward for an action.

        This does not mutate the environment and does not add a random tile.
        It is useful for one-step heuristic agents.
        """
        action = self._normalize_action(action)

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action!r}. Use one of {self.ACTIONS}.")

        return self._apply_action(self.board, action)

    def render(self):
        """Print a simple text view of the board and score."""
        print(f"Score: {self.score}")
        print(f"Status: {self.status}")
        for row in self.board:
            print(" ".join(f"{value:4d}" if value else "   ." for value in row))
        print()

    def clone(self):
        """Return an independent copy of this environment state."""
        cloned = Game2048Env(seed=self.seed, start_tiles=self.start_tiles)
        cloned.rng.setstate(self.rng.getstate())
        cloned.board = self.get_state()
        cloned.score = self.score
        cloned.done = self.done
        cloned.status = self.status
        return cloned

    def get_state(self):
        """Return a copy of the current board."""
        return logic.copy_board(self.board)

    def _update_status(self):
        self.status = logic.get_current_state(self.board)
        self.done = self.status in ("WON", "LOST")

    def _make_info(self, changed):
        return {
            "changed": changed,
            "score": self.score,
            "status": self.status,
            "highest_tile": max(max(row) for row in self.board),
            "legal_actions": self.get_legal_actions() if not self.done else [],
        }

    @classmethod
    def _apply_action(cls, board, action):
        board_copy = logic.copy_board(board)

        if action == "up":
            return logic.move_up(board_copy)
        if action == "down":
            return logic.move_down(board_copy)
        if action == "left":
            return logic.move_left(board_copy)
        if action == "right":
            return logic.move_right(board_copy)

        raise ValueError(f"Invalid action {action!r}. Use one of {cls.ACTIONS}.")

    @staticmethod
    def _normalize_action(action):
        if isinstance(action, str):
            return action.lower()
        return action
