"""Agent implementations for the 2048 environment."""

import random

from heuristics import (
    count_empty_cells,
    evaluate_board,
    format_weights,
    merge_potential,
    resolve_weights,
)


TIE_BREAK_PRIORITY = {
    "down": 4,
    "right": 3,
    "left": 2,
    "up": 1,
}


class RandomAgent:
    """Choose uniformly at random among legal actions."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select_action(self, env):
        """Return one legal action, or None if no legal actions exist."""
        action, _ = self.select_action_with_info(env)
        return action

    def select_action_with_info(self, env):
        """Return (action, info) for diagnostics-aware runners."""
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None, {
                "agent": self.__class__.__name__,
                "legal_actions": [],
                "is_random_action": True,
                "is_deviation_step": None,
            }

        action = self.rng.choice(legal_actions)
        return action, {
            "agent": self.__class__.__name__,
            "legal_actions": legal_actions,
            "chosen_action": action,
            "is_random_action": True,
            "is_deviation_step": None,
        }


class RigidCornerAgent:
    """Choose the legal action with the best one-step heuristic score."""

    def __init__(self, target_corner="bottom_right", weights=None):
        self.target_corner = target_corner
        self.weights = resolve_weights(weights)

    def select_action(self, env):
        """Return the best legal action, or None if no legal actions exist."""
        action, _ = self.select_action_with_info(env)
        return action

    def select_action_with_info(self, env):
        """Return (action, info) with action scores for diagnostics."""
        scores = self.score_legal_actions(env)
        if not scores:
            return None, {
                "agent": self.__class__.__name__,
                "action_scores": {},
                "chosen_action": None,
                "is_deviation_step": False,
                "weights": self.get_weight_summary(),
            }

        action = _best_action_from_scores(scores)
        return action, {
            "agent": self.__class__.__name__,
            "action_scores": scores,
            "chosen_action": action,
            "is_deviation_step": False,
            "weights": self.get_weight_summary(),
        }

    def score_legal_actions(self, env):
        """Return a dictionary mapping each legal action to its board score."""
        action_scores = {}

        for action in env.get_legal_actions():
            board_after, changed, _ = env.simulate_action(action)
            if changed:
                action_scores[action] = evaluate_board(
                    board_after,
                    target_corner=self.target_corner,
                    weights=self.weights,
                )

        return action_scores

    def get_weight_summary(self):
        """Return a display-friendly summary of active heuristic weights."""
        return format_weights(self.weights)


class RandomDeviationAgent:
    """Mostly follow the rigid heuristic, but sometimes choose another action."""

    def __init__(self, p=0.1, seed=None, target_corner="bottom_right", weights=None):
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")

        self.p = p
        self.rng = random.Random(seed)
        self.rigid_agent = RigidCornerAgent(
            target_corner=target_corner,
            weights=weights,
        )

    def select_action(self, env):
        """Return a heuristic action with occasional random deviation."""
        action, _ = self.select_action_with_info(env)
        return action

    def select_action_with_info(self, env):
        """Return (action, info) describing any random deviation."""
        scores = self.rigid_agent.score_legal_actions(env)
        if not scores:
            return None, {
                "agent": self.__class__.__name__,
                "action_scores": {},
                "rigid_best_action": None,
                "chosen_action": None,
                "is_deviation_step": False,
                "deviation_probability": self.p,
                "weights": self.get_weight_summary(),
            }

        best_action = _best_action_from_scores(scores)
        legal_actions = list(scores.keys())

        if len(legal_actions) == 1:
            return best_action, self._make_info(
                scores,
                best_action,
                best_action,
                is_deviation_step=False,
            )

        if self.rng.random() >= self.p:
            return best_action, self._make_info(
                scores,
                best_action,
                best_action,
                is_deviation_step=False,
            )

        other_actions = [action for action in legal_actions if action != best_action]
        if not other_actions:
            return best_action, self._make_info(
                scores,
                best_action,
                best_action,
                is_deviation_step=False,
            )

        chosen_action = self.rng.choice(other_actions)
        return chosen_action, self._make_info(
            scores,
            best_action,
            chosen_action,
            is_deviation_step=True,
        )

    def get_weight_summary(self):
        """Return a display-friendly summary of active heuristic weights."""
        return self.rigid_agent.get_weight_summary()

    def _make_info(self, scores, best_action, chosen_action, is_deviation_step):
        return {
            "agent": self.__class__.__name__,
            "action_scores": scores,
            "rigid_best_action": best_action,
            "chosen_action": chosen_action,
            "is_deviation_step": is_deviation_step,
            "deviation_probability": self.p,
            "weights": self.get_weight_summary(),
        }


class ControlledDeviationAgent:
    """Rigid heuristic agent with short survival-focused deviations.

    A state is risky when it has too few empty cells, no immediate merge
    potential, or a recent strictly decreasing heuristic-score trend. When risk
    is detected, the agent opens a temporary flexibility window. During that
    window it still chooses deterministically, but scores actions with lower
    corner pressure and stronger survival terms.
    """

    DEFAULT_DEVIATION_WEIGHTS = {
        # Compared with the rigid defaults, corner pressure is reduced while
        # empty-space and merge opportunities are emphasized for survival.
        "corner": 250.0,
        "monotonicity": 1.0,
        "empty": 100.0,
        "merge": 75.0,
    }

    def __init__(
        self,
        target_corner="bottom_right",
        rigid_weights=None,
        deviation_weights=None,
        empty_cell_threshold=2,
        use_merge_risk=True,
        use_score_drop_risk=True,
        score_drop_window=3,
        flexibility_window=2,
    ):
        if empty_cell_threshold < 0:
            raise ValueError("empty_cell_threshold must be non-negative")
        if score_drop_window < 1:
            raise ValueError("score_drop_window must be at least 1")
        if flexibility_window < 0:
            raise ValueError("flexibility_window must be non-negative")

        self.target_corner = target_corner
        self.rigid_weights = resolve_weights(rigid_weights)
        self.deviation_weights = self.DEFAULT_DEVIATION_WEIGHTS.copy()
        if deviation_weights:
            self.deviation_weights.update(deviation_weights)
        self.empty_cell_threshold = empty_cell_threshold
        self.use_merge_risk = use_merge_risk
        self.use_score_drop_risk = use_score_drop_risk
        self.score_drop_window = score_drop_window
        self.flexibility_window = flexibility_window
        self.rigid_agent = RigidCornerAgent(
            target_corner=target_corner,
            weights=self.rigid_weights,
        )
        self.reset()

    def reset(self):
        """Clear episode-specific risk history and flexibility state."""
        self.score_history = []
        self.flex_moves_remaining = 0
        self._last_board_signature = None

    def select_action(self, env):
        """Return the selected action, or None if no legal actions exist."""
        action, _ = self.select_action_with_info(env)
        return action

    def select_action_with_info(self, env):
        """Return (action, info) with risk and deviation diagnostics."""
        board = env.get_state()
        current_score = self._current_board_score(board)
        self._record_board_score(board, current_score)

        risky, risk_reasons, risk_metrics = self._is_risky_state(board)
        if risky:
            self.flex_moves_remaining = max(
                self.flex_moves_remaining,
                self.flexibility_window,
            )

        rigid_scores = self.rigid_agent.score_legal_actions(env)
        if not rigid_scores:
            return None, self._make_info(
                action_scores={},
                chosen_action=None,
                rigid_best_action=None,
                mode="rigid",
                is_risky_state=risky,
                risk_reasons=risk_reasons,
                flex_moves_remaining=self.flex_moves_remaining,
                current_board_score=current_score,
                risk_metrics=risk_metrics,
            )

        rigid_best_action = _best_action_from_scores(rigid_scores)
        use_deviation = self.flex_moves_remaining > 0

        if use_deviation:
            action_scores = self._score_actions_with_weights(
                env,
                self.deviation_weights,
            )
            mode = "deviation"
        else:
            action_scores = rigid_scores
            mode = "rigid"

        chosen_action = _best_action_from_scores(action_scores)
        decision_flex_moves_remaining = self.flex_moves_remaining

        if use_deviation and self.flex_moves_remaining > 0:
            self.flex_moves_remaining -= 1

        return chosen_action, self._make_info(
            action_scores=action_scores,
            chosen_action=chosen_action,
            rigid_best_action=rigid_best_action,
            mode=mode,
            is_risky_state=risky,
            risk_reasons=risk_reasons,
            flex_moves_remaining=decision_flex_moves_remaining,
            current_board_score=current_score,
            risk_metrics=risk_metrics,
        )

    def get_rigid_weight_summary(self):
        """Return a display-friendly summary of rigid heuristic weights."""
        return format_weights(self.rigid_weights)

    def get_deviation_weight_summary(self):
        """Return a display-friendly summary of deviation heuristic weights."""
        return format_weights(self.deviation_weights)

    def _current_board_score(self, board):
        return evaluate_board(
            board,
            target_corner=self.target_corner,
            weights=self.rigid_weights,
        )

    def _record_board_score(self, board, score):
        board_signature = tuple(tuple(row) for row in board)
        if board_signature == self._last_board_signature:
            return

        self.score_history.append(score)
        max_history = self.score_drop_window + 1
        if len(self.score_history) > max_history:
            self.score_history = self.score_history[-max_history:]
        self._last_board_signature = board_signature

    def _is_risky_state(self, board):
        risk_reasons = []
        empty_cells = count_empty_cells(board)
        merge_score = merge_potential(board)

        if empty_cells <= self.empty_cell_threshold:
            risk_reasons.append("low_empty_cells")
        if self.use_merge_risk and merge_score == 0:
            risk_reasons.append("no_merges")
        if self.use_score_drop_risk and self._score_drop_triggered():
            risk_reasons.append("score_drop")

        return bool(risk_reasons), risk_reasons, {
            "empty_cells": empty_cells,
            "merge_potential": merge_score,
        }

    def _score_drop_triggered(self):
        needed_scores = self.score_drop_window + 1
        if len(self.score_history) < needed_scores:
            return False

        recent_scores = self.score_history[-needed_scores:]
        return all(
            earlier > later
            for earlier, later in zip(recent_scores, recent_scores[1:])
        )

    def _score_actions_with_weights(self, env, weights):
        action_scores = {}

        for action in env.get_legal_actions():
            board_after, changed, _ = env.simulate_action(action)
            if changed:
                action_scores[action] = evaluate_board(
                    board_after,
                    target_corner=self.target_corner,
                    weights=weights,
                )

        return action_scores

    def _make_info(
        self,
        action_scores,
        chosen_action,
        rigid_best_action,
        mode,
        is_risky_state,
        risk_reasons,
        flex_moves_remaining,
        current_board_score,
        risk_metrics,
    ):
        return {
            "agent": self.__class__.__name__,
            "action_scores": action_scores,
            "chosen_action": chosen_action,
            "rigid_best_action": rigid_best_action,
            "mode": mode,
            "is_deviation_step": mode == "deviation",
            "is_risky_state": is_risky_state,
            "risk_reasons": risk_reasons,
            "flexibility_window": self.flexibility_window,
            "flex_moves_remaining": flex_moves_remaining,
            "rigid_weights": self.rigid_weights.copy(),
            "deviation_weights": self.deviation_weights.copy(),
            "rigid_weight_summary": self.get_rigid_weight_summary(),
            "deviation_weight_summary": self.get_deviation_weight_summary(),
            "current_board_score": current_board_score,
            "empty_cells": risk_metrics["empty_cells"],
            "merge_potential": risk_metrics["merge_potential"],
        }


def _best_action_from_scores(scores):
    """Choose the best-scoring action with deterministic priority ties."""
    return max(
        scores,
        key=lambda action: (scores[action], TIE_BREAK_PRIORITY[action]),
    )
