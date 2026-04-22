"""Agent implementations for the 2048 environment."""

import random

from heuristics import evaluate_board, format_weights, resolve_weights


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


def _best_action_from_scores(scores):
    """Choose the best-scoring action with deterministic priority ties."""
    return max(
        scores,
        key=lambda action: (scores[action], TIE_BREAK_PRIORITY[action]),
    )
