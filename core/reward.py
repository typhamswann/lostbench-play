"""Reward functions.

Composed via vf.Rubric in wanderbench.load_environment. After the submit_guess
redesign, the agent gets a single explicit declaration of "I've arrived"; the
episode ends there and rewards are computed against that final position.

  guess_reward       1.0   graduated by guess-to-goal distance (the main signal)
  efficiency_reward  0.2   gated: optimal_steps / actual_steps, only on success
  format_reward      0.0   metric only: fraction of valid tool calls
"""
from __future__ import annotations

from typing import Any

from core.tasks import _haversine_m

DEFAULT_GOAL_RADIUS_M = 25.0


def guess_reward(state: dict | None = None, **_: Any) -> float:
    """Linear from 1.0 (guess on the goal) to 0.0 (guess at or past the initial
    start-to-goal distance). Returns 0 if the agent never submitted."""
    if state is None:
        return 0.0
    sim = state.get("sim")
    if sim is None or not getattr(sim, "guess_submitted", False):
        return 0.0
    initial = sim.task.initial_distance_m
    if initial <= 0:
        return 1.0
    err = _haversine_m(sim.guess_lat, sim.guess_lng,
                       sim.task.goal_lat, sim.task.goal_lng)
    return max(0.0, min(1.0, 1.0 - err / initial))


def efficiency_reward(state: dict | None = None, **_: Any) -> float:
    """Only counts when the agent submitted AND the guess was within goal radius."""
    if state is None:
        return 0.0
    sim = state.get("sim")
    if sim is None or not getattr(sim, "guess_submitted", False):
        return 0.0
    err = _haversine_m(sim.guess_lat, sim.guess_lng,
                       sim.task.goal_lat, sim.task.goal_lng)
    if err > DEFAULT_GOAL_RADIUS_M:
        return 0.0
    if sim.steps_taken <= 0:
        return 0.0
    return min(1.0, sim.task.optimal_steps / sim.steps_taken)


def format_reward(state: dict | None = None, **_: Any) -> float:
    if state is None:
        return 0.0
    sim = state.get("sim")
    if sim is None or sim.turn_count == 0:
        return 0.0
    invalid = state.get("format_errors", 0)
    return max(0.0, 1.0 - invalid / sim.turn_count)
