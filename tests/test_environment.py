"""Tests for deterministic grid navigation transitions."""

from __future__ import annotations

from src.env.grid import GridConfig, GridNavigationEnv, RewardConfig


def make_env() -> GridNavigationEnv:
    return GridNavigationEnv(
        GridConfig(
            width=3,
            height=3,
            start=(0, 0),
            goal=(2, 0),
            blocked_cells={(1, 1)},
            blocked_edges={((0, 0), (0, 1))},
            rewards=RewardConfig(goal=20.0, movement=-1.0, invalid=-5.0),
            seed=123,
        )
    )


def test_reset_returns_start_state() -> None:
    env = make_env()
    env.step(3)
    assert env.reset() == (0, 0)


def test_attempt_to_leave_grid_is_blocked() -> None:
    env = make_env()
    result = env.step(2)
    assert result.state == (0, 0)
    assert result.reward == -5.0
    assert not result.terminated
    assert result.info["invalid"] is True


def test_blocked_cell_movement_is_blocked() -> None:
    env = make_env()
    env.step(3)
    result = env.step(1)
    assert result.state == (1, 0)
    assert result.reward == -5.0
    assert result.info["reason"] == "blocked_or_out_of_bounds"


def test_blocked_edge_movement_is_blocked() -> None:
    env = make_env()
    result = env.step(1)
    assert result.state == (0, 0)
    assert result.reward == -5.0
    assert result.info["reason"] == "blocked_edge"


def test_deterministic_valid_transition_and_reward() -> None:
    env_a = make_env()
    env_b = make_env()
    assert env_a.step(3) == env_b.step(3)
    assert env_a.state == (1, 0)
    assert env_a.step(3).reward == 20.0
    assert env_a.terminated


def test_terminal_goal_behaviour_is_absorbing() -> None:
    env = make_env()
    env.step(3)
    result = env.step(3)
    assert result.state == (2, 0)
    assert result.reward == 20.0
    assert result.terminated
    after_goal = env.step(2)
    assert after_goal.state == (2, 0)
    assert after_goal.reward == 0.0
    assert after_goal.terminated
