"""Tests for the deterministic shortest-path oracle."""

from __future__ import annotations

from src.env.grid import GridConfig, GridNavigationEnv
from src.env.scenarios import central_route_closure
from src.evaluation.oracle import shortest_path


def test_shortest_path_finds_optimal_length() -> None:
    env = GridNavigationEnv(GridConfig(width=4, height=1, start=(0, 0), goal=(3, 0)))
    result = shortest_path(env)
    assert result.reachable
    assert result.path_length == 3
    assert result.path == ((0, 0), (1, 0), (2, 0), (3, 0))


def test_shortest_path_reports_unreachable_goal() -> None:
    env = GridNavigationEnv(
        GridConfig(
            width=3,
            height=1,
            start=(0, 0),
            goal=(2, 0),
            blocked_edges={((0, 0), (1, 0)), ((1, 0), (2, 0))},
        )
    )
    result = shortest_path(env)
    assert not result.reachable
    assert result.path_length is None


def test_disruption_changes_topology_but_goal_remains_reachable() -> None:
    scenario = central_route_closure(seed=11)
    env = scenario.make_env()
    before = shortest_path(env)
    env.apply_route_closure(blocked_edges=scenario.disrupted_edges)
    after = shortest_path(env)
    assert before.reachable
    assert after.reachable
    assert before.path_length == 6
    assert after.path_length == 8


def test_scenario_make_env_returns_independent_topologies() -> None:
    scenario = central_route_closure(seed=11)
    env_a = scenario.make_env()
    env_a.apply_route_closure(blocked_edges=scenario.disrupted_edges)
    env_b = scenario.make_env()
    assert shortest_path(env_a).path_length == 8
    assert shortest_path(env_b).path_length == 6


def test_blocked_edges_are_bidirectional() -> None:
    env = GridNavigationEnv(
        GridConfig(
            width=2,
            height=1,
            start=(1, 0),
            goal=(0, 0),
            blocked_edges={((0, 0), (1, 0))},
        )
    )
    assert not shortest_path(env).reachable


def test_shortest_path_start_at_goal_has_zero_return() -> None:
    env = GridNavigationEnv(GridConfig(width=2, height=2, start=(0, 0), goal=(0, 0)))
    result = shortest_path(env)
    assert result.reachable
    assert result.path_length == 0
    assert result.path_return == 0.0


def test_shortest_path_path_return_uses_reward_semantics() -> None:
    env = GridNavigationEnv(GridConfig(width=7, height=5, start=(0, 2), goal=(6, 2)))
    result = shortest_path(env)
    assert result.path_length == 6
    assert result.path_return == 15.0
