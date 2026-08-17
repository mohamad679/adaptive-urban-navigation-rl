"""Tests for benchmark configuration loading and CLI override resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_experiments import (
    make_scenario,
    resolve_benchmark_config,
    selected_agent_types,
    validate_benchmark_config,
)


def write_config(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "seeds": [11, 22, 33, 44, 55],
        "episodes": 1000,
        "disruption_episode": 500,
        "max_steps_per_episode": 40,
        "recovery_window": 25,
        "recovery_success_rate": 0.8,
        "recovery_efficiency": 0.75,
        "scenario": "central_route_closure",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_json_values_are_loaded() -> None:
    config = resolve_benchmark_config(Path("configs/default_experiment.json"))
    assert config.seeds == (11, 22, 33, 44, 55)
    assert config.scenario == "central_route_closure"
    assert config.experiment.episodes == 1000
    assert config.experiment.disruption_episode == 500
    assert config.experiment.max_steps_per_episode == 40
    assert config.experiment.recovery_window == 25
    assert config.experiment.recovery_success_rate == 0.8
    assert config.experiment.recovery_efficiency == 0.75


def test_seeds_come_from_json_not_python_constant(tmp_path: Path) -> None:
    path = write_config(tmp_path / "experiment.json", seeds=[101, 202])
    config = resolve_benchmark_config(path)
    assert config.seeds == (101, 202)


def test_cli_style_overrides_win_over_json(tmp_path: Path) -> None:
    path = write_config(tmp_path / "experiment.json", episodes=30, disruption_episode=10, max_steps_per_episode=12)
    config = resolve_benchmark_config(path, episodes=40, disruption_episode=15, max_steps=20)
    assert config.experiment.episodes == 40
    assert config.experiment.disruption_episode == 15
    assert config.experiment.max_steps_per_episode == 20


def test_unsupported_scenario_name_fails(tmp_path: Path) -> None:
    path = write_config(tmp_path / "experiment.json", scenario="unknown_city")
    with pytest.raises(ValueError, match="Unsupported scenario"):
        resolve_benchmark_config(path)
    with pytest.raises(ValueError, match="Unsupported scenario"):
        make_scenario("unknown_city", seed=11)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeds": []},
        {"episodes": 0},
        {"disruption_episode": -1},
        {"disruption_episode": 1000},
        {"max_steps_per_episode": 0},
        {"recovery_window": 0},
        {"recovery_success_rate": 1.5},
        {"recovery_efficiency": -0.1},
    ],
)
def test_invalid_benchmark_configuration_fails(overrides: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "seeds": [11, 22, 33, 44, 55],
        "episodes": 1000,
        "disruption_episode": 500,
        "max_steps_per_episode": 40,
        "recovery_window": 25,
        "recovery_success_rate": 0.8,
        "recovery_efficiency": 0.75,
        "scenario": "central_route_closure",
    }
    payload.update(overrides)
    with pytest.raises(ValueError):
        validate_benchmark_config(payload)


def test_agent_selection_remains_cli_controlled() -> None:
    assert selected_agent_types(include_dqn=False) == ("q_learning",)
    assert selected_agent_types(include_dqn=True) == ("q_learning", "dqn")
