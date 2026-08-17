from __future__ import annotations

import pytest

from src.evaluation.metrics import EpisodeRecord
from src.visualization.plots import (
    display_agent_label,
    disruption_boundary,
    disruption_boundary_label,
    rolling_seed_aggregate,
)


def make_record(
    *,
    episode: int,
    seed: int,
    disrupted: bool = False,
    episodic_return: float = 0.0,
    route_efficiency: float = 0.0,
) -> EpisodeRecord:
    return EpisodeRecord(
        episode=episode,
        seed=seed,
        agent_type="q_learning",
        scenario="central_route_closure",
        disrupted=disrupted,
        success=True,
        episodic_return=episodic_return,
        steps=1,
        path_length=1,
        optimal_path_length=1,
        route_efficiency=route_efficiency,
        regret=0.0,
    )


def test_disruption_boundary_uses_first_disrupted_episode() -> None:
    records = [
        make_record(episode=500, seed=11, disrupted=False),
        make_record(episode=501, seed=11, disrupted=True),
        make_record(episode=502, seed=11, disrupted=True),
    ]

    assert disruption_boundary(records) == 500.5
    assert disruption_boundary_label(500.5) == "Route closure after episode 500"


def test_rolling_seed_aggregate_rolls_within_seed_before_cross_seed_stats() -> None:
    records = [
        make_record(episode=1, seed=1, episodic_return=10.0),
        make_record(episode=2, seed=1, episodic_return=20.0),
        make_record(episode=3, seed=1, episodic_return=30.0),
        make_record(episode=1, seed=2, episodic_return=30.0),
        make_record(episode=2, seed=2, episodic_return=30.0),
        make_record(episode=3, seed=2, episodic_return=30.0),
    ]

    episodes, means, sds = rolling_seed_aggregate(records, "episodic_return", window=2)

    assert episodes == [1, 2, 3]
    assert means == pytest.approx([20.0, 22.5, 27.5])
    assert sds == pytest.approx([14.1421356237, 10.6066017178, 3.5355339059])


def test_rolling_seed_aggregate_single_seed_sd_is_zero() -> None:
    records = [
        make_record(episode=1, seed=11, route_efficiency=0.5),
        make_record(episode=2, seed=11, route_efficiency=1.0),
    ]

    episodes, means, sds = rolling_seed_aggregate(records, "route_efficiency", window=2)

    assert episodes == [1, 2]
    assert means == pytest.approx([0.5, 0.75])
    assert sds == [0.0, 0.0]


def test_display_agent_label_maps_public_agent_names() -> None:
    assert display_agent_label("q_learning") == "Q-learning"
    assert display_agent_label("dqn") == "DQN"
    assert display_agent_label("other") == "other"
