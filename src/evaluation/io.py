"""Input/output helpers for saved experiment results."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.evaluation.metrics import EpisodeRecord, ImmediatePostDisruptionEvaluation


def load_episode_records(path: Path) -> list[EpisodeRecord]:
    """Load episode records from a CSV file written by the experiment runner."""

    records: list[EpisodeRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                EpisodeRecord(
                    episode=int(row["episode"]),
                    seed=int(row["seed"]),
                    agent_type=row["agent_type"],
                    scenario=row["scenario"],
                    disrupted=row["disrupted"] == "True",
                    success=row["success"] == "True",
                    episodic_return=float(row["episodic_return"]),
                    steps=int(row["steps"]),
                    path_length=int(row["path_length"]) if row["path_length"] else None,
                    optimal_path_length=int(row["optimal_path_length"]) if row["optimal_path_length"] else None,
                    route_efficiency=float(row["route_efficiency"]),
                    regret=float(row["regret"]),
                    exploration_rate=float(row["exploration_rate"]) if row["exploration_rate"] else None,
                )
            )
    return records


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Write a JSON payload with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_immediate_post_disruption_evaluation(path: Path) -> ImmediatePostDisruptionEvaluation:
    """Load an immediate post-disruption evaluation JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ImmediatePostDisruptionEvaluation(
        seed=int(payload["seed"]),
        agent_type=payload["agent_type"],
        scenario=payload["scenario"],
        disrupted=bool(payload["disrupted"]),
        post_disruption_training_episodes=int(payload["post_disruption_training_episodes"]),
        success=bool(payload["success"]),
        episodic_return=float(payload["episodic_return"]),
        steps=int(payload["steps"]),
        path_length=int(payload["path_length"]) if payload["path_length"] is not None else None,
        optimal_path_length=int(payload["optimal_path_length"]) if payload["optimal_path_length"] is not None else None,
        optimal_path_return=float(payload["optimal_path_return"]) if payload["optimal_path_return"] is not None else None,
        route_efficiency=float(payload["route_efficiency"]),
        regret=float(payload["regret"]),
        exploration_rate=float(payload["exploration_rate"]) if payload["exploration_rate"] is not None else None,
    )


def write_records_csv(path: Path, records: list[EpisodeRecord]) -> None:
    """Write episode records as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
