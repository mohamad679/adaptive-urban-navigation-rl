"""Aggregate saved multi-seed experiment outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.evaluation.io import load_episode_records, load_immediate_post_disruption_evaluation, write_json
from src.evaluation.metrics import EpisodeRecord, ImmediatePostDisruptionEvaluation, summarize_records


def aggregate_experiment(
    metrics_dir: Path,
    *,
    disruption_episode: int,
    recovery_window: int,
    recovery_success_rate: float,
    recovery_efficiency: float,
    run_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Aggregate all run directories below ``metrics_dir``."""

    records_by_agent_seed: dict[tuple[str, int], list[EpisodeRecord]] = defaultdict(list)
    immediate_evaluations: list[ImmediatePostDisruptionEvaluation] = []
    selected_run_dirs = run_dirs if run_dirs is not None else sorted(path.parent for path in metrics_dir.glob("*/episodes.csv"))
    episodes_paths = (
        [run_dir / "episodes.csv" for run_dir in selected_run_dirs]
    )
    for episodes_path in episodes_paths:
        if not episodes_path.exists():
            continue
        for record in load_episode_records(episodes_path):
            records_by_agent_seed[(record.agent_type, record.seed)].append(record)
    for run_dir in selected_run_dirs:
        immediate_path = run_dir / "immediate_post_disruption_evaluation.json"
        if immediate_path.exists():
            immediate_evaluations.append(load_immediate_post_disruption_evaluation(immediate_path))

    run_summaries: list[dict[str, Any]] = []
    for (agent_type, seed), records in sorted(records_by_agent_seed.items()):
        summary = summarize_records(
            records,
            disruption_episode=disruption_episode,
            recovery_window=recovery_window,
            recovery_success_rate=recovery_success_rate,
            recovery_efficiency=recovery_efficiency,
        )
        run_summaries.append({"agent_type": agent_type, "seed": seed, **asdict(summary)})

    aggregate: dict[str, Any] = {
        "runs": run_summaries,
        "by_agent": {},
        "immediate_post_disruption_evaluations": [
            asdict(evaluation) for evaluation in sorted(immediate_evaluations, key=lambda item: (item.agent_type, item.seed))
        ],
        "metric_definitions": {
            "success_rate": "Mean fraction of episodes reaching the goal within max_steps_per_episode.",
            "mean_episodic_return": "Mean undiscounted return over episodes.",
            "mean_path_length": "Mean action count over successful episodes only, including invalid actions before success.",
            "mean_route_efficiency": "Mean optimal_path_length / realized_path_length; failed episodes receive 0.",
            "recovery_window_onset_latency": (
                "Number of post-disruption training episodes completed before the start of the first evaluation "
                "window satisfying the predefined success-rate and route-efficiency thresholds; null if unrecovered."
            ),
            "recovery_window_confirmation_latency": (
                "Number of post-disruption training episodes completed by the end of the first qualifying "
                "recovery window; null if unrecovered."
            ),
            "immediate_post_disruption_robustness": (
                "Greedy no-update evaluation of the learned pre-disruption policy after applying the route closure "
                "and before any post-disruption training update."
            ),
            "cumulative_regret": "Sum over episodes of realized path length minus optimal length; failures use max_steps.",
        },
    }
    for agent_type in sorted({item["agent_type"] for item in run_summaries}):
        items = [item for item in run_summaries if item["agent_type"] == agent_type]
        agent_immediate = [item for item in immediate_evaluations if item.agent_type == agent_type]
        aggregate["by_agent"][agent_type] = _summarize_agent(items, agent_immediate)
    write_json(metrics_dir / "aggregate_summary.json", aggregate)
    return aggregate


def _summarize_agent(
    items: list[dict[str, Any]],
    immediate_evaluations: list[ImmediatePostDisruptionEvaluation],
) -> dict[str, Any]:
    metrics = [
        "success_rate",
        "mean_episodic_return",
        "mean_path_length",
        "mean_route_efficiency",
        "cumulative_regret",
    ]
    output: dict[str, Any] = {"n_seeds": len(items)}
    for metric in metrics:
        values = [item[metric] for item in items if item[metric] is not None]
        output[f"{metric}_mean"] = mean(values) if values else None
        output[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0 if values else None
    for latency in ["recovery_window_onset_latency", "recovery_window_confirmation_latency"]:
        values = [item[latency] for item in items if item[latency] is not None]
        output[f"{latency}_mean"] = mean(values) if values else None
        output[f"{latency}_sd"] = stdev(values) if len(values) > 1 else 0.0 if values else None
    output["recovery_window_unrecovered_seeds"] = [
        item["seed"] for item in items if item["recovery_window_onset_latency"] is None
    ]
    if immediate_evaluations:
        output["immediate_post_disruption_success_rate"] = mean(
            1.0 if item.success else 0.0 for item in immediate_evaluations
        )
        output["immediate_post_disruption_route_efficiency_mean"] = mean(
            item.route_efficiency for item in immediate_evaluations
        )
        output["immediate_post_disruption_regret_mean"] = mean(item.regret for item in immediate_evaluations)
    else:
        output["immediate_post_disruption_success_rate"] = None
        output["immediate_post_disruption_route_efficiency_mean"] = None
        output["immediate_post_disruption_regret_mean"] = None
    return output
