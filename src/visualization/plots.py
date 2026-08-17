"""Figures generated from saved experiment outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from src.env.scenarios import central_route_closure
from src.evaluation.aggregate import aggregate_experiment
from src.evaluation.io import load_episode_records
from src.evaluation.metrics import EpisodeRecord


def display_agent_label(agent_type: str) -> str:
    """Return publication-friendly agent labels."""

    labels = {
        "q_learning": "Q-learning",
        "dqn": "DQN",
    }
    return labels.get(agent_type, agent_type)


def disruption_boundary(records: list[EpisodeRecord]) -> float:
    """Return the boundary after the last pre-disruption episode."""

    disrupted_episodes = [record.episode for record in records if record.disrupted]
    if not disrupted_episodes:
        raise ValueError("Cannot plot disruption marker because no disrupted episodes were found")
    return min(disrupted_episodes) - 0.5


def disruption_boundary_label(boundary: float) -> str:
    """Return a label for a disruption boundary between integer episodes."""

    return f"Route closure after episode {int(boundary)}"


def rolling_seed_aggregate(
    records: list[EpisodeRecord],
    field: str,
    *,
    window: int = 25,
) -> tuple[list[int], list[float], list[float]]:
    """Aggregate trailing rolling means per seed, then mean and SD across seeds."""

    by_seed: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_seed[record.seed].append(record)

    values_by_episode: dict[int, list[float]] = defaultdict(list)
    for seed_records in by_seed.values():
        ordered = sorted(seed_records, key=lambda record: record.episode)
        raw_values = [float(getattr(record, field)) for record in ordered]
        for index, record in enumerate(ordered):
            rolled = mean(raw_values[max(0, index - window + 1) : index + 1])
            values_by_episode[record.episode].append(rolled)

    episodes = sorted(values_by_episode)
    means = [mean(values_by_episode[episode]) for episode in episodes]
    sds = [
        stdev(values_by_episode[episode]) if len(values_by_episode[episode]) > 1 else 0.0
        for episode in episodes
    ]
    return episodes, means, sds


def plot_environment(output_path: Path) -> None:
    """Plot the default environment and disrupted route segment."""

    plt = _load_pyplot()
    scenario = central_route_closure()
    env = scenario.make_env()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(-0.5, env.config.width - 0.5)
    ax.set_ylim(env.config.height - 0.5, -0.5)
    ax.set_xticks(range(env.config.width))
    ax.set_yticks(range(env.config.height))
    ax.grid(True, color="#c7c7c7", linewidth=0.8)
    ax.scatter(*env.config.start, marker="o", s=180, color="#1f77b4", label="Start")
    ax.scatter(*env.config.goal, marker="*", s=260, color="#2ca02c", label="Goal")
    for source, target in scenario.disrupted_edges:
        ax.plot(
            [source[0], target[0]],
            [source[1], target[1]],
            color="#d62728",
            linewidth=5,
            solid_capstyle="round",
            label="Closed after disruption",
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper center", ncol=3)
    ax.set_title("Deterministic grid navigation with central route disruption")
    ax.set_aspect("equal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_learning_curve(records: list[EpisodeRecord], output_path: Path) -> None:
    """Plot rolling evaluation return from saved episode records."""

    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    boundary = disruption_boundary(records)
    for agent_type, agent_records in _group_by_agent(records).items():
        xs, ys, sds = rolling_seed_aggregate(agent_records, "episodic_return", window=25)
        label = display_agent_label(agent_type)
        ax.plot(xs, ys, label=label)
        lower = [y - sd for y, sd in zip(ys, sds)]
        upper = [y + sd for y, sd in zip(ys, sds)]
        ax.fill_between(xs, lower, upper, alpha=0.18)
    ax.axvline(
        boundary,
        color="#d62728",
        linestyle="--",
        label=disruption_boundary_label(boundary),
    )
    ax.set_xlabel("Training episode completed before evaluation")
    ax.set_ylabel("25-episode rolling mean greedy evaluation return")
    ax.set_title("Evaluation performance across training episodes")
    _add_uncertainty_note(ax)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_recovery(records: list[EpisodeRecord], output_path: Path) -> None:
    """Plot rolling route efficiency around the disruption."""

    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    boundary = disruption_boundary(records)
    for agent_type, agent_records in _group_by_agent(records).items():
        xs, ys, sds = rolling_seed_aggregate(agent_records, "route_efficiency", window=25)
        label = display_agent_label(agent_type)
        ax.plot(xs, ys, label=label)
        lower = [y - sd for y, sd in zip(ys, sds)]
        upper = [y + sd for y, sd in zip(ys, sds)]
        ax.fill_between(xs, lower, upper, alpha=0.18)
    ax.axvline(
        boundary,
        color="#d62728",
        linestyle="--",
        label=disruption_boundary_label(boundary),
    )
    ax.set_xlabel("Training episode completed before evaluation")
    ax.set_ylabel("25-episode rolling mean greedy route efficiency")
    ax.set_title("Recovery after route disruption")
    _add_uncertainty_note(ax)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_agent_comparison(aggregate: dict[str, object], output_path: Path) -> None:
    """Plot aggregate comparison metrics by agent."""

    plt = _load_pyplot()
    by_agent = aggregate["by_agent"]  # type: ignore[index]
    agent_names = sorted(by_agent)
    display_names = [display_agent_label(agent) for agent in agent_names]
    metric_names = ["success_rate", "mean_route_efficiency"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, metric_names):
        means = [by_agent[agent][f"{metric}_mean"] for agent in agent_names]
        sds = [by_agent[agent][f"{metric}_sd"] for agent in agent_names]
        ax.bar(display_names, means, yerr=sds, color=["#4c78a8", "#f58518"][: len(agent_names)])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
    fig.suptitle("Agent comparison (mean ± SD across fixed seeds)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_all_records(metrics_dir: Path) -> list[EpisodeRecord]:
    """Load all saved episode records below a metrics directory."""

    records: list[EpisodeRecord] = []
    for path in sorted(metrics_dir.glob("*/episodes.csv")):
        records.extend(load_episode_records(path))
    return records


def generate_all_figures(
    metrics_dir: Path,
    figures_dir: Path,
    aggregate: dict[str, object] | None = None,
    run_dirs: list[Path] | None = None,
) -> None:
    """Generate the required project figures from saved outputs."""

    records = (
        [record for run_dir in run_dirs or [] for record in load_episode_records(run_dir / "episodes.csv")]
        if run_dirs is not None
        else load_all_records(metrics_dir)
    )
    if aggregate is None:
        aggregate = _aggregate_from_saved_run_configs(metrics_dir, run_dirs=run_dirs)
    plot_environment(figures_dir / "figure_1_environment.png")
    plot_learning_curve(records, figures_dir / "figure_2_learning_curve.png")
    plot_recovery(records, figures_dir / "figure_3_recovery_after_disruption.png")
    plot_agent_comparison(aggregate, figures_dir / "figure_4_agent_comparison.png")


def _group_by_agent(records: Iterable[EpisodeRecord]) -> dict[str, list[EpisodeRecord]]:
    grouped: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    return grouped


def _add_uncertainty_note(ax: object) -> None:
    ax.text(
        0.01,
        0.02,
        "Shaded bands: ±1 SD across fixed seeds",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
    )


def _aggregate_from_saved_run_configs(
    metrics_dir: Path,
    *,
    run_dirs: list[Path] | None,
) -> dict[str, Any]:
    selected_run_dirs = (
        run_dirs
        if run_dirs is not None
        else sorted(path.parent for path in metrics_dir.glob("*/episodes.csv"))
    )
    configs = [_load_run_experiment_config(run_dir / "config.json") for run_dir in selected_run_dirs]
    if not configs:
        raise ValueError("Cannot infer aggregate settings because no run config files were found")

    first_config = configs[0]
    required_keys = [
        "disruption_episode",
        "recovery_window",
        "recovery_success_rate",
        "recovery_efficiency",
    ]
    for config in configs[1:]:
        for key in required_keys:
            if config[key] != first_config[key]:
                raise ValueError(f"Cannot aggregate runs with inconsistent {key} values")

    return aggregate_experiment(
        metrics_dir,
        disruption_episode=int(first_config["disruption_episode"]),
        recovery_window=int(first_config["recovery_window"]),
        recovery_success_rate=float(first_config["recovery_success_rate"]),
        recovery_efficiency=float(first_config["recovery_efficiency"]),
        run_dirs=run_dirs,
    )


def _load_run_experiment_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Cannot infer aggregate settings because {path} does not exist")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError(f"Cannot infer aggregate settings because {path} has no experiment section")
    return experiment


def _load_pyplot() -> object:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
