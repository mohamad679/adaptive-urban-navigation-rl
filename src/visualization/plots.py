"""Figures generated from saved experiment outputs."""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from statistics import mean
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env.scenarios import central_route_closure
from src.evaluation.aggregate import aggregate_experiment
from src.evaluation.io import load_episode_records
from src.evaluation.metrics import EpisodeRecord


def plot_environment(output_path: Path) -> None:
    """Plot the default environment and disrupted route segment."""

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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for agent_type, agent_records in _group_by_agent(records).items():
        xs, ys = _rolling_by_episode(agent_records, "episodic_return", window=25)
        ax.plot(xs, ys, label=agent_type)
    disruption_episode = min(record.episode for record in records if record.disrupted)
    ax.axvline(disruption_episode, color="#d62728", linestyle="--", label="Disruption")
    ax.set_xlabel("Training episode completed before evaluation")
    ax.set_ylabel("Rolling mean greedy evaluation return")
    ax.set_title("Evaluation performance across training episodes")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_recovery(records: list[EpisodeRecord], output_path: Path) -> None:
    """Plot rolling route efficiency around the disruption."""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for agent_type, agent_records in _group_by_agent(records).items():
        xs, ys = _rolling_by_episode(agent_records, "route_efficiency", window=25)
        ax.plot(xs, ys, label=agent_type)
    disruption_episode = min(record.episode for record in records if record.disrupted)
    ax.axvline(disruption_episode, color="#d62728", linestyle="--", label="Disruption")
    ax.set_xlabel("Training episode completed before evaluation")
    ax.set_ylabel("Rolling mean greedy evaluation route efficiency")
    ax.set_title("Recovery after route disruption")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_agent_comparison(aggregate: dict[str, object], output_path: Path) -> None:
    """Plot aggregate comparison metrics by agent."""

    by_agent = aggregate["by_agent"]  # type: ignore[index]
    agent_names = sorted(by_agent)
    metric_names = ["success_rate", "mean_route_efficiency"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, metric_names):
        means = [by_agent[agent][f"{metric}_mean"] for agent in agent_names]
        sds = [by_agent[agent][f"{metric}_sd"] for agent in agent_names]
        ax.bar(agent_names, means, yerr=sds, color=["#4c78a8", "#f58518"][: len(agent_names)])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
    fig.suptitle("Agent comparison across fixed seeds")
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
        aggregate = aggregate_experiment(
            metrics_dir,
            disruption_episode=500,
            recovery_window=25,
            recovery_success_rate=0.8,
            recovery_efficiency=0.75,
            run_dirs=run_dirs,
        )
    plot_environment(figures_dir / "figure_1_environment.png")
    plot_learning_curve(records, figures_dir / "figure_2_learning_curve.png")
    plot_recovery(records, figures_dir / "figure_3_recovery_after_disruption.png")
    plot_agent_comparison(aggregate, figures_dir / "figure_4_agent_comparison.png")


def _group_by_agent(records: Iterable[EpisodeRecord]) -> dict[str, list[EpisodeRecord]]:
    grouped: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    return grouped


def _rolling_by_episode(records: list[EpisodeRecord], field: str, window: int) -> tuple[list[int], list[float]]:
    by_episode: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_episode[record.episode].append(float(getattr(record, field)))
    episodes = sorted(by_episode)
    values = [mean(by_episode[episode]) for episode in episodes]
    rolled = [
        mean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]
    return episodes, rolled
