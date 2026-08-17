"""Evaluation metrics for adaptive navigation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class EpisodeRecord:
    """Per-episode outcome saved by training and evaluation runs."""

    episode: int
    seed: int
    agent_type: str
    scenario: str
    disrupted: bool
    success: bool
    episodic_return: float
    steps: int
    path_length: int | None
    optimal_path_length: int | None
    route_efficiency: float
    regret: float
    exploration_rate: float | None = None


@dataclass(frozen=True)
class ImmediatePostDisruptionEvaluation:
    """Greedy no-update evaluation of the pre-disruption policy after closure."""

    seed: int
    agent_type: str
    scenario: str
    disrupted: bool
    post_disruption_training_episodes: int
    success: bool
    episodic_return: float
    steps: int
    path_length: int | None
    optimal_path_length: int | None
    optimal_path_return: float | None
    route_efficiency: float
    regret: float
    exploration_rate: float | None = None


@dataclass(frozen=True)
class SummaryMetrics:
    """Aggregate metrics for one run or a filtered window."""

    success_rate: float
    mean_episodic_return: float
    mean_path_length: float | None
    mean_route_efficiency: float
    recovery_window_onset_latency: int | None
    recovery_window_confirmation_latency: int | None
    cumulative_regret: float


def route_efficiency(success: bool, path_length: int | None, optimal_path_length: int | None) -> float:
    """Compute optimal_path_length / action_count, assigning 0 to failures."""

    if not success or path_length is None or optimal_path_length is None or path_length <= 0:
        return 0.0
    return optimal_path_length / path_length


def episode_regret(
    success: bool,
    path_length: int | None,
    optimal_path_length: int | None,
    max_steps: int,
) -> float:
    """Compute extra actions relative to optimal, using max_steps for failed episodes."""

    if optimal_path_length is None:
        return float(max_steps)
    realized_length = path_length if success and path_length is not None else max_steps
    return float(max(realized_length - optimal_path_length, 0))


def summarize_records(
    records: list[EpisodeRecord],
    *,
    disruption_episode: int,
    recovery_window: int = 25,
    recovery_success_rate: float = 0.8,
    recovery_efficiency: float = 0.75,
) -> SummaryMetrics:
    """Summarize records and compute predefined recovery-window latency.

    Recovery-window onset latency is the number of post-disruption training
    episodes completed before the start of the first evaluation window that
    reaches the predefined success-rate and route-efficiency thresholds.
    """

    if not records:
        raise ValueError("Cannot summarize an empty record list")
    successes = [1.0 if record.success else 0.0 for record in records]
    returns = [record.episodic_return for record in records]
    successful_lengths = [record.path_length for record in records if record.path_length is not None]
    efficiencies = [record.route_efficiency for record in records]
    regrets = [record.regret for record in records]
    window = recovery_window_latency(
        records,
        disruption_episode=disruption_episode,
        recovery_window=recovery_window,
        recovery_success_rate=recovery_success_rate,
        recovery_efficiency=recovery_efficiency,
    )
    return SummaryMetrics(
        success_rate=mean(successes),
        mean_episodic_return=mean(returns),
        mean_path_length=mean(successful_lengths) if successful_lengths else None,
        mean_route_efficiency=mean(efficiencies),
        recovery_window_onset_latency=window[0],
        recovery_window_confirmation_latency=window[1],
        cumulative_regret=sum(regrets),
    )


def recovery_window_latency(
    records: list[EpisodeRecord],
    *,
    disruption_episode: int,
    recovery_window: int,
    recovery_success_rate: float,
    recovery_efficiency: float,
) -> tuple[int | None, int | None]:
    """Return onset and confirmation latencies for the first recovery window."""

    ordered = sorted(records, key=lambda record: record.episode)
    by_episode = {record.episode: record for record in ordered}
    disrupted_episodes = [record.episode for record in ordered if record.disrupted]
    if not disrupted_episodes:
        return None, None
    first_disrupted_episode = min(disrupted_episodes)
    max_episode = max(by_episode)
    for start in range(first_disrupted_episode, max_episode - recovery_window + 2):
        window = [by_episode[index] for index in range(start, start + recovery_window) if index in by_episode]
        if len(window) != recovery_window:
            continue
        success_rate = mean(1.0 if record.success else 0.0 for record in window)
        efficiency = mean(record.route_efficiency for record in window)
        if success_rate >= recovery_success_rate and efficiency >= recovery_efficiency:
            onset_latency = max(1, start - disruption_episode)
            confirmation_latency = max(onset_latency, start + recovery_window - 1 - disruption_episode)
            return onset_latency, confirmation_latency
    return None, None
