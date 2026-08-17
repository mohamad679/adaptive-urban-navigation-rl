"""Tests for required evaluation metrics."""

from __future__ import annotations

from src.evaluation.metrics import EpisodeRecord, episode_regret, route_efficiency, summarize_records


def test_route_efficiency_success_and_failure_handling() -> None:
    assert route_efficiency(True, 8, 6) == 0.75
    assert route_efficiency(False, None, 6) == 0.0


def test_regret_uses_max_steps_for_failed_episode() -> None:
    assert episode_regret(True, 8, 6, 40) == 2.0
    assert episode_regret(False, None, 6, 40) == 34.0


def test_summary_metrics_include_recovery_window_latencies() -> None:
    records = [
        EpisodeRecord(i, 11, "q_learning", "s", i >= 501, True, 1.0, 8, 8, 8, 1.0, 0.0)
        for i in range(498, 506)
    ]
    summary = summarize_records(
        records,
        disruption_episode=500,
        recovery_window=3,
        recovery_success_rate=1.0,
        recovery_efficiency=1.0,
    )
    assert summary.success_rate == 1.0
    assert summary.recovery_window_onset_latency == 1
    assert summary.recovery_window_confirmation_latency == 3
    assert summary.cumulative_regret == 0.0


def test_recovery_window_onset_latency_never_reports_zero() -> None:
    records = [
        EpisodeRecord(i, 11, "q_learning", "s", i >= 500, True, 1.0, 8, 8, 8, 1.0, 0.0)
        for i in range(500, 503)
    ]
    summary = summarize_records(
        records,
        disruption_episode=500,
        recovery_window=2,
        recovery_success_rate=1.0,
        recovery_efficiency=1.0,
    )
    assert summary.recovery_window_onset_latency == 1
    assert summary.recovery_window_confirmation_latency == 1
