"""Training and evaluation pipeline for route-disruption experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
from typing import Protocol

from src.agents.q_learning import QLearningConfig, TabularQLearningAgent
from src.env.grid import GridNavigationEnv
from src.env.scenarios import DisruptionScenario, central_route_closure
from src.evaluation.metrics import (
    EpisodeRecord,
    ImmediatePostDisruptionEvaluation,
    episode_regret,
    route_efficiency,
    summarize_records,
)
from src.evaluation.oracle import shortest_path


class LearningAgent(Protocol):
    """Protocol shared by tabular and DQN agents."""

    exploration_rate: float

    def select_action(self, state: tuple[int, int], *, greedy: bool = False) -> int:
        """Select an action."""

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        terminated: bool,
    ) -> None:
        """Update from one transition."""

    def end_episode(self) -> None:
        """Run end-of-episode maintenance."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Configurable route-disruption experiment schedule."""

    episodes: int = 1000
    disruption_episode: int = 500
    max_steps_per_episode: int = 40
    evaluation_interval: int = 25
    recovery_window: int = 25
    recovery_success_rate: float = 0.8
    recovery_efficiency: float = 0.75


@dataclass(frozen=True)
class RunResult:
    """Outputs from one seeded experiment run."""

    config: dict[str, object]
    records: list[EpisodeRecord]
    summary: dict[str, object]
    immediate_post_disruption_evaluation: ImmediatePostDisruptionEvaluation | None


def run_q_learning_experiment(seed: int, config: ExperimentConfig, scenario: DisruptionScenario | None = None) -> RunResult:
    """Run one seeded tabular Q-learning route-disruption experiment."""

    scenario = scenario or central_route_closure(seed=seed)
    env = scenario.make_env()
    agent_config = QLearningConfig(seed=seed)
    agent = TabularQLearningAgent(
        env.states,
        env.n_actions,
        agent_config,
    )
    return run_training_loop(
        env=env,
        agent=agent,
        agent_type="q_learning",
        seed=seed,
        scenario=scenario,
        config=config,
        agent_hyperparameters=asdict(agent_config),
    )


def run_training_loop(
    *,
    env: GridNavigationEnv,
    agent: LearningAgent,
    agent_type: str,
    seed: int,
    scenario: DisruptionScenario,
    config: ExperimentConfig,
    agent_hyperparameters: dict[str, object],
) -> RunResult:
    """Train an agent while applying the configured route disruption once."""

    records: list[EpisodeRecord] = []
    immediate_post_disruption_evaluation: ImmediatePostDisruptionEvaluation | None = None
    disrupted = False
    for episode in range(1, config.episodes + 1):
        if not disrupted and episode == config.disruption_episode + 1:
            env.apply_route_closure(
                blocked_cells=scenario.disrupted_cells,
                blocked_edges=scenario.disrupted_edges,
            )
            disrupted = True
            immediate_post_disruption_evaluation = _run_immediate_post_disruption_evaluation(
                env=env.copy(),
                agent=agent,
                agent_type=agent_type,
                seed=seed,
                scenario_name=scenario.name,
                max_steps=config.max_steps_per_episode,
            )
        _run_training_episode(
            env=env,
            agent=agent,
            max_steps=config.max_steps_per_episode,
        )
        records.append(
            _run_evaluation_episode(
                env=env.copy(),
                agent=agent,
                agent_type=agent_type,
                seed=seed,
                scenario_name=scenario.name,
                episode=episode,
                disrupted=disrupted,
                max_steps=config.max_steps_per_episode,
            )
        )
        agent.end_episode()

    summary = summarize_records(
        records,
        disruption_episode=config.disruption_episode,
        recovery_window=config.recovery_window,
        recovery_success_rate=config.recovery_success_rate,
        recovery_efficiency=config.recovery_efficiency,
    )
    return RunResult(
        config={
            "experiment": asdict(config),
            "scenario": scenario.name,
            "seed": seed,
            "agent_type": agent_type,
            "agent_hyperparameters": agent_hyperparameters,
        },
        records=records,
        summary=asdict(summary),
        immediate_post_disruption_evaluation=immediate_post_disruption_evaluation,
    )


def _run_training_episode(
    *,
    env: GridNavigationEnv,
    agent: LearningAgent,
    max_steps: int,
) -> None:
    state = env.reset()
    for _ in range(max_steps):
        action = agent.select_action(state)
        result = env.step(action)
        agent.update(state, action, result.reward, result.state, result.terminated)
        state = result.state
        if result.terminated:
            break


def _run_evaluation_episode(
    *,
    env: GridNavigationEnv,
    agent: LearningAgent,
    agent_type: str,
    seed: int,
    scenario_name: str,
    episode: int,
    disrupted: bool,
    max_steps: int,
) -> EpisodeRecord:
    state = env.reset()
    total_return = 0.0
    steps = 0
    success = False
    for steps in range(1, max_steps + 1):
        action = agent.select_action(state, greedy=True)
        result = env.step(action)
        total_return += result.reward
        state = result.state
        if result.terminated:
            success = True
            break

    optimal = shortest_path(env)
    path_length = steps if success else None
    optimal_length = optimal.path_length
    efficiency = route_efficiency(success, path_length, optimal_length)
    regret = episode_regret(success, path_length, optimal_length, max_steps)
    return EpisodeRecord(
        episode=episode,
        seed=seed,
        agent_type=agent_type,
        scenario=scenario_name,
        disrupted=disrupted,
        success=success,
        episodic_return=total_return,
        steps=steps,
        path_length=path_length,
        optimal_path_length=optimal_length,
        route_efficiency=efficiency,
        regret=regret,
        exploration_rate=getattr(agent, "exploration_rate", None),
    )


def _run_immediate_post_disruption_evaluation(
    *,
    env: GridNavigationEnv,
    agent: LearningAgent,
    agent_type: str,
    seed: int,
    scenario_name: str,
    max_steps: int,
) -> ImmediatePostDisruptionEvaluation:
    evaluation = _run_evaluation_episode(
        env=env,
        agent=agent,
        agent_type=agent_type,
        seed=seed,
        scenario_name=scenario_name,
        episode=0,
        disrupted=True,
        max_steps=max_steps,
    )
    optimal = shortest_path(env)
    return ImmediatePostDisruptionEvaluation(
        seed=evaluation.seed,
        agent_type=evaluation.agent_type,
        scenario=evaluation.scenario,
        disrupted=evaluation.disrupted,
        post_disruption_training_episodes=0,
        success=evaluation.success,
        episodic_return=evaluation.episodic_return,
        steps=evaluation.steps,
        path_length=evaluation.path_length,
        optimal_path_length=evaluation.optimal_path_length,
        optimal_path_return=optimal.path_return,
        route_efficiency=evaluation.route_efficiency,
        regret=evaluation.regret,
        exploration_rate=evaluation.exploration_rate,
    )


def save_run_result(result: RunResult, output_dir: Path) -> None:
    """Save one run's configuration, episode records, and summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(result.config, handle, indent=2, sort_keys=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2, sort_keys=True)
    if result.immediate_post_disruption_evaluation is not None:
        with (output_dir / "immediate_post_disruption_evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(result.immediate_post_disruption_evaluation), handle, indent=2, sort_keys=True)
    with (output_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(result.records[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for record in result.records:
            writer.writerow(asdict(record))
