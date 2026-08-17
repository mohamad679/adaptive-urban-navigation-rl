"""Tests for the route-disruption training pipeline."""

from __future__ import annotations

from src.env.scenarios import DisruptionScenario, central_route_closure
from src.env.grid import GridConfig
from src.training.experiment import ExperimentConfig, run_q_learning_experiment
from src.training.experiment import run_training_loop


def test_q_learning_pipeline_records_disruption_and_metrics() -> None:
    result = run_q_learning_experiment(
        seed=11,
        config=ExperimentConfig(
            episodes=20,
            disruption_episode=10,
            max_steps_per_episode=25,
            recovery_window=5,
        ),
    )
    assert len(result.records) == 20
    assert not result.records[9].disrupted
    assert result.records[10].disrupted
    assert result.records[10].optimal_path_length == 8
    assert "success_rate" in result.summary
    assert result.config["agent_hyperparameters"]["seed"] == 11
    assert result.immediate_post_disruption_evaluation is not None
    assert result.immediate_post_disruption_evaluation.post_disruption_training_episodes == 0
    assert result.immediate_post_disruption_evaluation.optimal_path_length == 8
    assert result.immediate_post_disruption_evaluation.optimal_path_return == 13.0


class CountingAgent:
    exploration_rate = 0.0

    def __init__(self) -> None:
        self.update_count = 0
        self.training_actions = 0
        self.greedy_actions = 0
        self.greedy_update_counts: list[int] = []

    def select_action(self, state: tuple[int, int], *, greedy: bool = False) -> int:
        if greedy:
            self.greedy_actions += 1
            self.greedy_update_counts.append(self.update_count)
        else:
            self.training_actions += 1
        return 3

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        terminated: bool,
    ) -> None:
        self.update_count += 1

    def end_episode(self) -> None:
        pass


def test_training_loop_records_greedy_no_update_evaluations() -> None:
    scenario = DisruptionScenario(
        name="simple_closure",
        base_config=GridConfig(width=3, height=1, start=(0, 0), goal=(2, 0)),
        disrupted_edges=(((0, 0), (1, 0)),),
    )
    agent = CountingAgent()
    result = run_training_loop(
        env=scenario.make_env(),
        agent=agent,
        agent_type="counting",
        seed=11,
        scenario=scenario,
        config=ExperimentConfig(episodes=2, disruption_episode=1, max_steps_per_episode=1, recovery_window=1),
        agent_hyperparameters={"seed": 11},
    )
    assert agent.training_actions > 0
    assert agent.greedy_actions > 0
    assert agent.update_count == agent.training_actions
    assert result.immediate_post_disruption_evaluation is not None
    assert not result.immediate_post_disruption_evaluation.success
    assert result.immediate_post_disruption_evaluation.optimal_path_length is None
    assert agent.greedy_update_counts[1] == 1
    assert agent.greedy_update_counts[-1] == 2
