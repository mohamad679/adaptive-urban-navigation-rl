"""DQN experiment entry points."""

from __future__ import annotations

from dataclasses import asdict

from src.agents.dqn import DQNAgent, DQNConfig
from src.env.scenarios import DisruptionScenario, central_route_closure
from src.training.experiment import ExperimentConfig, RunResult, run_training_loop


def run_dqn_experiment(seed: int, config: ExperimentConfig, scenario: DisruptionScenario | None = None) -> RunResult:
    """Run one seeded DQN route-disruption experiment."""

    scenario = scenario or central_route_closure(seed=seed)
    env = scenario.make_env()
    agent_config = DQNConfig(seed=seed)
    agent = DQNAgent(
        width=env.config.width,
        height=env.config.height,
        n_actions=env.n_actions,
        config=agent_config,
    )
    return run_training_loop(
        env=env,
        agent=agent,
        agent_type="dqn",
        seed=seed,
        scenario=scenario,
        config=config,
        agent_hyperparameters=asdict(agent_config),
    )
