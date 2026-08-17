"""Tests for the DQN baseline."""

from __future__ import annotations

import torch

from src.agents.dqn import DQNAgent, DQNConfig, QNetwork


def test_dqn_output_dimensions() -> None:
    network = QNetwork(input_dim=2, n_actions=4)
    output = network(torch.zeros((3, 2), dtype=torch.float32))
    assert output.shape == (3, 4)


def test_dqn_agent_selects_valid_action() -> None:
    agent = DQNAgent(width=7, height=5, n_actions=4, config=DQNConfig(seed=11, exploration_rate=0.0))
    action = agent.select_action((0, 2), greedy=True)
    assert 0 <= action < 4


def test_dqn_update_changes_policy_and_updates_target_network() -> None:
    agent = DQNAgent(
        width=2,
        height=1,
        n_actions=4,
        config=DQNConfig(seed=11, batch_size=2, target_update_interval=1),
    )
    before = [parameter.detach().clone() for parameter in agent.policy_network.parameters()]
    agent.update((0, 0), 3, 1.0, (1, 0), False)
    agent.update((0, 0), 3, 1.0, (1, 0), True)
    after = list(agent.policy_network.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after))
    for policy_param, target_param in zip(agent.policy_network.parameters(), agent.target_network.parameters()):
        assert torch.equal(policy_param, target_param)
