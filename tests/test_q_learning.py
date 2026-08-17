"""Tests for tabular Q-learning behaviour."""

from __future__ import annotations

import numpy as np

from src.agents.q_learning import QLearningConfig, TabularQLearningAgent
from src.env.grid import GridConfig, GridNavigationEnv


def test_q_table_dimensions_match_states_and_actions() -> None:
    env = GridNavigationEnv(GridConfig(width=2, height=2, start=(0, 0), goal=(1, 1)))
    agent = TabularQLearningAgent(env.states, env.n_actions, QLearningConfig(seed=11))
    assert agent.q_table.shape == (4, 4)


def test_q_learning_update_changes_expected_entry() -> None:
    env = GridNavigationEnv(GridConfig(width=2, height=1, start=(0, 0), goal=(1, 0)))
    agent = TabularQLearningAgent(
        env.states,
        env.n_actions,
        QLearningConfig(learning_rate=0.5, discount_factor=0.9, seed=11),
    )
    agent.update((0, 0), 3, 20.0, (1, 0), True)
    assert np.isclose(agent.q_table[agent.state_to_index[(0, 0)], 3], 10.0)


def test_exploration_decay_respects_minimum() -> None:
    env = GridNavigationEnv(GridConfig(width=2, height=1, start=(0, 0), goal=(1, 0)))
    agent = TabularQLearningAgent(
        env.states,
        env.n_actions,
        QLearningConfig(exploration_rate=0.2, exploration_decay=0.1, min_exploration_rate=0.05),
    )
    agent.end_episode()
    agent.end_episode()
    assert agent.exploration_rate == 0.05
