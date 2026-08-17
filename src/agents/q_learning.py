"""Tabular Q-learning agent for small discrete navigation environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.env.grid import Position


@dataclass(frozen=True)
class QLearningConfig:
    """Hyperparameters for tabular Q-learning."""

    learning_rate: float = 0.25
    discount_factor: float = 0.95
    exploration_rate: float = 0.35
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.05
    seed: int = 11


class TabularQLearningAgent:
    """Epsilon-greedy tabular Q-learning agent."""

    def __init__(self, states: list[Position], n_actions: int, config: QLearningConfig) -> None:
        self.states = list(states)
        self.n_actions = n_actions
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.q_table = np.zeros((len(self.states), n_actions), dtype=float)
        self.exploration_rate = config.exploration_rate

    def select_action(self, state: Position, *, greedy: bool = False) -> int:
        """Select an action using epsilon-greedy exploration unless greedy is true."""

        if state not in self.state_to_index:
            raise KeyError(f"Unknown state {state}")
        if not greedy and self.rng.random() < self.exploration_rate:
            return int(self.rng.integers(self.n_actions))
        values = self.q_table[self.state_to_index[state]]
        max_value = np.max(values)
        candidates = np.flatnonzero(np.isclose(values, max_value))
        if greedy:
            return int(candidates[0])
        return int(self.rng.choice(candidates))

    def update(
        self,
        state: Position,
        action: int,
        reward: float,
        next_state: Position,
        terminated: bool,
    ) -> None:
        """Apply the standard one-step Q-learning update."""

        state_index = self.state_to_index[state]
        next_index = self.state_to_index[next_state]
        current = self.q_table[state_index, action]
        bootstrap = 0.0 if terminated else np.max(self.q_table[next_index])
        target = reward + self.config.discount_factor * bootstrap
        self.q_table[state_index, action] = current + self.config.learning_rate * (target - current)

    def end_episode(self) -> None:
        """Decay exploration after one training episode."""

        self.exploration_rate = max(
            self.config.min_exploration_rate,
            self.exploration_rate * self.config.exploration_decay,
        )
