"""Small Deep Q-Network baseline for the grid navigation benchmark."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn

from src.env.grid import Position

torch.set_num_threads(1)


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for a compact DQN baseline."""

    learning_rate: float = 1e-3
    discount_factor: float = 0.95
    exploration_rate: float = 0.4
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.05
    replay_capacity: int = 5000
    batch_size: int = 32
    target_update_interval: int = 50
    seed: int = 11


class QNetwork(nn.Module):
    """Two-hidden-layer Q-network with one output per action."""

    def __init__(self, input_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action."""

        return self.net(inputs)


class DQNAgent:
    """DQN agent with replay memory and a periodically updated target network."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        n_actions: int,
        config: DQNConfig,
    ) -> None:
        self.width = width
        self.height = height
        self.n_actions = n_actions
        self.config = config
        self.exploration_rate = config.exploration_rate
        self.rng = np.random.default_rng(config.seed)
        self.py_rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.policy_network = QNetwork(2, n_actions)
        self.target_network = QNetwork(2, n_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        self.loss = nn.SmoothL1Loss()
        self.replay: deque[tuple[Position, int, float, Position, bool]] = deque(maxlen=config.replay_capacity)
        self.update_count = 0

    def select_action(self, state: Position, *, greedy: bool = False) -> int:
        """Select an action using epsilon-greedy exploration unless greedy is true."""

        if not greedy and self.rng.random() < self.exploration_rate:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q_values = self.policy_network(self._encode_state(state).unsqueeze(0))[0]
        max_value = torch.max(q_values)
        candidates = torch.nonzero(torch.isclose(q_values, max_value), as_tuple=False).flatten().numpy()
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
        """Store a transition and run one DQN optimization step if possible."""

        self.replay.append((state, action, reward, next_state, terminated))
        if len(self.replay) < self.config.batch_size:
            return

        batch = self.py_rng.sample(list(self.replay), self.config.batch_size)
        states = torch.stack([self._encode_state(item[0]) for item in batch])
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        next_states = torch.stack([self._encode_state(item[3]) for item in batch])
        terminated = torch.tensor([item[4] for item in batch], dtype=torch.float32)

        current_q = self.policy_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1).values
            target_q = rewards + self.config.discount_factor * next_q * (1.0 - terminated)

        loss = self.loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def end_episode(self) -> None:
        """Decay exploration after one episode."""

        self.exploration_rate = max(
            self.config.min_exploration_rate,
            self.exploration_rate * self.config.exploration_decay,
        )

    def _encode_state(self, state: Position) -> torch.Tensor:
        x, y = state
        return torch.tensor(
            [
                x / max(self.width - 1, 1),
                y / max(self.height - 1, 1),
            ],
            dtype=torch.float32,
        )
