"""Deterministic grid navigation environment for route-disruption experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, NamedTuple

Position = tuple[int, int]
Edge = tuple[Position, Position]


class StepResult(NamedTuple):
    """Result returned by one environment transition."""

    state: Position
    reward: float
    terminated: bool
    info: dict[str, object]


@dataclass(frozen=True)
class RewardConfig:
    """Configurable reward values for grid navigation."""

    goal: float = 20.0
    movement: float = -1.0
    invalid: float = -5.0


@dataclass
class GridConfig:
    """Configuration for a deterministic grid navigation environment."""

    width: int
    height: int
    start: Position
    goal: Position
    blocked_cells: set[Position] = field(default_factory=set)
    blocked_edges: set[Edge] = field(default_factory=set)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    seed: int | None = None


class GridNavigationEnv:
    """Small deterministic grid world with blocked cells and route segments."""

    ACTIONS: tuple[str, ...] = ("up", "down", "left", "right")
    ACTION_DELTAS: dict[int, Position] = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
    }

    def __init__(self, config: GridConfig) -> None:
        self.config = config
        self._validate_config()
        self.state: Position = config.start
        self.terminated = False

    @property
    def n_actions(self) -> int:
        """Number of available discrete actions."""

        return len(self.ACTIONS)

    @property
    def states(self) -> list[Position]:
        """All traversable grid states."""

        return [
            (x, y)
            for y in range(self.config.height)
            for x in range(self.config.width)
            if (x, y) not in self.config.blocked_cells
        ]

    def reset(self) -> Position:
        """Return the environment to the configured start state."""

        self.state = self.config.start
        self.terminated = self.state == self.config.goal
        return self.state

    def step(self, action: int) -> StepResult:
        """Apply an action and return the deterministic transition result."""

        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Unknown action {action!r}; expected 0..{self.n_actions - 1}")
        if self.terminated:
            return StepResult(self.state, 0.0, True, {"already_terminated": True})

        dx, dy = self.ACTION_DELTAS[action]
        candidate = (self.state[0] + dx, self.state[1] + dy)
        invalid_reason = self._invalid_move_reason(self.state, candidate)
        if invalid_reason is not None:
            return StepResult(
                self.state,
                self.config.rewards.invalid,
                False,
                {"invalid": True, "reason": invalid_reason},
            )

        self.state = candidate
        self.terminated = self.state == self.config.goal
        reward = self.config.rewards.goal if self.terminated else self.config.rewards.movement
        return StepResult(self.state, reward, self.terminated, {"invalid": False})

    def neighbors(self, state: Position) -> list[tuple[Position, int]]:
        """Return valid neighboring states and the actions that reach them."""

        if not self.is_traversable(state):
            return []
        neighbors: list[tuple[Position, int]] = []
        for action, (dx, dy) in self.ACTION_DELTAS.items():
            candidate = (state[0] + dx, state[1] + dy)
            if self._invalid_move_reason(state, candidate) is None:
                neighbors.append((candidate, action))
        return neighbors

    def apply_route_closure(
        self,
        *,
        blocked_cells: Iterable[Position] = (),
        blocked_edges: Iterable[Edge] = (),
    ) -> None:
        """Modify the topology by adding blocked cells and/or route segments."""

        self.config.blocked_cells.update(blocked_cells)
        self.config.blocked_edges.update(normalize_edge(edge) for edge in blocked_edges)
        self._validate_config()
        if self.state in self.config.blocked_cells:
            self.reset()

    def is_traversable(self, state: Position) -> bool:
        """Return whether a position is inside the grid and not blocked."""

        x, y = state
        return (
            0 <= x < self.config.width
            and 0 <= y < self.config.height
            and state not in self.config.blocked_cells
        )

    def copy(self) -> "GridNavigationEnv":
        """Create an independent environment with the same topology."""

        return GridNavigationEnv(
            GridConfig(
                width=self.config.width,
                height=self.config.height,
                start=self.config.start,
                goal=self.config.goal,
                blocked_cells=set(self.config.blocked_cells),
                blocked_edges=set(self.config.blocked_edges),
                rewards=self.config.rewards,
                seed=self.config.seed,
            )
        )

    def _invalid_move_reason(self, source: Position, target: Position) -> str | None:
        if not self.is_traversable(target):
            return "blocked_or_out_of_bounds"
        if normalize_edge((source, target)) in self.config.blocked_edges:
            return "blocked_edge"
        return None

    def _validate_config(self) -> None:
        if self.config.width <= 0 or self.config.height <= 0:
            raise ValueError("Grid width and height must be positive")
        if not self.is_traversable(self.config.start):
            raise ValueError("Start position must be inside the grid and traversable")
        if not self.is_traversable(self.config.goal):
            raise ValueError("Goal position must be inside the grid and traversable")
        self.config.blocked_edges = {normalize_edge(edge) for edge in self.config.blocked_edges}
        for source, target in self.config.blocked_edges:
            if not self.is_traversable(source) or not self.is_traversable(target):
                raise ValueError("Blocked edges must connect traversable in-grid cells")
            distance = abs(source[0] - target[0]) + abs(source[1] - target[1])
            if distance != 1:
                raise ValueError("Blocked edges must connect adjacent cells")


def normalize_edge(edge: Edge) -> Edge:
    """Normalize an undirected edge so route closures work both ways."""

    source, target = edge
    return tuple(sorted((source, target)))  # type: ignore[return-value]
