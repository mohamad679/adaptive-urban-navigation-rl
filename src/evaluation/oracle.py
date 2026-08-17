"""Shortest-path oracle used for evaluation, not as a learned policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from src.env.grid import GridNavigationEnv, Position


@dataclass(frozen=True)
class ShortestPathResult:
    """Result from deterministic shortest-path search."""

    reachable: bool
    path: tuple[Position, ...]
    actions: tuple[int, ...]
    path_length: int | None
    path_return: float | None


def shortest_path(env: GridNavigationEnv) -> ShortestPathResult:
    """Find an optimal path in an unweighted deterministic grid using BFS."""

    start = env.config.start
    goal = env.config.goal
    frontier: deque[Position] = deque([start])
    parent: dict[Position, tuple[Position, int] | None] = {start: None}

    while frontier:
        state = frontier.popleft()
        if state == goal:
            break
        for next_state, action in env.neighbors(state):
            if next_state not in parent:
                parent[next_state] = (state, action)
                frontier.append(next_state)

    if goal not in parent:
        return ShortestPathResult(False, (), (), None, None)

    path: list[Position] = []
    actions: list[int] = []
    current = goal
    while current != start:
        path.append(current)
        previous = parent[current]
        assert previous is not None
        current, action = previous
        actions.append(action)
    path.append(start)
    path.reverse()
    actions.reverse()
    path_length = len(path) - 1
    path_return = 0.0 if path_length == 0 else (path_length - 1) * env.config.rewards.movement + env.config.rewards.goal
    return ShortestPathResult(True, tuple(path), tuple(actions), path_length, path_return)
