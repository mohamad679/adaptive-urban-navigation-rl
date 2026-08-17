"""Reference scenarios for adaptive navigation experiments."""

from __future__ import annotations

from dataclasses import dataclass

from src.env.grid import Edge, GridConfig, GridNavigationEnv, RewardConfig


@dataclass(frozen=True)
class DisruptionScenario:
    """Environment before and after a planned route disruption."""

    name: str
    base_config: GridConfig
    disrupted_edges: tuple[Edge, ...]
    disrupted_cells: tuple[tuple[int, int], ...] = ()

    def make_env(self) -> GridNavigationEnv:
        """Create a fresh environment for the pre-disruption topology."""

        return GridNavigationEnv(
            GridConfig(
                width=self.base_config.width,
                height=self.base_config.height,
                start=self.base_config.start,
                goal=self.base_config.goal,
                blocked_cells=set(self.base_config.blocked_cells),
                blocked_edges=set(self.base_config.blocked_edges),
                rewards=self.base_config.rewards,
                seed=self.base_config.seed,
            )
        )


def central_route_closure(seed: int | None = None) -> DisruptionScenario:
    """Return the default scenario with a central road segment closure."""

    return DisruptionScenario(
        name="central_route_closure",
        base_config=GridConfig(
            width=7,
            height=5,
            start=(0, 2),
            goal=(6, 2),
            rewards=RewardConfig(goal=20.0, movement=-1.0, invalid=-5.0),
            seed=seed,
        ),
        disrupted_edges=(((2, 2), (3, 2)),),
    )
