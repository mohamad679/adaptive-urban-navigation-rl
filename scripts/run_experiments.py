"""Run reproducible adaptive-navigation experiments and save outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aggregate import aggregate_experiment
from src.env.scenarios import DisruptionScenario, central_route_closure
from src.training.experiment import ExperimentConfig, run_q_learning_experiment, save_run_result

DEFAULT_CONFIG_PATH = Path("configs/default_experiment.json")
SUPPORTED_SCENARIOS = {"central_route_closure"}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved benchmark configuration loaded from JSON plus explicit CLI overrides."""

    seeds: tuple[int, ...]
    scenario: str
    experiment: ExperimentConfig


def load_benchmark_config(path: Path) -> dict[str, Any]:
    """Load benchmark configuration from a JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Benchmark configuration must be a JSON object")
    return payload


def resolve_benchmark_config(
    config_path: Path,
    *,
    episodes: int | None = None,
    disruption_episode: int | None = None,
    max_steps: int | None = None,
) -> BenchmarkConfig:
    """Resolve JSON defaults with explicit CLI-style overrides."""

    payload = load_benchmark_config(config_path)
    resolved = dict(payload)
    if episodes is not None:
        resolved["episodes"] = episodes
    if disruption_episode is not None:
        resolved["disruption_episode"] = disruption_episode
    if max_steps is not None:
        resolved["max_steps_per_episode"] = max_steps
    validate_benchmark_config(resolved)
    return BenchmarkConfig(
        seeds=tuple(int(seed) for seed in resolved["seeds"]),
        scenario=str(resolved["scenario"]),
        experiment=ExperimentConfig(
            episodes=int(resolved["episodes"]),
            disruption_episode=int(resolved["disruption_episode"]),
            max_steps_per_episode=int(resolved["max_steps_per_episode"]),
            recovery_window=int(resolved["recovery_window"]),
            recovery_success_rate=float(resolved["recovery_success_rate"]),
            recovery_efficiency=float(resolved["recovery_efficiency"]),
        ),
    )


def validate_benchmark_config(config: dict[str, Any]) -> None:
    """Validate benchmark configuration before experiments run."""

    required = {
        "seeds",
        "episodes",
        "disruption_episode",
        "max_steps_per_episode",
        "recovery_window",
        "recovery_success_rate",
        "recovery_efficiency",
        "scenario",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing benchmark configuration field(s): {', '.join(missing)}")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Benchmark configuration must include a non-empty seeds list")
    if any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("All seeds must be integers")
    episodes = int(config["episodes"])
    disruption_episode = int(config["disruption_episode"])
    max_steps_per_episode = int(config["max_steps_per_episode"])
    recovery_window = int(config["recovery_window"])
    recovery_success_rate = float(config["recovery_success_rate"])
    recovery_efficiency = float(config["recovery_efficiency"])
    scenario = str(config["scenario"])
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if disruption_episode < 0 or disruption_episode >= episodes:
        raise ValueError("disruption_episode must be non-negative and less than episodes")
    if max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive")
    if recovery_window <= 0:
        raise ValueError("recovery_window must be positive")
    if not 0.0 <= recovery_success_rate <= 1.0:
        raise ValueError("recovery_success_rate must be within [0, 1]")
    if not 0.0 <= recovery_efficiency <= 1.0:
        raise ValueError("recovery_efficiency must be within [0, 1]")
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(f"Unsupported scenario {scenario!r}; supported scenarios: central_route_closure")


def make_scenario(name: str, *, seed: int) -> DisruptionScenario:
    """Resolve a configured scenario name to the existing scenario factory."""

    if name == "central_route_closure":
        return central_route_closure(seed=seed)
    raise ValueError(f"Unsupported scenario {name!r}; supported scenarios: central_route_closure")


def selected_agent_types(*, include_dqn: bool) -> tuple[str, ...]:
    """Return agent types selected by CLI switches."""

    return ("q_learning", "dqn") if include_dqn else ("q_learning",)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--disruption-episode", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--include-dqn", action="store_true")
    args = parser.parse_args()

    benchmark = resolve_benchmark_config(
        args.config,
        episodes=args.episodes,
        disruption_episode=args.disruption_episode,
        max_steps=args.max_steps,
    )
    config = benchmark.experiment
    metrics_dir = args.output_dir / "metrics"
    run_dirs: list[Path] = []
    for seed in benchmark.seeds:
        scenario = make_scenario(benchmark.scenario, seed=seed)
        result = run_q_learning_experiment(seed, config, scenario)
        run_dir = metrics_dir / f"q_learning_seed_{seed}"
        save_run_result(result, run_dir)
        run_dirs.append(run_dir)

    if "dqn" in selected_agent_types(include_dqn=args.include_dqn):
        from src.training.dqn_experiment import run_dqn_experiment

        for seed in benchmark.seeds:
            scenario = make_scenario(benchmark.scenario, seed=seed)
            result = run_dqn_experiment(seed, config, scenario)
            run_dir = metrics_dir / f"dqn_seed_{seed}"
            save_run_result(result, run_dir)
            run_dirs.append(run_dir)

    aggregate = aggregate_experiment(
        metrics_dir,
        disruption_episode=config.disruption_episode,
        recovery_window=config.recovery_window,
        recovery_success_rate=config.recovery_success_rate,
        recovery_efficiency=config.recovery_efficiency,
        run_dirs=run_dirs,
    )
    from src.visualization.plots import generate_all_figures

    generate_all_figures(metrics_dir, args.output_dir / "figures", aggregate, run_dirs=run_dirs)


if __name__ == "__main__":
    main()
