"""Run reproducible adaptive-navigation experiments and save outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aggregate import aggregate_experiment
from src.training.experiment import ExperimentConfig, run_q_learning_experiment, save_run_result
from src.visualization.plots import generate_all_figures

REQUIRED_SEEDS = (11, 22, 33, 44, 55)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--disruption-episode", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--include-dqn", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig(
        episodes=args.episodes,
        disruption_episode=args.disruption_episode,
        max_steps_per_episode=args.max_steps,
    )
    metrics_dir = args.output_dir / "metrics"
    run_dirs: list[Path] = []
    for seed in REQUIRED_SEEDS:
        result = run_q_learning_experiment(seed, config)
        run_dir = metrics_dir / f"q_learning_seed_{seed}"
        save_run_result(result, run_dir)
        run_dirs.append(run_dir)

    if args.include_dqn:
        from src.training.dqn_experiment import run_dqn_experiment

        for seed in REQUIRED_SEEDS:
            result = run_dqn_experiment(seed, config)
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
    generate_all_figures(metrics_dir, args.output_dir / "figures", aggregate, run_dirs=run_dirs)


if __name__ == "__main__":
    main()
