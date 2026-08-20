"""Run every model in a benchmark suite (YAML config) on the same dataset and
print + save a comparison table.

    python benchmarks/run_all.py --config benchmarks/configs/classification_suite.yaml --device auto
    python benchmarks/run_all.py --config benchmarks/configs/forecasting_suite.yaml
    python benchmarks/run_all.py --config benchmarks/configs/person_activity_suite.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquid_playground.benchmark import run_benchmark  # noqa: E402
from liquid_playground.device import add_device_arg, resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = []
    for model_name, model_kwargs in cfg["models"].items():
        print(f"\n=== {model_name} on {cfg['dataset']} ===")
        result = run_benchmark(
            model_name=model_name,
            dataset_name=cfg["dataset"],
            device=device,
            epochs=cfg.get("epochs", 20),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            seed=args.seed,
            model_kwargs=model_kwargs or {},
            dataset_kwargs=cfg.get("dataset_kwargs", {}),
        )
        print(result)
        results.append(result)

    header = ["model", "dataset", "task_type", "params", "epochs", "train_time_s", "metric_name", "metric", "device"]
    print("\n" + " | ".join(f"{h:>12}" for h in header))
    for r in results:
        print(" | ".join(f"{str(r[h]):>12}" for h in header))

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.config.stem}_{int(time.time())}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
