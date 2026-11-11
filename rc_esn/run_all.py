from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all RC/ESN bearing experiments.")
    parser.add_argument("--datasets", default="nasa,cwru,paderborn")
    parser.add_argument("--fast", action="store_true", help="Use a smaller sweep for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    summary = []
    for dataset in datasets:
        output_dir = Path("results/rc_esn") / dataset
        command = [
            sys.executable,
            "-m",
            "rc_esn.train",
            "--dataset",
            dataset,
            "--output-dir",
            str(output_dir),
        ]
        if args.fast:
            command += [
                "--reservoir-sizes",
                "200",
                "--seeds",
                "42",
                "--alphas",
                "0.001,0.01",
                "--max-train-samples",
                "3000",
                "--max-val-samples",
                "1000",
            ]
        subprocess.run(command, check=True)
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        summary.append(
            {
                "dataset": dataset,
                "health_regression_accuracy": metrics["best"]["health"]["regression_accuracy"],
                "fault_macro_accuracy": metrics["best"]["fault"]["macro_accuracy"],
                "primary_task": metrics["primary_task"],
            }
        )
    Path("results/rc_esn").mkdir(parents=True, exist_ok=True)
    (Path("results/rc_esn") / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
