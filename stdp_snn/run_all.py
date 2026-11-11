from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all simplified STDP-SNN experiments.")
    parser.add_argument("--datasets", default="nasa,cwru,paderborn")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    summary = []
    for dataset in datasets:
        output_dir = Path("results/stdp_snn") / dataset
        command = [
            sys.executable,
            "-m",
            "stdp_snn.train",
            "--dataset",
            dataset,
            "--output-dir",
            str(output_dir),
        ]
        if args.fast:
            command += [
                "--hidden-sizes",
                "256",
                "--time-steps",
                "6",
                "--stdp-epochs",
                "1",
                "--max-train-samples",
                "8000",
                "--max-val-samples",
                "2000",
                "--alphas",
                "0.001,0.01,0.1",
            ]
        subprocess.run(command, check=True)
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        summary.append(
            {
                "dataset": dataset,
                "health_accuracy_at_tolerance": metrics["best"]["health"]["accuracy_at_tolerance"],
                "health_mae": metrics["best"]["health"]["mae"],
                "fault_macro_accuracy": metrics["best"]["fault"]["macro_accuracy"],
                "target_met": (
                    metrics["best"]["health"]["accuracy_at_tolerance"] >= 0.95
                    and metrics["best"]["fault"]["macro_accuracy"] >= 0.95
                ),
            }
        )
    Path("results/stdp_snn").mkdir(parents=True, exist_ok=True)
    (Path("results/stdp_snn") / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

