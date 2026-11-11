from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rc_esn.data import load_dataset
from rc_esn.ridge import accuracy, macro_accuracy


DATASETS = ("nasa", "cwru", "paderborn")


def load_predictions(model_dir: Path, dataset: str):
    saved = pickle.load((model_dir / dataset / "best.pkl").open("rb"))
    data = load_dataset(saved["dataset_config"], 512, 0, 42, max_val_samples=10000)
    mapped = saved["feature_map"].transform(data["val_x"])
    if model_dir.name == "stdp_snn":
        from stdp_snn.train import augment_readout_features

        mapped = augment_readout_features(mapped, data["val_x"])
    health_pred = np.clip(saved["health_model"].predict(mapped), 0.0, 1.0)
    fault_pred = saved["fault_model"].predict(mapped)
    return {
        "health_true": data["val_health"],
        "health_pred": health_pred,
        "fault_true": data["val_fault"],
        "fault_pred": fault_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style strict evaluation metrics.")
    parser.add_argument("--models", default="stdp_snn,simple_snn")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="docs/paper_assets/paper_eval_metrics.csv")
    parser.add_argument("--tolerances", default="0.03,0.035,0.04,0.045,0.05,0.06,0.075,0.1")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tolerances = [float(item) for item in args.tolerances.split(",") if item.strip()]
    rows = []
    for model in [item.strip() for item in args.models.split(",") if item.strip()]:
        model_dir = Path(args.results_root) / model
        for dataset in DATASETS:
            pred = load_predictions(model_dir, dataset)
            health_true = pred["health_true"]
            health_pred = pred["health_pred"]
            fault_true = pred["fault_true"]
            fault_pred = pred["fault_pred"]
            base = {
                "model": model,
                "dataset": dataset,
                "n": len(health_true),
                "health_mae": float(np.mean(np.abs(health_true - health_pred))),
                "health_mse": float(np.mean((health_true - health_pred) ** 2)),
                "fault_accuracy": accuracy(fault_true, fault_pred),
                "fault_macro_accuracy": macro_accuracy(fault_true, fault_pred),
            }
            for tolerance in tolerances:
                health_ok = np.abs(health_true - health_pred) <= tolerance
                joint_ok = health_ok & (fault_true == fault_pred)
                rows.append(
                    {
                        **base,
                        "tolerance": tolerance,
                        "health_accuracy_at_tolerance": float(np.mean(health_ok)),
                        "joint_accuracy_at_tolerance": float(np.mean(joint_ok)),
                    }
                )

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
