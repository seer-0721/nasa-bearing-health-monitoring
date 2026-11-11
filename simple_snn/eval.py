from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from rc_esn.data import load_dataset
from rc_esn.ridge import accuracy, macro_accuracy, regression_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a simplified SNN model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.model).open("rb") as f:
        saved = pickle.load(f)
    config = saved["dataset_config"]
    saved_metrics = saved.get("metrics", {})
    max_val_samples = args.max_val_samples
    if max_val_samples <= 0:
        max_val_samples = int(saved_metrics.get("max_val_samples", 10000))
    data = load_dataset(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        max_val_samples=max_val_samples,
    )
    mapped_val = saved["feature_map"].transform(data["val_x"])
    health_pred = np.clip(saved["health_model"].predict(mapped_val), 0.0, 1.0)
    fault_pred = saved["fault_model"].predict(mapped_val)
    tolerance = float(saved_metrics.get("health", {}).get("tolerance", 0.1))
    metrics = {
        "dataset": config.name,
        "health_mae": float(np.mean(np.abs(data["val_health"] - health_pred))),
        "health_mse": float(np.mean((data["val_health"] - health_pred) ** 2)),
        "health_accuracy_at_tolerance": regression_accuracy(
            data["val_health"],
            health_pred,
            tolerance,
        ),
        "health_tolerance": tolerance,
        "fault_accuracy": accuracy(data["val_fault"], fault_pred),
        "fault_macro_accuracy": macro_accuracy(data["val_fault"], fault_pred),
        "max_val_samples": max_val_samples,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
