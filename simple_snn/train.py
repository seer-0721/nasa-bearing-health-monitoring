from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from rc_esn.data import DATASET_CONFIGS, DatasetConfig, load_dataset
from rc_esn.ridge import (
    RidgeClassifier,
    accuracy,
    macro_accuracy,
    regression_accuracy,
)
from simple_snn.model import SimpleRateSNNFeatureMap, SimpleSNNConfig
from simple_snn.regression import BalancedRidgeRegressor


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def fit_classifier(mapped_train, mapped_val, y_train, y_val, alphas):
    best = None
    for alpha in alphas:
        model = RidgeClassifier(alpha=alpha).fit(mapped_train, y_train, balanced=True)
        pred = model.predict(mapped_val)
        metrics = {
            "alpha": alpha,
            "accuracy": accuracy(y_val, pred),
            "macro_accuracy": macro_accuracy(y_val, pred),
        }
        if best is None or metrics["macro_accuracy"] > best["metrics"]["macro_accuracy"]:
            best = {"model": model, "metrics": metrics}
    if best is None:
        raise RuntimeError("No classifier was trained.")
    return best


def fit_regressor(mapped_train, mapped_val, y_train, y_val, alphas, tolerance: float):
    best = None
    for alpha in alphas:
        model = BalancedRidgeRegressor(alpha=alpha).fit(mapped_train, y_train, balanced=True)
        pred = np.clip(model.predict(mapped_val), 0.0, 1.0)
        metrics = {
            "alpha": alpha,
            "mae": float(np.mean(np.abs(y_val - pred))),
            "mse": float(np.mean((y_val - pred) ** 2)),
            "accuracy_at_tolerance": regression_accuracy(y_val, pred, tolerance=tolerance),
            "tolerance": tolerance,
        }
        if best is None or metrics["accuracy_at_tolerance"] > best["metrics"]["accuracy_at_tolerance"]:
            best = {"model": model, "metrics": metrics}
    if best is None:
        raise RuntimeError("No regressor was trained.")
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train simplified SNN readouts.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-val-samples", type=int, default=10000)
    parser.add_argument("--hidden-sizes", default="256")
    parser.add_argument("--time-steps", default="6")
    parser.add_argument("--betas", default="0.85")
    parser.add_argument("--thresholds", default="0.65")
    parser.add_argument("--input-scales", default="1.0")
    parser.add_argument("--recurrent-scales", default="0.9")
    parser.add_argument("--alphas", default="1e-3,1e-2,1e-1")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--health-tolerance", type=float, default=0.1)
    parser.add_argument("--no-include-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config: DatasetConfig = DATASET_CONFIGS[args.dataset]
    output_dir = Path(args.output_dir or f"results/simple_snn/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    alphas = parse_float_list(args.alphas)
    best = None
    trials = []

    for seed in parse_int_list(args.seeds):
        for hidden_size in parse_int_list(args.hidden_sizes):
            for time_steps in parse_int_list(args.time_steps):
                for beta in parse_float_list(args.betas):
                    for threshold in parse_float_list(args.thresholds):
                        for input_scale in parse_float_list(args.input_scales):
                            for recurrent_scale in parse_float_list(args.recurrent_scales):
                                config = SimpleSNNConfig(
                                    input_size=data["input_size"],
                                    hidden_size=hidden_size,
                                    time_steps=time_steps,
                                    beta=beta,
                                    threshold=threshold,
                                    input_scale=input_scale,
                                    recurrent_scale=recurrent_scale,
                                    include_raw=not args.no_include_raw,
                                    seed=seed,
                                )
                                feature_map = SimpleRateSNNFeatureMap(config)
                                mapped_train = feature_map.transform(data["train_x"])
                                mapped_val = feature_map.transform(data["val_x"])
                                health = fit_regressor(
                                    mapped_train,
                                    mapped_val,
                                    data["train_health"],
                                    data["val_health"],
                                    alphas,
                                    args.health_tolerance,
                                )
                                fault = fit_classifier(
                                    mapped_train,
                                    mapped_val,
                                    data["train_fault"],
                                    data["val_fault"],
                                    alphas,
                                )
                                trial = {
                                    "config": config.__dict__,
                                    "max_train_samples": args.max_train_samples,
                                    "max_val_samples": args.max_val_samples,
                                    "health": health["metrics"],
                                    "fault": fault["metrics"],
                                }
                                trials.append(trial)
                                score = min(
                                    trial["health"]["accuracy_at_tolerance"],
                                    trial["fault"]["macro_accuracy"],
                                )
                                print(
                                    f"dataset={args.dataset} seed={seed} hidden={hidden_size} "
                                    f"T={time_steps} beta={beta:g} th={threshold:g} scale={input_scale:g} "
                                    f"rec={recurrent_scale:g} h_mae={trial['health']['mae']:.4f} "
                                    f"h_acc@{args.health_tolerance:g}={trial['health']['accuracy_at_tolerance']:.4f} "
                                    f"faultM={trial['fault']['macro_accuracy']:.4f}",
                                    flush=True,
                                )
                                if best is None or score > best["score"]:
                                    best = {
                                        "score": score,
                                        "feature_map": feature_map,
                                        "health_model": health["model"],
                                        "fault_model": fault["model"],
                                        "trial": trial,
                                    }

    if best is None:
        raise RuntimeError("No simplified SNN trial was trained.")

    summary = {
        "dataset": args.dataset,
        "data_root": dataset_config.data_root,
        "sample_layout": dataset_config.sample_layout,
        "health_task": "continuous_score_regression",
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "best": best["trial"],
        "trials": trials,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "best.pkl").open("wb") as f:
        pickle.dump(
            {
                "dataset_config": dataset_config,
                "feature_map": best["feature_map"],
                "health_model": best["health_model"],
                "fault_model": best["fault_model"],
                "metrics": best["trial"],
                "health_task": summary["health_task"],
            },
            f,
        )
    print(json.dumps(summary["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
