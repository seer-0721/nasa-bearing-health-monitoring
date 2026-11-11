from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from rc_esn.data import DATASET_CONFIGS, DatasetConfig, load_dataset
from rc_esn.esn import ESNConfig, ESNFeatureMap
from rc_esn.ridge import (
    RidgeClassifier,
    RidgeRegressor,
    accuracy,
    macro_accuracy,
    regression_accuracy,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RC/ESN readouts for bearing datasets.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-val-samples", type=int, default=10000)
    parser.add_argument("--reservoir-sizes", default="800")
    parser.add_argument("--spectral-radii", default="0.9")
    parser.add_argument("--leaking-rates", default="0.3")
    parser.add_argument("--input-scales", default="0.5")
    parser.add_argument("--alphas", default="1e-4,1e-3,1e-2,1e-1,1")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--no-include-raw", action="store_true")
    parser.add_argument("--primary-task", choices=["health", "fault"], default=None)
    return parser.parse_args()


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


def fit_regressor(mapped_train, mapped_val, y_train, y_val, alphas):
    best = None
    for alpha in alphas:
        model = RidgeRegressor(alpha=alpha).fit(mapped_train, y_train)
        pred = model.predict(mapped_val)
        pred = np.clip(pred, 0.0, 1.0)
        mse = float(np.mean((y_val - pred) ** 2))
        acc = regression_accuracy(y_val, pred, tolerance=0.1)
        metrics = {
            "alpha": alpha,
            "mse": mse,
            "regression_accuracy": acc,
        }
        if best is None or acc > best["metrics"]["regression_accuracy"]:
            best = {"model": model, "metrics": metrics}
    if best is None:
        raise RuntimeError("No regressor was trained.")
    return best


def main() -> None:
    args = parse_args()
    dataset_config: DatasetConfig = DATASET_CONFIGS[args.dataset]
    output_dir = Path(args.output_dir or f"results/rc_esn/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    primary_task = args.primary_task or ("health" if args.dataset == "nasa" else "fault")
    alphas = parse_float_list(args.alphas)
    best = None
    trials = []

    for seed in parse_int_list(args.seeds):
        for reservoir_size in parse_int_list(args.reservoir_sizes):
            for spectral_radius in parse_float_list(args.spectral_radii):
                for leaking_rate in parse_float_list(args.leaking_rates):
                    for input_scale in parse_float_list(args.input_scales):
                        esn_config = ESNConfig(
                            input_dim=args.input_dim,
                            reservoir_size=reservoir_size,
                            spectral_radius=spectral_radius,
                            leaking_rate=leaking_rate,
                            input_scale=input_scale,
                            density=args.density,
                            include_raw=not args.no_include_raw,
                            seed=seed,
                        )
                        feature_map = ESNFeatureMap(esn_config)
                        mapped_train = feature_map.transform(data["train_x"])
                        mapped_val = feature_map.transform(data["val_x"])
                        health = fit_regressor(
                            mapped_train,
                            mapped_val,
                            data["train_health"],
                            data["val_health"],
                            alphas,
                        )
                        fault = fit_classifier(
                            mapped_train,
                            mapped_val,
                            data["train_fault"],
                            data["val_fault"],
                            alphas,
                        )
                        trial = {
                            "config": esn_config.__dict__,
                            "max_train_samples": args.max_train_samples,
                            "max_val_samples": args.max_val_samples,
                            "health": health["metrics"],
                            "fault": fault["metrics"],
                        }
                        trials.append(trial)
                        score = (
                            trial["health"]["regression_accuracy"]
                            if primary_task == "health"
                            else trial["fault"]["macro_accuracy"]
                        )
                        print(
                            f"dataset={args.dataset} seed={seed} res={reservoir_size} "
                            f"rho={spectral_radius:g} leak={leaking_rate:g} scale={input_scale:g} "
                            f"healthA={trial['health']['regression_accuracy']:.4f} "
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
        raise RuntimeError("No ESN trial was trained.")

    summary = {
        "dataset": args.dataset,
        "data_root": dataset_config.data_root,
        "sample_layout": dataset_config.sample_layout,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "primary_task": primary_task,
        "best": best["trial"],
        "trials": trials,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "best.pkl").open("wb") as f:
        pickle.dump(
            {
                "dataset_config": dataset_config,
                "primary_task": primary_task,
                "feature_map": best["feature_map"],
                "health_model": best["health_model"],
                "fault_model": best["fault_model"],
                "metrics": best["trial"],
            },
            f,
        )
    print(json.dumps(summary["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
