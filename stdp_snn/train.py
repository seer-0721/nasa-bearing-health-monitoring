from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from rc_esn.data import DATASET_CONFIGS, DatasetConfig, load_dataset
from rc_esn.ridge import RidgeClassifier, accuracy, macro_accuracy, regression_accuracy
from simple_snn.regression import BalancedRidgeRegressor
from stdp_snn.model import STDPSNNConfig, STDPSNNFeatureMap
from stdp_snn.regression import DiscreteHealthScoreRegressor
from utils import load_stdp_params


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
    use_discrete_head = np.unique(np.round(y_train, 6)).size <= 16
    for alpha in alphas:
        if use_discrete_head:
            model = DiscreteHealthScoreRegressor(alpha=alpha).fit(mapped_train, y_train)
            head = "discrete_linear_score"
        else:
            model = BalancedRidgeRegressor(alpha=alpha).fit(mapped_train, y_train, balanced=True)
            head = "balanced_ridge_regression"
        pred = np.clip(model.predict(mapped_val), 0.0, 1.0)
        metrics = {
            "alpha": alpha,
            "head": head,
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


def augment_readout_features(mapped: np.ndarray, raw: np.ndarray) -> np.ndarray:
    raw = np.clip(raw.astype(np.float32, copy=False), 0.0, 1.0)
    stats = np.stack(
        [
            raw.mean(axis=1),
            raw.std(axis=1),
            raw.min(axis=1),
            raw.max(axis=1),
            np.percentile(raw, 25, axis=1),
            np.percentile(raw, 75, axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([mapped, raw * raw, np.sqrt(raw), stats], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train simplified STDP-SNN readouts.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stdp-params-file", default="stdp_params.npz")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-val-samples", type=int, default=10000)
    parser.add_argument("--hidden-sizes", default="768")
    parser.add_argument("--time-steps", default="8")
    parser.add_argument("--betas", default="0.85")
    parser.add_argument("--thresholds", default="0.65")
    parser.add_argument("--input-scales", default="1.0")
    parser.add_argument("--stdp-lrs", default="1e-7")
    parser.add_argument("--stdp-ltd-scales", default="0.2")
    parser.add_argument("--stdp-epochs", type=int, default=1)
    parser.add_argument("--alphas", default="1e-4,1e-3,1e-2,1e-1,1,10")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--health-tolerance", type=float, default=0.1)
    parser.add_argument("--no-include-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config: DatasetConfig = DATASET_CONFIGS[args.dataset]
    output_dir = Path(args.output_dir or f"results/stdp_snn/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    stdp_params = load_stdp_params(args.stdp_params_file)
    alphas = parse_float_list(args.alphas)
    best = None
    trials = []

    for seed in parse_int_list(args.seeds):
        for hidden_size in parse_int_list(args.hidden_sizes):
            for time_steps in parse_int_list(args.time_steps):
                for beta in parse_float_list(args.betas):
                    for threshold in parse_float_list(args.thresholds):
                        for input_scale in parse_float_list(args.input_scales):
                            for stdp_lr in parse_float_list(args.stdp_lrs):
                                for ltd_scale in parse_float_list(args.stdp_ltd_scales):
                                    config = STDPSNNConfig(
                                        input_size=data["input_size"],
                                        hidden_size=hidden_size,
                                        time_steps=time_steps,
                                        beta=beta,
                                        threshold=threshold,
                                        input_scale=input_scale,
                                        stdp_lr=stdp_lr,
                                        stdp_ltd_scale=ltd_scale,
                                        include_raw=not args.no_include_raw,
                                        seed=seed,
                                    )
                                    feature_map = STDPSNNFeatureMap(config, stdp_params)
                                    feature_map.pretrain(
                                        data["train_x"],
                                        epochs=args.stdp_epochs,
                                        batch_size=args.batch_size,
                                    )
                                    mapped_train = augment_readout_features(
                                        feature_map.transform(data["train_x"]),
                                        data["train_x"],
                                    )
                                    mapped_val = augment_readout_features(
                                        feature_map.transform(data["val_x"]),
                                        data["val_x"],
                                    )
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
                                        "stdp_epochs": args.stdp_epochs,
                                        "stdp_params": stdp_params,
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
                                        f"T={time_steps} beta={beta:g} th={threshold:g} "
                                        f"stdp_lr={stdp_lr:g} ltd={ltd_scale:g} "
                                        f"h_mae={trial['health']['mae']:.4f} "
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
        raise RuntimeError("No STDP-SNN trial was trained.")

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
