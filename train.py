from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import DEFAULT_FAULT_MAP, NASA_BEARING_FAULT_PARTS, build_nasa_bearing_loaders
from model import NasaBearingSNN
from utils import load_stdp_params, resolve_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NASA Bearing STDP/SNN model.")
    parser.add_argument("--data-root", type=str, default="data/NasaBearingDataset")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/nasa_bearing_snn")
    parser.add_argument("--stdp-params-file", type=str, default="stdp_params.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--time-steps", type=int, default=16)
    parser.add_argument("--health-bins", type=int, default=11)
    parser.add_argument(
        "--health-task",
        type=str,
        default="regression",
        choices=["regression", "classification", "hybrid"],
        help="Deprecated compatibility flag; the current model only outputs continuous health_score.",
    )
    parser.add_argument(
        "--health-label-mode",
        type=str,
        default="time",
        choices=["time", "source", "anomaly"],
    )
    parser.add_argument(
        "--fault-map",
        type=str,
        default=DEFAULT_FAULT_MAP,
        help="Comma-separated component labels, for example B3=inner_ring,B4=rolling_element.",
    )
    parser.add_argument("--health-loss-weight", type=float, default=0.0)
    parser.add_argument("--health-regression-weight", type=float, default=2.0)
    parser.add_argument("--health-regression-low-score-weight", type=float, default=1.0)
    parser.add_argument(
        "--health-regression-high-score-weight",
        type=float,
        default=0.0,
        help="Extra regression weight for samples near health_score=1.0.",
    )
    parser.add_argument(
        "--health-regression-loss",
        type=str,
        default="l1",
        choices=["l1", "smooth_l1", "mse"],
        help="Loss for continuous health-score prediction.",
    )
    parser.add_argument(
        "--health-score-tolerance",
        type=float,
        default=0.1,
        help="Tolerance for reporting continuous health-score accuracy.",
    )
    parser.add_argument("--fault-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--health-decision",
        type=str,
        default="regression",
        choices=["regression", "classification"],
    )
    parser.add_argument("--class-weights", type=str, default="balanced", choices=["none", "balanced"])
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sampler-balance",
        type=str,
        default="auto",
        choices=["auto", "both", "health", "fault", "none"],
        help="auto uses both health-stage and fault-class balancing for regression.",
    )
    parser.add_argument("--include-bearing-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sample-layout",
        type=str,
        default="per_bearing",
        choices=["per_bearing", "row"],
        help="per_bearing splits B1-B4 into separate samples; row keeps all feature columns together.",
    )
    parser.add_argument("--hidden-size1", type=int, default=256)
    parser.add_argument("--hidden-size2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--readout", type=str, default="hybrid", choices=["spike", "membrane", "hybrid"])
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--stdp-lr", type=float, default=1e-6)
    parser.add_argument("--stdp-ltd-scale", type=float, default=1.0)
    parser.add_argument("--stdp-pretrain", action="store_true")
    parser.add_argument("--stdp-epochs", type=int, default=1)
    parser.add_argument("--stdp-during-supervised", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--limit-good", type=int, default=20000, help="Use 0 to load all good rows.")
    parser.add_argument("--limit-fault", type=int, default=None, help="Use 0 to load all fault rows.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-format",
        type=str,
        default="pretty",
        choices=["pretty", "legacy"],
        help="pretty prints compact tables; legacy prints one long key=value line per epoch.",
    )
    parser.add_argument(
        "--log-header-interval",
        type=int,
        default=20,
        help="Repeat the pretty metrics header every N epochs. Use 0 to print it once.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="composite",
        choices=[
            "composite",
            "val_fault_macro_acc",
            "val_health_score_acc_at_tolerance",
            "val_health_score_fault_acc_at_tolerance",
            "val_health_score_balanced_acc_at_tolerance",
            "neg_val_health_score_mae",
            "neg_val_health_score_balanced_mae",
        ],
        help="Metric used to choose best.pt.",
    )
    return parser.parse_args()


def make_model(args: argparse.Namespace, input_size: int, device: torch.device) -> NasaBearingSNN:
    return NasaBearingSNN(
        input_size=input_size,
        hidden_size1=args.hidden_size1,
        hidden_size2=args.hidden_size2,
        fault_classes=len(NASA_BEARING_FAULT_PARTS),
        time_steps=args.time_steps,
        beta=args.beta,
        threshold=args.threshold,
        stdp_params=load_stdp_params(args.stdp_params_file),
        stdp_lr=args.stdp_lr,
        stdp_ltd_scale=args.stdp_ltd_scale,
        readout=args.readout,
        dropout=args.dropout,
    ).to(device)


@torch.no_grad()
def stdp_pretrain_epoch(
    model: NasaBearingSNN,
    loader: DataLoader,
    device: torch.device,
) -> None:
    model.train()
    model.set_stdp(True)
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        model(features)


def batch_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    health_criterion: nn.Module,
    fault_criterion: nn.Module,
    args: argparse.Namespace,
) -> torch.Tensor:
    del health_criterion
    health_regression_loss = outputs["health_score_pred"].new_tensor(0.0)
    if args.health_regression_weight > 0:
        if args.health_regression_loss == "l1":
            health_regression_error = (outputs["health_score_pred"] - batch["health_score"]).abs()
        elif args.health_regression_loss == "mse":
            health_regression_error = (outputs["health_score_pred"] - batch["health_score"]).square()
        else:
            health_regression_error = nn.functional.smooth_l1_loss(
                outputs["health_score_pred"],
                batch["health_score"],
                reduction="none",
            )
        low_score_weight = 1.0 + args.health_regression_low_score_weight * (
            1.0 - batch["health_score"]
        )
        high_score_weight = args.health_regression_high_score_weight * batch["health_score"]
        health_regression_loss = (
            health_regression_error * (low_score_weight + high_score_weight)
        ).mean()
    fault_loss = fault_criterion(outputs["fault_logits"], batch["fault_part_label"])
    return (
        args.health_regression_weight * health_regression_loss
        + args.fault_loss_weight * fault_loss
    )


def label_counts(loader: DataLoader, label_key: str, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in loader:
        counts += torch.bincount(batch[label_key], minlength=num_classes)
    return counts


def balanced_class_weights(counts: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = counts.float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    if present.any():
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
        weights = weights / weights[present].mean()
    return weights.to(device)


def format_counts(counts: torch.Tensor, names: list[str] | None = None) -> str:
    parts = []
    for index, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        label = names[index] if names is not None else str(index)
        parts.append(f"{label}:{count}")
    return "{" + ", ".join(parts) + "}"


def rebuild_with_balanced_sampler(
    loader: DataLoader,
    health_counts: torch.Tensor,
    fault_counts: torch.Tensor,
    sampler_balance: str,
) -> DataLoader:
    health_counts = health_counts.float().clamp_min(1.0)
    fault_counts = fault_counts.float().clamp_min(1.0)
    weights = []
    for index in range(len(loader.dataset)):
        item = loader.dataset[index]
        health_label = int(item["health_label"])
        fault_label = int(item["fault_part_label"])
        health_weight = 1.0 / float(health_counts[health_label])
        fault_weight = 1.0 / float(fault_counts[fault_label])
        if sampler_balance == "both":
            weights.append(0.5 * (health_weight + fault_weight))
        elif sampler_balance == "health":
            weights.append(health_weight)
        elif sampler_balance == "fault":
            weights.append(fault_weight)
        else:
            weights.append(1.0)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def health_predictions(outputs: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    preds = torch.round((1.0 - outputs["health_score_pred"]) * (args.health_bins - 1)).long()
    return preds.clamp(0, args.health_bins - 1)


def init_metric_totals(args: argparse.Namespace) -> dict[str, float | torch.Tensor]:
    return {
        "loss": 0.0,
        "health_correct": 0.0,
        "fault_correct": 0.0,
        "health_score_abs_error": 0.0,
        "health_score_squared_error": 0.0,
        "health_score_tolerance_correct": 0.0,
        "health_score_target_sum": 0.0,
        "health_score_target_squared_sum": 0.0,
        "health_score_fault_abs_error": 0.0,
        "health_score_fault_tolerance_correct": 0.0,
        "fault_health_count": 0.0,
        "health_score_nonhealthy_abs_error": 0.0,
        "health_score_nonhealthy_tolerance_correct": 0.0,
        "nonhealthy_count": 0.0,
        "health_bin_abs_error": torch.zeros(args.health_bins, dtype=torch.float64),
        "health_bin_tolerance_correct": torch.zeros(args.health_bins, dtype=torch.float64),
        "health_bin_counts": torch.zeros(args.health_bins, dtype=torch.float64),
        "count": 0.0,
        "health_confusion": torch.zeros(args.health_bins, args.health_bins, dtype=torch.long),
        "fault_confusion": torch.zeros(
            len(NASA_BEARING_FAULT_PARTS),
            len(NASA_BEARING_FAULT_PARTS),
            dtype=torch.long,
        ),
    }


def update_metric_totals(
    totals: dict[str, float | torch.Tensor],
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    batch_size = int(batch["features"].shape[0])
    totals["loss"] = float(totals["loss"]) + float(loss.item()) * batch_size
    health_preds = health_predictions(outputs, args)
    fault_preds = outputs["fault_logits"].argmax(dim=1)
    totals["health_correct"] = float(totals["health_correct"]) + float(
        (health_preds == batch["health_label"]).sum().item()
    )
    totals["fault_correct"] = float(totals["fault_correct"]) + float(
        (fault_preds == batch["fault_part_label"]).sum().item()
    )

    health_score_abs_error = (outputs["health_score_pred"] - batch["health_score"]).abs()
    tolerance_hits = health_score_abs_error <= args.health_score_tolerance
    totals["health_score_abs_error"] = float(totals["health_score_abs_error"]) + float(
        health_score_abs_error.sum().item()
    )
    totals["health_score_tolerance_correct"] = float(
        totals["health_score_tolerance_correct"]
    ) + float(tolerance_hits.sum().item())
    totals["health_score_squared_error"] = float(totals["health_score_squared_error"]) + float(
        (outputs["health_score_pred"] - batch["health_score"]).square().sum().item()
    )
    totals["health_score_target_sum"] = float(totals["health_score_target_sum"]) + float(
        batch["health_score"].sum().item()
    )
    totals["health_score_target_squared_sum"] = float(
        totals["health_score_target_squared_sum"]
    ) + float(batch["health_score"].square().sum().item())

    fault_mask = batch["fault_part_label"] != 0
    if fault_mask.any():
        fault_abs_error = health_score_abs_error[fault_mask]
        totals["health_score_fault_abs_error"] = float(
            totals["health_score_fault_abs_error"]
        ) + float(fault_abs_error.sum().item())
        totals["health_score_fault_tolerance_correct"] = float(
            totals["health_score_fault_tolerance_correct"]
        ) + float(tolerance_hits[fault_mask].sum().item())
        totals["fault_health_count"] = float(totals["fault_health_count"]) + float(
            fault_mask.sum().item()
        )

    nonhealthy_mask = batch["health_score"] < 0.999
    if nonhealthy_mask.any():
        nonhealthy_abs_error = health_score_abs_error[nonhealthy_mask]
        totals["health_score_nonhealthy_abs_error"] = float(
            totals["health_score_nonhealthy_abs_error"]
        ) + float(nonhealthy_abs_error.sum().item())
        totals["health_score_nonhealthy_tolerance_correct"] = float(
            totals["health_score_nonhealthy_tolerance_correct"]
        ) + float(tolerance_hits[nonhealthy_mask].sum().item())
        totals["nonhealthy_count"] = float(totals["nonhealthy_count"]) + float(
            nonhealthy_mask.sum().item()
        )

    bin_abs_error = totals["health_bin_abs_error"]
    bin_tolerance_correct = totals["health_bin_tolerance_correct"]
    bin_counts = totals["health_bin_counts"]
    if not isinstance(bin_abs_error, torch.Tensor) or not isinstance(
        bin_tolerance_correct, torch.Tensor
    ) or not isinstance(bin_counts, torch.Tensor):
        raise TypeError("health-bin totals must be tensors")
    labels_cpu = batch["health_label"].detach().cpu().long()
    bin_abs_error += torch.bincount(
        labels_cpu,
        weights=health_score_abs_error.detach().cpu().double(),
        minlength=args.health_bins,
    )
    bin_tolerance_correct += torch.bincount(
        labels_cpu,
        weights=tolerance_hits.detach().cpu().double(),
        minlength=args.health_bins,
    )
    bin_counts += torch.bincount(labels_cpu, minlength=args.health_bins).double()

    health_confusion = totals["health_confusion"]
    fault_confusion = totals["fault_confusion"]
    if not isinstance(health_confusion, torch.Tensor) or not isinstance(fault_confusion, torch.Tensor):
        raise TypeError("confusion totals must be tensors")
    update_confusion(health_confusion, batch["health_label"], health_preds)
    update_confusion(fault_confusion, batch["fault_part_label"], fault_preds)
    totals["count"] = float(totals["count"]) + batch_size


def train_epoch(
    model: NasaBearingSNN,
    loader: DataLoader,
    health_criterion: nn.Module,
    fault_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    totals = init_metric_totals(args)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch["features"])
            loss = batch_loss(outputs, batch, health_criterion, fault_criterion, args)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        update_metric_totals(totals, outputs, batch, loss, args)
    return summarize_metrics(totals)


@torch.no_grad()
def evaluate(
    model: NasaBearingSNN,
    loader: DataLoader,
    health_criterion: nn.Module,
    fault_criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    totals = init_metric_totals(args)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch["features"])
            loss = batch_loss(outputs, batch, health_criterion, fault_criterion, args)
        update_metric_totals(totals, outputs, batch, loss, args)
    return summarize_metrics(totals)


def summarize_metrics(totals: dict[str, float | torch.Tensor]) -> dict[str, float]:
    count = max(totals["count"], 1.0)
    sse = totals["health_score_squared_error"]
    target_sum = totals["health_score_target_sum"]
    target_squared_sum = totals["health_score_target_squared_sum"]
    sst = target_squared_sum - target_sum * target_sum / count
    health_score_r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
    fault_health_count = max(float(totals["fault_health_count"]), 1.0)
    nonhealthy_count = max(float(totals["nonhealthy_count"]), 1.0)
    bin_abs_error = totals["health_bin_abs_error"]
    bin_tolerance_correct = totals["health_bin_tolerance_correct"]
    bin_counts = totals["health_bin_counts"]
    health_confusion = totals["health_confusion"]
    fault_confusion = totals["fault_confusion"]
    if not isinstance(bin_abs_error, torch.Tensor) or not isinstance(
        bin_tolerance_correct, torch.Tensor
    ) or not isinstance(bin_counts, torch.Tensor):
        raise TypeError("health-bin totals must be tensors")
    if not isinstance(health_confusion, torch.Tensor) or not isinstance(fault_confusion, torch.Tensor):
        raise TypeError("confusion totals must be tensors")
    present_bins = bin_counts > 0
    if present_bins.any():
        bin_mae = bin_abs_error[present_bins] / bin_counts[present_bins].clamp_min(1.0)
        bin_acc = (
            bin_tolerance_correct[present_bins]
            / bin_counts[present_bins].clamp_min(1.0)
        )
        health_score_balanced_mae = float(bin_mae.mean().item())
        health_score_balanced_acc_at_tolerance = float(bin_acc.mean().item())
    else:
        health_score_balanced_mae = 0.0
        health_score_balanced_acc_at_tolerance = 0.0
    return {
        "loss": totals["loss"] / count,
        "health_acc": totals["health_correct"] / count,
        "fault_acc": totals["fault_correct"] / count,
        "health_macro_acc": macro_accuracy(health_confusion),
        "fault_macro_acc": macro_accuracy(fault_confusion),
        "health_score_mae": totals["health_score_abs_error"] / count,
        "health_score_rmse": (sse / count) ** 0.5,
        "health_score_r2": health_score_r2,
        "health_score_acc_at_tolerance": totals["health_score_tolerance_correct"] / count,
        "health_score_fault_mae": totals["health_score_fault_abs_error"] / fault_health_count,
        "health_score_fault_acc_at_tolerance": (
            totals["health_score_fault_tolerance_correct"] / fault_health_count
        ),
        "health_score_nonhealthy_mae": (
            totals["health_score_nonhealthy_abs_error"] / nonhealthy_count
        ),
        "health_score_nonhealthy_acc_at_tolerance": (
            totals["health_score_nonhealthy_tolerance_correct"] / nonhealthy_count
        ),
        "health_score_balanced_mae": health_score_balanced_mae,
        "health_score_balanced_acc_at_tolerance": health_score_balanced_acc_at_tolerance,
    }


def update_confusion(confusion: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor) -> None:
    targets = targets.detach().cpu().long()
    preds = preds.detach().cpu().long()
    for target, pred in zip(targets.tolist(), preds.tolist()):
        confusion[target, pred] += 1


def macro_accuracy(confusion: torch.Tensor) -> float:
    support = confusion.sum(dim=1)
    present = support > 0
    if not present.any():
        return 0.0
    per_class = confusion.diag().float()[present] / support.float()[present].clamp_min(1.0)
    return float(per_class.mean().item())


def print_run_config(
    args: argparse.Namespace,
    device: torch.device,
    input_size: int,
    effective_sampler_balance: str,
    train_health_counts: torch.Tensor,
    train_fault_counts: torch.Tensor,
    val_health_counts: torch.Tensor,
    val_fault_counts: torch.Tensor,
    amp_enabled: bool,
) -> None:
    print("== Run ==")
    print(
        f"device={device}  input_size={input_size}  amp={amp_enabled}  "
        f"checkpoint_dir={args.checkpoint_dir}"
    )
    print(
        f"sample_layout={args.sample_layout}  include_bearing_id={args.include_bearing_id}"
    )
    print(
        f"health_output=continuous_score  health_tolerance={args.health_score_tolerance:g}"
    )
    print(
        f"fault_map={args.fault_map}  fault_parts={list(NASA_BEARING_FAULT_PARTS)}"
    )
    print(
        f"class_weights={args.class_weights}  balanced_sampler={args.balanced_sampler}  "
        f"sampler_balance={effective_sampler_balance}"
    )
    print(
        f"loss_weights: health_score={args.health_regression_weight:g}  "
        f"fault={args.fault_loss_weight:g}  "
        f"low_score_boost={args.health_regression_low_score_weight:g}  "
        f"high_score_boost={args.health_regression_high_score_weight:g}  "
        f"health_score_loss={args.health_regression_loss}"
    )
    print(
        f"model: hidden=({args.hidden_size1},{args.hidden_size2})  "
        f"time_steps={args.time_steps}  readout={args.readout}  dropout={args.dropout:g}"
    )
    print("== Labels ==")
    print(f"train health: {format_counts(train_health_counts)}")
    print(f"train fault : {format_counts(train_fault_counts, list(NASA_BEARING_FAULT_PARTS))}")
    print(f"val health  : {format_counts(val_health_counts)}")
    print(f"val fault   : {format_counts(val_fault_counts, list(NASA_BEARING_FAULT_PARTS))}")
    print("note: model outputs only health_score_pred and fault_logits; health bins are derived for diagnostics only.")
    print(
        "metric_note: scoreMAE/faultMAE/binMAE are lower-is-better; "
        "hit@tol is the fraction of health-score errors within tolerance, not class accuracy."
    )
    print(f"stdp_params={json.dumps(load_stdp_params(args.stdp_params_file))}")


def print_metrics_header(tolerance: float) -> None:
    print("== Metrics ==")
    print(
        f" ep | lr       | train: loss scoreMAE faultMAE binMAE hit faultHit binHit faultMacro | "
        f"val: loss scoreMAE faultMAE binMAE hit faultHit binHit faultMacro"
    )
    print(
        "----+----------+---------------------------------------------------------------------+"
        "-------------------------------------------------------------------"
    )


def should_print_metrics_header(epoch: int, interval: int) -> bool:
    if epoch == 1:
        return True
    return interval > 0 and (epoch - 1) % interval == 0


def print_pretty_epoch(
    epoch: int,
    metrics: dict[str, float],
    tolerance: float,
) -> None:
    del tolerance
    print(
        f"{epoch:3d} | "
        f"{metrics['lr']:<8.2g} | "
        f"{metrics['train_loss']:.4f} "
        f"{metrics['train_health_score_mae']:.4f} "
        f"{metrics['train_health_score_fault_mae']:.4f} "
        f"{metrics['train_health_score_balanced_mae']:.4f} "
        f"{metrics['train_health_score_acc_at_tolerance']:.4f} "
        f"{metrics['train_health_score_fault_acc_at_tolerance']:.4f} "
        f"{metrics['train_health_score_balanced_acc_at_tolerance']:.4f} "
        f"{metrics['train_fault_macro_acc']:.4f} | "
        f"{metrics['val_loss']:.4f} "
        f"{metrics['val_health_score_mae']:.4f} "
        f"{metrics['val_health_score_fault_mae']:.4f} "
        f"{metrics['val_health_score_balanced_mae']:.4f} "
        f"{metrics['val_health_score_acc_at_tolerance']:.4f} "
        f"{metrics['val_health_score_fault_acc_at_tolerance']:.4f} "
        f"{metrics['val_health_score_balanced_acc_at_tolerance']:.4f} "
        f"{metrics['val_fault_macro_acc']:.4f}"
    )


def print_legacy_epoch(
    epoch: int,
    metrics: dict[str, float],
    tolerance: float,
) -> None:
    print(
        f"epoch={epoch:03d} "
        f"train_loss={metrics['train_loss']:.4f} "
        f"train_fault_acc={metrics['train_fault_acc']:.4f} "
        f"train_fault_macro_acc={metrics['train_fault_macro_acc']:.4f} "
        f"train_health_score_mae={metrics['train_health_score_mae']:.4f} "
        f"train_health_score_fault_mae={metrics['train_health_score_fault_mae']:.4f} "
        f"train_health_score_balanced_mae={metrics['train_health_score_balanced_mae']:.4f} "
        f"train_health_score_rmse={metrics['train_health_score_rmse']:.4f} "
        f"train_health_score_r2={metrics['train_health_score_r2']:.4f} "
        f"train_health_score_acc@{tolerance:g}="
        f"{metrics['train_health_score_acc_at_tolerance']:.4f} "
        f"val_loss={metrics['val_loss']:.4f} "
        f"val_fault_acc={metrics['val_fault_acc']:.4f} "
        f"val_fault_macro_acc={metrics['val_fault_macro_acc']:.4f} "
        f"val_health_score_mae={metrics['val_health_score_mae']:.4f} "
        f"val_health_score_fault_mae={metrics['val_health_score_fault_mae']:.4f} "
        f"val_health_score_balanced_mae={metrics['val_health_score_balanced_mae']:.4f} "
        f"val_health_score_rmse={metrics['val_health_score_rmse']:.4f} "
        f"val_health_score_r2={metrics['val_health_score_r2']:.4f} "
        f"val_health_score_acc@{tolerance:g}="
        f"{metrics['val_health_score_acc_at_tolerance']:.4f} "
        f"lr={metrics['lr']:.6g}"
    )


def selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "composite":
        return (
            metrics["val_fault_macro_acc"]
            - 0.5 * metrics["val_health_score_fault_mae"]
            - 0.5 * metrics["val_health_score_balanced_mae"]
        )
    if selection_metric == "neg_val_health_score_mae":
        return -metrics["val_health_score_mae"]
    if selection_metric == "neg_val_health_score_balanced_mae":
        return -metrics["val_health_score_balanced_mae"]
    return metrics[selection_metric]


def main() -> None:
    args = parse_args()
    effective_sampler_balance = args.sampler_balance
    if effective_sampler_balance == "auto":
        effective_sampler_balance = "both"
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_loader, val_loader, input_size = build_nasa_bearing_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        health_bins=args.health_bins,
        health_label_mode=args.health_label_mode,
        fault_map=args.fault_map,
        include_bearing_id=args.include_bearing_id,
        sample_layout=args.sample_layout,
        val_fraction=args.val_fraction,
        limit_good=args.limit_good,
        limit_fault=args.limit_fault,
        seed=args.seed,
    )
    model = make_model(args, input_size=input_size, device=device)
    train_health_counts = label_counts(train_loader, "health_label", args.health_bins)
    train_fault_counts = label_counts(
        train_loader,
        "fault_part_label",
        len(NASA_BEARING_FAULT_PARTS),
    )
    val_health_counts = label_counts(val_loader, "health_label", args.health_bins)
    val_fault_counts = label_counts(
        val_loader,
        "fault_part_label",
        len(NASA_BEARING_FAULT_PARTS),
    )
    if args.balanced_sampler:
        train_loader = rebuild_with_balanced_sampler(
            train_loader,
            health_counts=train_health_counts,
            fault_counts=train_fault_counts,
            sampler_balance=effective_sampler_balance,
        )
    health_weights = (
        balanced_class_weights(train_health_counts, device)
        if args.class_weights == "balanced"
        else None
    )
    fault_weights = (
        balanced_class_weights(train_fault_counts, device)
        if args.class_weights == "balanced"
        else None
    )
    health_criterion = nn.CrossEntropyLoss(weight=health_weights)
    fault_criterion = nn.CrossEntropyLoss(weight=fault_weights)
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        if args.scheduler == "cosine"
        else None
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_dir = Path(args.checkpoint_dir)

    print_run_config(
        args=args,
        device=device,
        input_size=input_size,
        effective_sampler_balance=effective_sampler_balance,
        train_health_counts=train_health_counts,
        train_fault_counts=train_fault_counts,
        val_health_counts=val_health_counts,
        val_fault_counts=val_fault_counts,
        amp_enabled=amp_enabled,
    )

    if args.stdp_pretrain:
        model.set_stdp(True, args.stdp_lr)
        for epoch in range(1, args.stdp_epochs + 1):
            stdp_pretrain_epoch(model, train_loader, device)
            print(f"stdp_epoch={epoch}/{args.stdp_epochs} done")

    model.set_stdp(args.stdp_during_supervised, args.stdp_lr)
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            health_criterion,
            fault_criterion,
            optimizer,
            device,
            args,
            scaler,
            amp_enabled,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            health_criterion,
            fault_criterion,
            device,
            args,
            amp_enabled,
        )
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        metrics = {
            "train_loss": train_metrics["loss"],
            "train_health_acc": train_metrics["health_acc"],
            "train_health_stage_acc": train_metrics["health_acc"],
            "train_fault_acc": train_metrics["fault_acc"],
            "train_health_macro_acc": train_metrics["health_macro_acc"],
            "train_health_stage_macro_acc": train_metrics["health_macro_acc"],
            "train_fault_macro_acc": train_metrics["fault_macro_acc"],
            "train_health_score_mae": train_metrics["health_score_mae"],
            "train_health_score_fault_mae": train_metrics["health_score_fault_mae"],
            "train_health_score_nonhealthy_mae": train_metrics["health_score_nonhealthy_mae"],
            "train_health_score_balanced_mae": train_metrics["health_score_balanced_mae"],
            "train_health_score_rmse": train_metrics["health_score_rmse"],
            "train_health_score_r2": train_metrics["health_score_r2"],
            "train_health_score_acc_at_tolerance": train_metrics["health_score_acc_at_tolerance"],
            "train_health_score_fault_acc_at_tolerance": train_metrics[
                "health_score_fault_acc_at_tolerance"
            ],
            "train_health_score_nonhealthy_acc_at_tolerance": train_metrics[
                "health_score_nonhealthy_acc_at_tolerance"
            ],
            "train_health_score_balanced_acc_at_tolerance": train_metrics[
                "health_score_balanced_acc_at_tolerance"
            ],
            "val_loss": val_metrics["loss"],
            "val_health_acc": val_metrics["health_acc"],
            "val_health_stage_acc": val_metrics["health_acc"],
            "val_fault_acc": val_metrics["fault_acc"],
            "val_health_macro_acc": val_metrics["health_macro_acc"],
            "val_health_stage_macro_acc": val_metrics["health_macro_acc"],
            "val_fault_macro_acc": val_metrics["fault_macro_acc"],
            "val_health_score_mae": val_metrics["health_score_mae"],
            "val_health_score_fault_mae": val_metrics["health_score_fault_mae"],
            "val_health_score_nonhealthy_mae": val_metrics["health_score_nonhealthy_mae"],
            "val_health_score_balanced_mae": val_metrics["health_score_balanced_mae"],
            "val_health_score_rmse": val_metrics["health_score_rmse"],
            "val_health_score_r2": val_metrics["health_score_r2"],
            "val_health_score_acc_at_tolerance": val_metrics["health_score_acc_at_tolerance"],
            "val_health_score_fault_acc_at_tolerance": val_metrics[
                "health_score_fault_acc_at_tolerance"
            ],
            "val_health_score_nonhealthy_acc_at_tolerance": val_metrics[
                "health_score_nonhealthy_acc_at_tolerance"
            ],
            "val_health_score_balanced_acc_at_tolerance": val_metrics[
                "health_score_balanced_acc_at_tolerance"
            ],
            "lr": current_lr,
        }
        if args.log_format == "pretty":
            if should_print_metrics_header(epoch, args.log_header_interval):
                print_metrics_header(args.health_score_tolerance)
            print_pretty_epoch(epoch, metrics, args.health_score_tolerance)
        else:
            print_legacy_epoch(epoch, metrics, args.health_score_tolerance)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, args, metrics)
        score = selection_score(metrics, args.selection_metric)
        if score > best_score:
            best_score = score
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, args, metrics)


if __name__ == "__main__":
    main()
