from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from data import DEFAULT_FAULT_MAP, NASA_BEARING_FAULT_PARTS, build_nasa_bearing_loaders
from model import NasaBearingSNN
from utils import load_checkpoint, load_stdp_params, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a NASA Bearing STDP/SNN checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nasa_bearing_snn/best.pt")
    parser.add_argument("--data-root", type=str, default="data/NasaBearingDataset")
    parser.add_argument("--stdp-params-file", type=str, default="stdp_params.npz")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--health-label-mode",
        type=str,
        default=None,
        choices=["time", "source", "anomaly"],
    )
    parser.add_argument("--fault-map", type=str, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--limit-good", type=int, default=None)
    parser.add_argument("--limit-fault", type=int, default=None)
    parser.add_argument(
        "--sample-layout",
        type=str,
        default=None,
        choices=["per_bearing", "row"],
    )
    parser.add_argument("--health-score-tolerance", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(checkpoint: dict, input_size: int, stdp_params_file: str, device: torch.device) -> NasaBearingSNN:
    train_args = checkpoint.get("args", {})
    model = NasaBearingSNN(
        input_size=input_size,
        hidden_size1=train_args.get("hidden_size1", 256),
        hidden_size2=train_args.get("hidden_size2", 128),
        fault_classes=len(NASA_BEARING_FAULT_PARTS),
        time_steps=train_args.get("time_steps", 16),
        beta=train_args.get("beta", 0.9),
        threshold=train_args.get("threshold", 1.0),
        stdp_params=load_stdp_params(stdp_params_file),
        stdp_lr=train_args.get("stdp_lr", 1e-6),
        stdp_ltd_scale=train_args.get("stdp_ltd_scale", 1.0),
        readout=train_args.get("readout", "hybrid"),
        dropout=train_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    model: NasaBearingSNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    health_decision: str = "classification",
    train_args: dict | None = None,
    health_score_tolerance: float = 0.1,
) -> dict[str, float]:
    train_args = train_args or {}
    health_regression_weight = train_args.get(
        "health_regression_weight",
        1.0,
    )
    health_regression_loss = train_args.get("health_regression_loss", "smooth_l1")
    health_regression_low_score_weight = train_args.get(
        "health_regression_low_score_weight",
        0.0,
    )
    health_regression_high_score_weight = train_args.get(
        "health_regression_high_score_weight",
        0.0,
    )
    fault_loss_weight = train_args.get("fault_loss_weight", 1.0)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    health_correct = 0
    fault_correct = 0
    health_score_abs_error = 0.0
    health_score_squared_error = 0.0
    health_score_tolerance_correct = 0.0
    health_score_target_sum = 0.0
    health_score_target_squared_sum = 0.0
    health_score_fault_abs_error = 0.0
    health_score_fault_tolerance_correct = 0.0
    fault_health_count = 0.0
    health_score_nonhealthy_abs_error = 0.0
    health_score_nonhealthy_tolerance_correct = 0.0
    nonhealthy_count = 0.0
    total = 0
    health_bins = train_args.get("health_bins", 11)
    health_bin_abs_error = torch.zeros(health_bins, dtype=torch.float64)
    health_bin_tolerance_correct = torch.zeros(health_bins, dtype=torch.float64)
    health_bin_counts = torch.zeros(health_bins, dtype=torch.float64)
    health_confusion = torch.zeros(health_bins, health_bins, dtype=torch.long)
    fault_confusion = torch.zeros(model.fault_classes, model.fault_classes, dtype=torch.long)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        outputs = model(batch["features"])
        loss = outputs["health_score_pred"].new_tensor(0.0)
        if health_regression_weight > 0:
            if health_regression_loss == "l1":
                regression_error = (outputs["health_score_pred"] - batch["health_score"]).abs()
            elif health_regression_loss == "mse":
                regression_error = (outputs["health_score_pred"] - batch["health_score"]).square()
            else:
                regression_error = nn.functional.smooth_l1_loss(
                    outputs["health_score_pred"],
                    batch["health_score"],
                    reduction="none",
                )
            regression_weight = (
                1.0
                + health_regression_low_score_weight * (1.0 - batch["health_score"])
                + health_regression_high_score_weight * batch["health_score"]
            )
            regression_loss = (regression_error * regression_weight).mean()
            loss = loss + health_regression_weight * regression_loss
        loss = loss + fault_loss_weight * criterion(
            outputs["fault_logits"],
            batch["fault_part_label"],
        )
        batch_size = int(batch["features"].shape[0])
        total_loss += float(loss.item()) * batch_size
        health_preds = health_predictions(outputs, health_bins, health_decision)
        fault_preds = outputs["fault_logits"].argmax(dim=1)
        health_correct += int((health_preds == batch["health_label"]).sum().item())
        fault_correct += int((fault_preds == batch["fault_part_label"]).sum().item())
        if "health_score_pred" in outputs:
            abs_error = (outputs["health_score_pred"] - batch["health_score"]).abs()
            tolerance_hits = abs_error <= health_score_tolerance
            health_score_abs_error += float(abs_error.sum().item())
            health_score_tolerance_correct += float(
                tolerance_hits.sum().item()
            )
            health_score_squared_error += float(
                (outputs["health_score_pred"] - batch["health_score"]).square().sum().item()
            )
            health_score_target_sum += float(batch["health_score"].sum().item())
            health_score_target_squared_sum += float(batch["health_score"].square().sum().item())
            fault_mask = batch["fault_part_label"] != 0
            if fault_mask.any():
                fault_abs_error = abs_error[fault_mask]
                health_score_fault_abs_error += float(fault_abs_error.sum().item())
                health_score_fault_tolerance_correct += float(
                    tolerance_hits[fault_mask].sum().item()
                )
                fault_health_count += float(fault_mask.sum().item())
            nonhealthy_mask = batch["health_score"] < 0.999
            if nonhealthy_mask.any():
                nonhealthy_abs_error = abs_error[nonhealthy_mask]
                health_score_nonhealthy_abs_error += float(nonhealthy_abs_error.sum().item())
                health_score_nonhealthy_tolerance_correct += float(
                    tolerance_hits[nonhealthy_mask].sum().item()
                )
                nonhealthy_count += float(nonhealthy_mask.sum().item())
            labels_cpu = batch["health_label"].detach().cpu().long()
            health_bin_abs_error += torch.bincount(
                labels_cpu,
                weights=abs_error.detach().cpu().double(),
                minlength=health_bins,
            )
            health_bin_tolerance_correct += torch.bincount(
                labels_cpu,
                weights=tolerance_hits.detach().cpu().double(),
                minlength=health_bins,
            )
            health_bin_counts += torch.bincount(labels_cpu, minlength=health_bins).double()
        update_confusion(health_confusion, batch["health_label"], health_preds)
        update_confusion(fault_confusion, batch["fault_part_label"], fault_preds)
        total += batch_size
    total = max(total, 1)
    sst = health_score_target_squared_sum - health_score_target_sum * health_score_target_sum / total
    health_score_r2 = (
        1.0 - health_score_squared_error / sst
        if sst > 1e-12
        else 0.0
    )
    present_bins = health_bin_counts > 0
    if present_bins.any():
        bin_mae = health_bin_abs_error[present_bins] / health_bin_counts[present_bins].clamp_min(1.0)
        bin_acc = (
            health_bin_tolerance_correct[present_bins]
            / health_bin_counts[present_bins].clamp_min(1.0)
        )
        health_score_balanced_mae = float(bin_mae.mean().item())
        health_score_balanced_acc_at_tolerance = float(bin_acc.mean().item())
    else:
        health_score_balanced_mae = 0.0
        health_score_balanced_acc_at_tolerance = 0.0
    fault_health_denominator = max(fault_health_count, 1.0)
    nonhealthy_denominator = max(nonhealthy_count, 1.0)
    return {
        "loss": total_loss / total,
        "health_acc": health_correct / total,
        "fault_acc": fault_correct / total,
        "health_macro_acc": macro_accuracy(health_confusion),
        "fault_macro_acc": macro_accuracy(fault_confusion),
        "health_score_mae": health_score_abs_error / total,
        "health_score_rmse": (health_score_squared_error / total) ** 0.5,
        "health_score_r2": health_score_r2,
        "health_score_acc_at_tolerance": health_score_tolerance_correct / total,
        "health_score_fault_mae": health_score_fault_abs_error / fault_health_denominator,
        "health_score_fault_acc_at_tolerance": (
            health_score_fault_tolerance_correct / fault_health_denominator
        ),
        "health_score_nonhealthy_mae": (
            health_score_nonhealthy_abs_error / nonhealthy_denominator
        ),
        "health_score_nonhealthy_acc_at_tolerance": (
            health_score_nonhealthy_tolerance_correct / nonhealthy_denominator
        ),
        "health_score_balanced_mae": health_score_balanced_mae,
        "health_score_balanced_acc_at_tolerance": health_score_balanced_acc_at_tolerance,
    }


def health_predictions(
    outputs: dict[str, torch.Tensor],
    health_bins: int,
    health_decision: str,
) -> torch.Tensor:
    if health_decision == "regression" and "health_score_pred" in outputs:
        preds = torch.round((1.0 - outputs["health_score_pred"]) * (health_bins - 1)).long()
        return preds.clamp(0, health_bins - 1)
    preds = torch.round((1.0 - outputs["health_score_pred"]) * (health_bins - 1)).long()
    return preds.clamp(0, health_bins - 1)


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    _, val_loader, input_size = build_nasa_bearing_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        health_bins=train_args.get("health_bins", 11),
        health_label_mode=args.health_label_mode
        or train_args.get("health_label_mode", "time"),
        fault_map=args.fault_map or train_args.get("fault_map", DEFAULT_FAULT_MAP),
        include_bearing_id=train_args.get("include_bearing_id", False),
        sample_layout=args.sample_layout or train_args.get("sample_layout", "per_bearing"),
        val_fraction=args.val_fraction or train_args.get("val_fraction", 0.2),
        limit_good=args.limit_good if args.limit_good is not None else train_args.get("limit_good"),
        limit_fault=args.limit_fault if args.limit_fault is not None else train_args.get("limit_fault"),
        seed=train_args.get("seed", args.seed),
    )
    model = build_model(checkpoint, input_size, args.stdp_params_file, device)
    health_decision = train_args.get("health_decision", "regression")
    metrics = evaluate(
        model,
        val_loader,
        device,
        health_decision=health_decision,
        train_args=train_args,
        health_score_tolerance=args.health_score_tolerance,
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"input_size={input_size}")
    print(f"health_output=continuous_score")
    print(f"health_decision=regression")
    print(f"loss={metrics['loss']:.4f}")
    print(f"derived_health_stage_acc={metrics['health_acc']:.4f}")
    print(f"fault_acc={metrics['fault_acc']:.4f}")
    print(f"derived_health_stage_macro_acc={metrics['health_macro_acc']:.4f}")
    print(f"fault_macro_acc={metrics['fault_macro_acc']:.4f}")
    print(f"health_score_mae={metrics['health_score_mae']:.4f}")
    print(f"health_score_fault_mae={metrics['health_score_fault_mae']:.4f}")
    print(f"health_score_nonhealthy_mae={metrics['health_score_nonhealthy_mae']:.4f}")
    print(f"health_score_balanced_mae={metrics['health_score_balanced_mae']:.4f}")
    print(f"health_score_rmse={metrics['health_score_rmse']:.4f}")
    print(f"health_score_r2={metrics['health_score_r2']:.4f}")
    print(
        f"health_score_hit@{args.health_score_tolerance:g}="
        f"{metrics['health_score_acc_at_tolerance']:.4f}"
    )
    print(
        f"health_score_fault_hit@{args.health_score_tolerance:g}="
        f"{metrics['health_score_fault_acc_at_tolerance']:.4f}"
    )
    print(
        f"health_score_balanced_hit@{args.health_score_tolerance:g}="
        f"{metrics['health_score_balanced_acc_at_tolerance']:.4f}"
    )
    if "metrics" in checkpoint:
        saved = checkpoint["metrics"]
        print(f"saved_val_health_acc={saved.get('val_health_acc', float('nan')):.4f}")
        print(f"saved_val_fault_acc={saved.get('val_fault_acc', float('nan')):.4f}")
        print(f"saved_val_health_macro_acc={saved.get('val_health_macro_acc', float('nan')):.4f}")
        print(f"saved_val_fault_macro_acc={saved.get('val_fault_macro_acc', float('nan')):.4f}")
        print(f"saved_val_health_score_mae={saved.get('val_health_score_mae', float('nan')):.4f}")
        print(
            "saved_val_health_score_fault_mae="
            f"{saved.get('val_health_score_fault_mae', float('nan')):.4f}"
        )
        print(
            "saved_val_health_score_balanced_mae="
            f"{saved.get('val_health_score_balanced_mae', float('nan')):.4f}"
        )
        print(f"saved_val_health_score_rmse={saved.get('val_health_score_rmse', float('nan')):.4f}")
        print(f"saved_val_health_score_r2={saved.get('val_health_score_r2', float('nan')):.4f}")


if __name__ == "__main__":
    main()
