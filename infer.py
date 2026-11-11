from __future__ import annotations

import argparse

import torch

from data import DEFAULT_FAULT_MAP, NASA_BEARING_FAULT_PARTS, build_nasa_bearing_loaders
from eval import build_model
from utils import load_checkpoint, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NASA Bearing STDP/SNN inference.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nasa_bearing_snn/best.pt")
    parser.add_argument("--data-root", type=str, default="data/NasaBearingDataset")
    parser.add_argument("--stdp-params-file", type=str, default="stdp_params.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="first",
        choices=["first", "balanced", "errors"],
        help="first prints loader order; balanced prints a few samples per target fault class; errors prints mistakes first.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--health-error-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_predictions(
    model,
    loader,
    device: torch.device,
    health_bins: int,
    health_error_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            outputs = model(features)
            fault_probs = outputs["fault_logits"].softmax(dim=1)
            health_score_preds = torch.round(
                (1.0 - outputs["health_score_pred"]) * (health_bins - 1)
            ).long()
            health_score_preds = health_score_preds.clamp(0, health_bins - 1)
            fault_preds = fault_probs.argmax(dim=1)
            health_expected = outputs["health_score_pred"]

            for idx in range(features.shape[0]):
                health_class = int(health_score_preds[idx].item())
                fault_class = int(fault_preds[idx].item())
                target_health_class = int(batch["health_label"][idx].item())
                target_fault_class = int(batch["fault_part_label"][idx].item())
                pred_health_score = float(health_expected[idx].item())
                target_health_score = float(batch["health_score"][idx].item())
                health_score_abs_error = abs(pred_health_score - target_health_score)
                is_health_error = health_score_abs_error > health_error_threshold
                rows.append(
                    {
                        "bearing_index": int(batch["bearing_index"][idx].item()),
                        "source": "fault" if int(batch["source_is_fault"][idx].item()) else "good",
                        "source_position": int(batch["source_position"][idx].item()),
                        "pred_health_score": pred_health_score,
                        "pred_health_class_from_score": int(health_score_preds[idx].item()),
                        "target_health_score": target_health_score,
                        "health_score_abs_error": health_score_abs_error,
                        "target_health_class": target_health_class,
                        "pred_fault_class": fault_class,
                        "pred_fault_conf": float(fault_probs[idx, fault_class].item()),
                        "target_fault_class": target_fault_class,
                        "is_error": is_health_error or fault_class != target_fault_class,
                    }
                )
    return rows


def choose_rows(rows: list[dict[str, object]], sample_mode: str, limit: int) -> list[dict[str, object]]:
    if sample_mode == "errors":
        selected = [row for row in rows if row["is_error"]]
        return selected[:limit]
    if sample_mode == "balanced":
        selected: list[dict[str, object]] = []
        per_class_limit = max(1, limit // len(NASA_BEARING_FAULT_PARTS))
        for class_index in range(len(NASA_BEARING_FAULT_PARTS)):
            class_rows = [
                row for row in rows if int(row["target_fault_class"]) == class_index
            ]
            selected.extend(class_rows[:per_class_limit])
        if len(selected) < limit:
            selected.extend(row for row in rows if row not in selected)
        return selected[:limit]
    return rows[:limit]


def print_row(row: dict[str, object]) -> None:
    fault_class = int(row["pred_fault_class"])
    target_fault_class = int(row["target_fault_class"])
    print(
        f"B{row['bearing_index']} "
        f"source={row['source']} "
        f"pos={row['source_position']} "
        f"pred_health_score={float(row['pred_health_score']):.3f} "
        f"pred_health_class_from_score={row['pred_health_class_from_score']} "
        f"target_health_score={float(row['target_health_score']):.3f} "
        f"health_score_abs_error={float(row['health_score_abs_error']):.3f} "
        f"target_health_class={row['target_health_class']} "
        f"fault_part={NASA_BEARING_FAULT_PARTS[fault_class]} "
        f"fault_conf={float(row['pred_fault_conf']):.3f} "
        f"target_fault_part={NASA_BEARING_FAULT_PARTS[target_fault_class]}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    _, loader, input_size = build_nasa_bearing_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        health_bins=train_args.get("health_bins", 11),
        health_label_mode=train_args.get("health_label_mode", "time"),
        fault_map=train_args.get("fault_map", DEFAULT_FAULT_MAP),
        include_bearing_id=train_args.get("include_bearing_id", False),
        sample_layout=train_args.get("sample_layout", "per_bearing"),
        val_fraction=train_args.get("val_fraction", 0.2),
        limit_good=train_args.get("limit_good"),
        limit_fault=train_args.get("limit_fault"),
        seed=train_args.get("seed", args.seed),
    )
    model = build_model(checkpoint, input_size, args.stdp_params_file, device)

    rows = collect_predictions(
        model,
        loader,
        device,
        health_bins=train_args.get("health_bins", 11),
        health_error_threshold=args.health_error_threshold,
    )
    for row in choose_rows(rows, args.sample_mode, args.limit):
        print_row(row)


if __name__ == "__main__":
    main()
