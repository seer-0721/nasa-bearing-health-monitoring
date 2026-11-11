from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_cifar10_loaders, build_shd_loaders, encode_spikes
from eval import (
    build_model_from_checkpoint,
    checkpoint_dataset,
    dataset_input_size,
    is_snn_model,
)
from utils import load_checkpoint, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect spike activity for an SNN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, choices=["cifar10", "shd"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--stdp-params-file", type=str, default="stdp_params.npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--shd-duration", type=float, default=None)
    parser.add_argument("--shd-input-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def build_loader(args: argparse.Namespace, train_args: dict, dataset: str, time_steps: int):
    if dataset == "shd":
        shd_duration = args.shd_duration or train_args.get("shd_duration", 1.0)
        shd_input_size = dataset_input_size(train_args, dataset, args.shd_input_size)
        train_loader, test_loader = build_shd_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            time_bins=time_steps,
            duration=shd_duration,
            input_size=shd_input_size,
        )
    else:
        train_loader, test_loader = build_cifar10_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=False,
            normalize=not is_snn_model(train_args),
        )
    return train_loader if args.split == "train" else test_loader


def weighted_add(
    total_stats: dict[str, float],
    batch_stats: dict[str, float],
    batch_size: int,
) -> None:
    for key, value in batch_stats.items():
        total_stats[key] = total_stats.get(key, 0.0) + float(value) * batch_size


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    dataset = checkpoint_dataset(train_args, args.dataset)
    time_steps = args.time_steps or train_args.get("time_steps", 10)
    encoding = args.encoding or train_args.get("encoding", "poisson")
    shd_input_size = dataset_input_size(train_args, dataset, args.shd_input_size)

    model = build_model_from_checkpoint(
        checkpoint,
        args.stdp_params_file,
        device,
        dataset=dataset,
        input_size=shd_input_size,
    )
    if not hasattr(model, "spike_stats"):
        raise TypeError(
            f"{train_args.get('model_type', 'unknown')} does not expose spike_stats(). "
            "The current diagnostic supports checkpoints with a spike_stats() method."
        )

    loader = build_loader(args, train_args, dataset, time_steps)
    amp_enabled = args.amp and device.type == "cuda"
    total_stats: dict[str, float] = {}
    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            spikes = inputs if dataset == "shd" else encode_spikes(inputs, time_steps, encoding)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(spikes)
                batch_stats = model.spike_stats(spikes)

            batch_size = labels.size(0)
            weighted_add(total_stats, batch_stats, batch_size)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += batch_size

    if total == 0:
        raise RuntimeError("No samples were loaded for spike diagnostics.")

    stats = {key: value / total for key, value in total_stats.items()}
    stdp_weight = model.stdp1.weight.detach()
    stats.update(
        {
            "diagnostic_acc": correct / total,
            "samples": total,
            "stdp_weight_min": float(stdp_weight.min().item()),
            "stdp_weight_max": float(stdp_weight.max().item()),
            "stdp_weight_mean": float(stdp_weight.mean().item()),
            "stdp_weight_std": float(stdp_weight.std().item()),
            "stdp_weight_at_min_fraction": float((stdp_weight <= model.stdp1.w_min).float().mean().item()),
            "stdp_weight_at_max_fraction": float((stdp_weight >= model.stdp1.w_max).float().mean().item()),
        }
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"dataset={dataset}")
    print(f"split={args.split}")
    print(f"model_type={train_args.get('model_type', 'unknown')}")
    print(f"time_steps={time_steps}")
    print(f"encoding={'preencoded-shd' if dataset == 'shd' else encoding}")
    print(f"stdp_enabled_during_diagnostic={getattr(model, 'stdp_enabled', 'unknown')}")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
