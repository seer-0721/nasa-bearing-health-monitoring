from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from data import CIFAR10_CLASSES, CIFAR10_INPUT_SIZE, build_cifar10_loaders, poisson_encode
from model import ConvLifModelSTDP, LifModelSTDP
from utils import load_checkpoint, load_stdp_params, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 STDP/SNN checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    parser.add_argument("--stdp-params-file", type=str, default="stdp_params.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model_from_checkpoint(
    checkpoint: dict,
    stdp_params_file: str,
    device: torch.device,
) -> LifModelSTDP | ConvLifModelSTDP:
    train_args = checkpoint.get("args", {})
    stdp_params = load_stdp_params(stdp_params_file)
    model_type = train_args.get("model_type", "mlp-stdp")
    if model_type == "conv-stdp":
        model = ConvLifModelSTDP(
            hidden_size1=train_args.get("hidden_size1", 256),
            hidden_size2=train_args.get("hidden_size2", 128),
            output_size=CIFAR10_CLASSES,
            beta=train_args.get("beta", 0.9),
            threshold=train_args.get("threshold", 1.0),
            stdp_params=stdp_params,
            stdp_lr=train_args.get("stdp_lr", 1e-6),
            conv_channels=train_args.get("conv_channels", 96),
        ).to(device)
    else:
        model = LifModelSTDP(
            input_size=CIFAR10_INPUT_SIZE,
            hidden_size1=train_args.get("hidden_size1", 256),
            hidden_size2=train_args.get("hidden_size2", 128),
            output_size=CIFAR10_CLASSES,
            beta=train_args.get("beta", 0.9),
            threshold=train_args.get("threshold", 1.0),
            stdp_params=stdp_params,
            stdp_lr=train_args.get("stdp_lr", 1e-6),
        ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    model: LifModelSTDP,
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    time_steps: int,
) -> tuple[float, float]:
    _, test_loader = build_cifar10_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        download=False,
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(poisson_encode(images, time_steps))
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    time_steps = args.time_steps or train_args.get("time_steps", 10)

    model = build_model_from_checkpoint(checkpoint, args.stdp_params_file, device)
    test_loss, test_acc = evaluate(
        model=model,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        time_steps=time_steps,
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"time_steps={time_steps}")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print(f"saved_test_loss={metrics.get('test_loss', float('nan')):.4f}")
        print(f"saved_test_acc={metrics.get('test_acc', float('nan')):.4f}")


if __name__ == "__main__":
    main()
