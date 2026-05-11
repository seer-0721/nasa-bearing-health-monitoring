from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import (
    CIFAR10_CLASSES,
    CIFAR10_INPUT_SIZE,
    build_cifar10_loaders,
    poisson_encode,
)
from model import ConvLifModelSTDP, LifModelSTDP
from utils import load_stdp_params, resolve_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an STDP/LIF network on CIFAR-10.")
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--stdp-params-file", type=str, default="stdp_params.npz")

    parser.add_argument(
        "--model-type",
        type=str,
        default="mlp-stdp",
        choices=["mlp-stdp", "conv-stdp"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--hidden-size1", type=int, default=256)
    parser.add_argument("--hidden-size2", type=int, default=128)
    parser.add_argument("--conv-channels", type=int, default=96)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)

    parser.add_argument("--stdp-pretrain", action="store_true")
    parser.add_argument("--stdp-epochs", type=int, default=1)
    parser.add_argument("--stdp-lr", type=float, default=1e-6)
    parser.add_argument(
        "--stdp-during-supervised",
        action="store_true",
        help="Keep STDP updates enabled while supervised layers are trained.",
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    return parser.parse_args()


def make_model(args: argparse.Namespace, device: torch.device) -> LifModelSTDP:
    stdp_params = load_stdp_params(args.stdp_params_file)
    if args.model_type == "conv-stdp":
        return ConvLifModelSTDP(
            hidden_size1=args.hidden_size1,
            hidden_size2=args.hidden_size2,
            output_size=CIFAR10_CLASSES,
            beta=args.beta,
            threshold=args.threshold,
            stdp_params=stdp_params,
            stdp_lr=args.stdp_lr,
            conv_channels=args.conv_channels,
        ).to(device)
    return LifModelSTDP(
            input_size=CIFAR10_INPUT_SIZE,
            hidden_size1=args.hidden_size1,
            hidden_size2=args.hidden_size2,
            output_size=CIFAR10_CLASSES,
            beta=args.beta,
            threshold=args.threshold,
            stdp_params=stdp_params,
            stdp_lr=args.stdp_lr,
        ).to(device)


@torch.no_grad()
def stdp_pretrain_epoch(
    model: LifModelSTDP,
    loader: DataLoader,
    device: torch.device,
    time_steps: int,
) -> None:
    model.train()
    model.set_stdp(True)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        spikes = poisson_encode(images, time_steps)
        model(spikes)


def train_epoch(
    model: LifModelSTDP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    time_steps: int,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        spikes = poisson_encode(images, time_steps)

        optimizer.zero_grad(set_to_none=True)
        logits = model(spikes)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: LifModelSTDP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    time_steps: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        spikes = poisson_encode(images, time_steps)
        logits = model(spikes)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_loader, test_loader = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        augment=args.augment,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    model = make_model(args, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        if args.scheduler == "cosine"
        else None
    )

    stdp_params = load_stdp_params(args.stdp_params_file)
    print(f"device={device}")
    print(f"stdp_params={json.dumps(stdp_params, indent=2)}")

    if args.stdp_pretrain:
        model.set_stdp(True, args.stdp_lr)
        for epoch in range(1, args.stdp_epochs + 1):
            stdp_pretrain_epoch(model, train_loader, device, args.time_steps)
            print(f"stdp_epoch={epoch}/{args.stdp_epochs} done")

    model.set_stdp(args.stdp_during_supervised, args.stdp_lr)
    best_acc = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.time_steps,
            args.grad_clip,
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, args.time_steps
        )
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": current_lr,
        }
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"lr={current_lr:.6g}"
        )

        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, args, metrics)
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, args, metrics)


if __name__ == "__main__":
    main()
