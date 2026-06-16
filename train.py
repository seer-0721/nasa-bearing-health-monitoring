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
    encode_spikes,
)
from model import CifarResNet, ConvLifModelSTDP, LifModelSTDP, SpikingConvLifModelSTDP
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
        choices=["mlp-stdp", "conv-stdp", "spiking-conv-stdp", "cifar-resnet"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument(
        "--encoding",
        type=str,
        default="poisson",
        choices=["poisson", "deterministic-rate", "latency"],
    )
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--hidden-size1", type=int, default=256)
    parser.add_argument("--hidden-size2", type=int, default=128)
    parser.add_argument("--conv-channels", type=int, default=96)
    parser.add_argument("--resnet-width", type=int, default=64)
    parser.add_argument("--resnet-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--stdp-current-scale", type=float, default=None)
    parser.add_argument("--auto-augment", action="store_true")
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")

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


def make_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    stdp_params = load_stdp_params(args.stdp_params_file)
    if args.model_type == "cifar-resnet":
        return CifarResNet(
            num_classes=CIFAR10_CLASSES,
            width=args.resnet_width,
            blocks_per_stage=args.resnet_blocks,
            dropout=args.dropout,
        ).to(device)
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
            stdp_current_scale=args.stdp_current_scale,
        ).to(device)
    if args.model_type == "spiking-conv-stdp":
        return SpikingConvLifModelSTDP(
            hidden_size1=args.hidden_size1,
            hidden_size2=args.hidden_size2,
            output_size=CIFAR10_CLASSES,
            beta=args.beta,
            threshold=args.threshold,
            stdp_params=stdp_params,
            stdp_lr=args.stdp_lr,
            conv_channels=args.conv_channels,
            stdp_current_scale=args.stdp_current_scale,
            dropout=args.dropout,
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


def is_snn_model(args: argparse.Namespace | dict[str, object]) -> bool:
    model_type = args["model_type"] if isinstance(args, dict) else args.model_type
    return model_type in {"mlp-stdp", "conv-stdp", "spiking-conv-stdp"}


@torch.no_grad()
def stdp_pretrain_epoch(
    model: LifModelSTDP,
    loader: DataLoader,
    device: torch.device,
    time_steps: int,
    encoding: str,
) -> None:
    model.train()
    model.set_stdp(True)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        spikes = encode_spikes(images, time_steps, encoding)
        model(spikes)


def forward_model(model: nn.Module, images: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if is_snn_model(args):
        return model(encode_spikes(images, args.time_steps, args.encoding))
    return model(images)


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    indices = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[indices]
    return mixed_images, labels, labels[indices], lam


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    grad_clip: float,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if not is_snn_model(args) and args.mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_batch(images, labels, args.mixup_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = forward_model(model, images, args)
            loss = mixup_loss(criterion, logits, labels_a, labels_b, lam)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            repeat_count = max(args.eval_repeats, 1) if is_snn_model(args) else 1
            logits = forward_model(model, images, args)
            for _ in range(1, repeat_count):
                logits = logits + forward_model(model, images, args)
            logits = logits / repeat_count
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    normalize = args.model_type == "cifar-resnet"

    train_loader, test_loader = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        augment=args.augment,
        normalize=normalize,
        auto_augment=args.auto_augment,
        random_erasing=args.random_erasing,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    model = make_model(args, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
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
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    stdp_params = load_stdp_params(args.stdp_params_file)
    print(f"device={device}")
    print(f"model_type={args.model_type}")
    print(f"encoding={args.encoding if is_snn_model(args) else 'none'}")
    print(f"normalize={normalize}")
    print(f"amp={amp_enabled}")
    print(f"stdp_params={json.dumps(stdp_params, indent=2)}")

    if args.stdp_pretrain:
        if not is_snn_model(args):
            raise ValueError("--stdp-pretrain is only supported for mlp-stdp and conv-stdp")
        model.set_stdp(True, args.stdp_lr)
        for epoch in range(1, args.stdp_epochs + 1):
            stdp_pretrain_epoch(model, train_loader, device, args.time_steps, args.encoding)
            print(f"stdp_epoch={epoch}/{args.stdp_epochs} done")

    if is_snn_model(args):
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
            args,
            args.grad_clip,
            scaler,
            amp_enabled,
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, args, amp_enabled
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
