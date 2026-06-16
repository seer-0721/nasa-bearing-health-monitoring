from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import (
    CIFAR10_CLASSES,
    CIFAR10_INPUT_SIZE,
    CIFAR10_MEAN,
    CIFAR10_STD,
    build_cifar10_loaders,
    encode_spikes,
)
from model import CifarResNet, ConvLifModelSTDP, LifModelSTDP, SpikingConvLifModelSTDP
from utils import load_checkpoint, load_stdp_params, resolve_device, save_checkpoint, set_seed


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
    parser.add_argument("--conv-blocks-per-stage", type=int, default=1)
    parser.add_argument("--resnet-width", type=int, default=64)
    parser.add_argument("--resnet-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--readout",
        type=str,
        default="spike",
        choices=["spike", "membrane", "hybrid"],
    )
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--stdp-current-scale", type=float, default=None)
    parser.add_argument("--auto-augment", action="store_true")
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--distill-alpha", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=4.0)
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
            conv_blocks_per_stage=args.conv_blocks_per_stage,
            stdp_current_scale=args.stdp_current_scale,
            dropout=args.dropout,
            readout=args.readout,
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


def make_teacher_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})
    model_type = train_args.get("model_type")
    if model_type != "cifar-resnet":
        raise ValueError(
            f"Teacher checkpoint must be cifar-resnet, got {model_type!r}: {checkpoint_path}"
        )
    teacher = CifarResNet(
        num_classes=CIFAR10_CLASSES,
        width=train_args.get("resnet_width", 64),
        blocks_per_stage=train_args.get("resnet_blocks", 3),
        dropout=train_args.get("dropout", 0.0),
    ).to(device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


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


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    batch_size, _, height, width = images.shape
    indices = torch.randperm(batch_size, device=images.device)

    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    center_x = int(torch.randint(width, (1,), device=images.device).item())
    center_y = int(torch.randint(height, (1,), device=images.device).item())

    x1 = max(center_x - cut_w // 2, 0)
    y1 = max(center_y - cut_h // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    y2 = min(center_y + cut_h // 2, height)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed_images, labels, labels[indices], lam


def augment_labels_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if args.cutmix_alpha > 0:
        return cutmix_batch(images, labels, args.cutmix_alpha)
    if args.mixup_alpha > 0:
        return mixup_batch(images, labels, args.mixup_alpha)
    return images, labels, labels, 1.0


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def normalize_for_teacher(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
        temperature * temperature
    )


def train_epoch(
    model: nn.Module,
    teacher_model: nn.Module | None,
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
        images, labels_a, labels_b, lam = augment_labels_batch(images, labels, args)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = forward_model(model, images, args)
            hard_loss = mixup_loss(criterion, logits, labels_a, labels_b, lam)
            if teacher_model is not None and args.distill_alpha > 0:
                with torch.no_grad():
                    teacher_logits = teacher_model(normalize_for_teacher(images))
                soft_loss = distillation_loss(
                    logits,
                    teacher_logits,
                    args.distill_temperature,
                )
                loss = (1.0 - args.distill_alpha) * hard_loss + args.distill_alpha * soft_loss
            else:
                loss = hard_loss
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
    teacher_model = (
        make_teacher_model(args.teacher_checkpoint, device)
        if args.teacher_checkpoint is not None and args.distill_alpha > 0
        else None
    )
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
    print(f"readout={args.readout if args.model_type == 'spiking-conv-stdp' else 'none'}")
    print(f"mixup_alpha={args.mixup_alpha}")
    print(f"cutmix_alpha={args.cutmix_alpha}")
    print(f"auto_augment={args.auto_augment}")
    print(f"random_erasing={args.random_erasing}")
    print(f"teacher_checkpoint={args.teacher_checkpoint}")
    print(f"distill_alpha={args.distill_alpha}")
    print(f"distill_temperature={args.distill_temperature}")
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
            teacher_model,
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
