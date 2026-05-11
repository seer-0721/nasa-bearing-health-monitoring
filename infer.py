from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data import CIFAR10_CLASSES, CIFAR10_INPUT_SIZE, poisson_encode
from model import ConvLifModelSTDP, LifModelSTDP
from utils import load_checkpoint, load_stdp_params, resolve_device


CIFAR10_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-10 inference from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    parser.add_argument("--stdp-params-file", type=str, default="stdp_params.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})

    stdp_params = load_stdp_params(args.stdp_params_file)
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

    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    seen = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(poisson_encode(images, args.time_steps))
            preds = logits.argmax(dim=1)

            for pred, target in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                print(f"pred={CIFAR10_LABELS[pred]} target={CIFAR10_LABELS[target]}")
                correct += int(pred == target)
                seen += 1
                if seen >= args.limit:
                    print(f"shown_acc={correct / seen:.4f}")
                    return


if __name__ == "__main__":
    main()
