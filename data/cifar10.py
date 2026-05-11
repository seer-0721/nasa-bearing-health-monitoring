from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_INPUT_SIZE = 3 * 32 * 32
CIFAR10_CLASSES = 10


def build_cifar10_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    download: bool,
    augment: bool = False,
    limit_train: int | None = None,
    limit_test: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    data_root = Path(data_root)

    try:
        train_set = datasets.CIFAR10(
            root=str(data_root),
            train=True,
            transform=train_transform,
            download=download,
        )
        test_set = datasets.CIFAR10(
            root=str(data_root),
            train=False,
            transform=test_transform,
            download=download,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"CIFAR-10 was not found under {data_root}. "
            "Put the dataset there, or rerun with --download."
        ) from exc

    if limit_train is not None:
        train_set = Subset(train_set, range(min(limit_train, len(train_set))))
    if limit_test is not None:
        test_set = Subset(test_set, range(min(limit_test, len(test_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
