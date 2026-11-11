from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data import DEFAULT_FAULT_MAP, build_nasa_bearing_loaders


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_root: str
    sample_layout: str
    health_label_mode: str
    fault_map: str = DEFAULT_FAULT_MAP
    include_bearing_id: bool = True


DATASET_CONFIGS = {
    "nasa": DatasetConfig(
        name="nasa",
        data_root="data/NasaBearingDataset",
        sample_layout="per_bearing",
        health_label_mode="anomaly",
        fault_map="B3=inner_ring,B4=rolling_element",
        include_bearing_id=True,
    ),
    "cwru": DatasetConfig(
        name="cwru",
        data_root="data/CWRU/converted_full",
        sample_layout="row",
        health_label_mode="source",
    ),
    "paderborn": DatasetConfig(
        name="paderborn",
        data_root="data/Paderborn/converted",
        sample_layout="row",
        health_label_mode="source",
    ),
}


def _health_bins(scores: np.ndarray) -> np.ndarray:
    bins = np.digitize(scores, [0.2, 0.4, 0.6, 0.8]).astype(np.int64)
    return bins


def collect_loader(loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    health_scores = []
    fault = []
    for batch in loader:
        features.append(batch["features"].detach().cpu().numpy())
        health_scores.append(batch["health_score"].detach().cpu().numpy())
        fault.append(batch["fault_part_label"].detach().cpu().numpy().astype(np.int64))
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.concatenate(health_scores, axis=0).astype(np.float32),
        np.concatenate(fault, axis=0).astype(np.int64),
    )


def stratified_limit(
    x: np.ndarray,
    health: np.ndarray,
    fault: np.ndarray,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if limit <= 0 or len(x) <= limit:
        return x, health, fault
    rng = np.random.default_rng(seed)
    health_bins = _health_bins(health)
    selected = []
    strata = np.stack([health_bins, fault], axis=1)
    unique = np.unique(strata, axis=0)
    base = max(limit // max(len(unique), 1), 1)
    for item in unique:
        indices = np.where((strata == item).all(axis=1))[0]
        take = min(len(indices), base)
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(x)), np.asarray(selected), assume_unique=False)
        extra = min(limit - len(selected), len(remaining))
        if extra > 0:
            selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    selected = np.asarray(selected[:limit], dtype=np.int64)
    rng.shuffle(selected)
    return x[selected], health[selected], fault[selected]


def load_dataset(
    config: DatasetConfig,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
):
    train_loader, val_loader, input_size = build_nasa_bearing_loaders(
        data_root=config.data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        health_bins=11,
        health_label_mode=config.health_label_mode,
        fault_map=config.fault_map,
        include_bearing_id=config.include_bearing_id,
        sample_layout=config.sample_layout,
        val_fraction=0.2,
        limit_good=0,
        limit_fault=0,
        seed=seed,
    )
    torch.set_grad_enabled(False)
    train_x, train_health, train_fault = collect_loader(train_loader)
    val_x, val_health, val_fault = collect_loader(val_loader)
    train_x, train_health, train_fault = stratified_limit(
        train_x,
        train_health,
        train_fault,
        max_train_samples,
        seed,
    )
    val_x, val_health, val_fault = stratified_limit(
        val_x,
        val_health,
        val_fault,
        max_val_samples,
        seed + 1,
    )
    return {
        "input_size": input_size,
        "train_x": train_x,
        "train_health": train_health,
        "train_fault": train_fault,
        "val_x": val_x,
        "val_health": val_health,
        "val_fault": val_fault,
    }
