from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


NASA_BEARING_FAULT_PARTS = (
    "normal",
    "inner_ring",
    "outer_ring",
    "rolling_element",
    "cage",
    "unknown_fault",
)

DEFAULT_FAULT_MAP = "B3=inner_ring,B4=rolling_element"


@dataclass(frozen=True)
class NasaBearingNormalizer:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "NasaBearingNormalizer":
        q25 = np.percentile(features, 25, axis=0)
        q50 = np.percentile(features, 50, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        scale = q75 - q25
        scale[scale < 1e-6] = 1.0
        return cls(center=q50.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        z = (features.astype(np.float32) - self.center) / self.scale
        # A smooth bounded transform is useful for rate coding into spikes.
        z = np.clip(z, -12.0, 12.0)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


class NasaBearingFeatureDataset(Dataset):
    """NASA bearing engineered-feature dataset.

    The current local data has two feature files:
    - features_good_named.csv
    - features_fault_named.csv

    Each feature row contains B1-B4. This dataset converts each row into four
    per-bearing samples, because component fault labels belong to a specific
    bearing rather than to the whole four-bearing row.
    """

    def __init__(
        self,
        features: np.ndarray,
        health_labels: np.ndarray,
        fault_part_labels: np.ndarray,
        health_scores: np.ndarray,
        bearing_indices: np.ndarray,
        source_is_fault: np.ndarray,
        source_positions: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.health_labels = torch.from_numpy(health_labels.astype(np.int64))
        self.fault_part_labels = torch.from_numpy(fault_part_labels.astype(np.int64))
        self.health_scores = torch.from_numpy(health_scores.astype(np.float32))
        self.bearing_indices = torch.from_numpy(bearing_indices.astype(np.int64))
        self.source_is_fault = torch.from_numpy(source_is_fault.astype(np.int64))
        self.source_positions = torch.from_numpy(source_positions.astype(np.int64))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "health_label": self.health_labels[index],
            "fault_part_label": self.fault_part_labels[index],
            "health_score": self.health_scores[index],
            "bearing_index": self.bearing_indices[index],
            "source_is_fault": self.source_is_fault[index],
            "source_position": self.source_positions[index],
        }


def _read_feature_file(path: Path, source_label: str, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"NASA bearing feature file not found: {path}")
    if limit is not None and limit <= 0:
        limit = None
    df = pd.read_csv(path)
    if limit is not None and len(df) > limit:
        indices = np.linspace(0, len(df) - 1, num=limit, dtype=np.int64)
        df = df.iloc[indices].reset_index(drop=True)
    df["source_label"] = source_label
    return df


def parse_fault_map(fault_map: str) -> dict[int, str]:
    if not fault_map.strip():
        return {}
    part_names = set(NASA_BEARING_FAULT_PARTS)
    mapping: dict[int, str] = {}
    for item in fault_map.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid fault map item {item!r}; expected B3=inner_ring")
        bearing_name, part_name = [value.strip().lower() for value in item.split("=", 1)]
        if not bearing_name.startswith("b") or not bearing_name[1:].isdigit():
            raise ValueError(f"Invalid bearing name {bearing_name!r}; expected B1, B2, B3, or B4")
        if part_name not in part_names or part_name == "normal":
            raise ValueError(
                f"Invalid fault part {part_name!r}; expected one of "
                f"{', '.join(name for name in NASA_BEARING_FAULT_PARTS if name != 'normal')}"
            )
        mapping[int(bearing_name[1:])] = part_name
    return mapping


def _anomaly_health_scores_for_bearing(
    df: pd.DataFrame,
    bearing_index: int,
) -> np.ndarray:
    """Build feature-aligned pseudo health scores from robust anomaly distance."""
    scores = np.ones(len(df), dtype=np.float32)
    fault_mask = df["source_label"].to_numpy() == "fault"
    if not fault_mask.any():
        return scores

    columns = _bearing_feature_columns(df, bearing_index)
    values = df[columns].to_numpy(dtype=np.float32)
    good_values = values[~fault_mask]
    if len(good_values) == 0:
        return scores

    center = np.median(good_values, axis=0)
    q25 = np.percentile(good_values, 25, axis=0)
    q75 = np.percentile(good_values, 75, axis=0)
    scale = q75 - q25
    scale[scale < 1e-6] = 1.0

    z = np.abs((values - center) / scale)
    top_k = min(5, z.shape[1])
    anomaly = np.partition(z, -top_k, axis=1)[:, -top_k:].mean(axis=1)

    good_reference = anomaly[~fault_mask]
    fault_reference = anomaly[fault_mask]
    normal_level = float(np.percentile(good_reference, 95))
    fault_level = float(np.percentile(fault_reference, 95))
    span = max(fault_level - normal_level, 1e-6)
    severity = np.clip((anomaly - normal_level) / span, 0.0, 1.0)
    scores[fault_mask] = 1.0 - severity[fault_mask]
    return scores.astype(np.float32)


def _health_scores_for_bearing(
    df: pd.DataFrame,
    bearing_index: int,
    fault_map: dict[int, str],
    health_label_mode: str,
) -> np.ndarray:
    per_bearing_health_col = f"B{bearing_index}__health_score"
    per_bearing_health_alias = f"B{bearing_index}__health"
    if per_bearing_health_col in df.columns:
        scores = df[per_bearing_health_col].astype(np.float32).clip(0.0, 1.0).to_numpy()
    elif per_bearing_health_alias in df.columns:
        scores = df[per_bearing_health_alias].astype(np.float32).clip(0.0, 1.0).to_numpy()
    elif "health_score" in df.columns:
        scores = df["health_score"].astype(np.float32).clip(0.0, 1.0).to_numpy()
    elif "health" in df.columns:
        scores = df["health"].astype(np.float32).clip(0.0, 1.0).to_numpy()
    elif health_label_mode == "anomaly":
        if bearing_index in fault_map:
            scores = _anomaly_health_scores_for_bearing(df, bearing_index)
        else:
            scores = np.ones(len(df), dtype=np.float32)
    elif health_label_mode == "source":
        scores = np.ones(len(df), dtype=np.float32)
        if bearing_index in fault_map:
            scores[df["source_label"].to_numpy() == "fault"] = 0.0
    elif health_label_mode == "time":
        if bearing_index in fault_map:
            scores = np.ones(len(df), dtype=np.float32)
            fault_mask = df["source_label"].to_numpy() == "fault"
            fault_positions = df.loc[fault_mask, "source_position"].to_numpy(dtype=np.float32)
            if len(fault_positions) > 1:
                fault_span = max(float(fault_positions.max() - fault_positions.min()), 1.0)
                scores[fault_mask] = 1.0 - (fault_positions - fault_positions.min()) / fault_span
            elif len(fault_positions) == 1:
                scores[fault_mask] = 0.0
        else:
            scores = np.ones(len(df), dtype=np.float32)
    else:
        raise ValueError(f"Unknown health_label_mode: {health_label_mode}")
    return scores.astype(np.float32)


def _health_labels(scores: np.ndarray, health_bins: int) -> np.ndarray:
    labels = np.rint((1.0 - scores) * (health_bins - 1)).astype(np.int64)
    return np.clip(labels, 0, health_bins - 1)


def _fault_part_labels_for_bearing(
    df: pd.DataFrame,
    bearing_index: int,
    fault_map: dict[int, str],
) -> np.ndarray:
    part_to_index = {name: idx for idx, name in enumerate(NASA_BEARING_FAULT_PARTS)}
    per_bearing_fault_col = f"B{bearing_index}__fault_part"
    if per_bearing_fault_col in df.columns:
        normalized = df[per_bearing_fault_col].fillna("normal").astype(str).str.strip().str.lower()
        return normalized.map(
            lambda value: part_to_index.get(value, part_to_index["unknown_fault"])
        ).to_numpy(dtype=np.int64)
    if "fault_part" in df.columns:
        normalized = df["fault_part"].fillna("normal").astype(str).str.strip().str.lower()
        return normalized.map(
            lambda value: part_to_index.get(value, part_to_index["unknown_fault"])
        ).to_numpy(dtype=np.int64)

    labels = np.full(len(df), part_to_index["normal"], dtype=np.int64)
    fault_mask = df["source_label"].to_numpy() == "fault"
    if bearing_index in fault_map:
        labels[fault_mask] = part_to_index[fault_map[bearing_index]]
    return labels


def _bearing_feature_columns(df: pd.DataFrame, bearing_index: int) -> list[str]:
    prefix = f"B{bearing_index}__"
    columns = [
        column
        for column in df.columns
        if column.startswith(prefix)
        and not column.endswith("__health")
        and not column.endswith("__health_score")
        and not column.endswith("__fault_part")
    ]
    if not columns:
        raise ValueError(f"No feature columns found for bearing B{bearing_index}")
    return columns


def _row_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in df.columns
        if "__" in column
        and column.split("__", 1)[0].startswith("B")
        and not column.endswith("__health")
        and not column.endswith("__health_score")
        and not column.endswith("__fault_part")
    ]
    if not columns:
        raise ValueError("No row-level bearing feature columns found")
    return columns


def _build_per_bearing_arrays(
    df: pd.DataFrame,
    health_bins: int,
    fault_map: dict[int, str],
    health_label_mode: str,
    include_bearing_id: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features_by_bearing = []
    health_labels_by_bearing = []
    fault_labels_by_bearing = []
    health_scores_by_bearing = []
    bearing_indices_by_bearing = []
    source_is_fault_by_bearing = []
    source_positions_by_bearing = []

    for bearing_index in range(1, 5):
        columns = _bearing_feature_columns(df, bearing_index)
        bearing_features = df[columns].to_numpy(dtype=np.float32)
        if include_bearing_id:
            bearing_id = np.zeros((len(df), 4), dtype=np.float32)
            bearing_id[:, bearing_index - 1] = 1.0
            bearing_features = np.concatenate([bearing_features, bearing_id], axis=1)
        features_by_bearing.append(bearing_features)
        scores = _health_scores_for_bearing(
            df=df,
            bearing_index=bearing_index,
            fault_map=fault_map,
            health_label_mode=health_label_mode,
        )
        health_scores_by_bearing.append(scores)
        health_labels_by_bearing.append(_health_labels(scores, health_bins))
        fault_labels_by_bearing.append(
            _fault_part_labels_for_bearing(
                df=df,
                bearing_index=bearing_index,
                fault_map=fault_map,
            )
        )
        bearing_indices_by_bearing.append(
            np.full(len(df), bearing_index, dtype=np.int64)
        )
        source_is_fault_by_bearing.append(
            (df["source_label"].to_numpy() == "fault").astype(np.int64)
        )
        source_positions_by_bearing.append(
            df["source_position"].to_numpy(dtype=np.int64)
        )

    return (
        np.concatenate(features_by_bearing, axis=0),
        np.concatenate(health_labels_by_bearing, axis=0),
        np.concatenate(fault_labels_by_bearing, axis=0),
        np.concatenate(health_scores_by_bearing, axis=0),
        np.concatenate(bearing_indices_by_bearing, axis=0),
        np.concatenate(source_is_fault_by_bearing, axis=0),
        np.concatenate(source_positions_by_bearing, axis=0),
    )


def _row_anomaly_health_scores(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    scores = np.ones(len(df), dtype=np.float32)
    fault_mask = df["source_label"].to_numpy() == "fault"
    if not fault_mask.any():
        return scores

    values = df[columns].to_numpy(dtype=np.float32)
    good_values = values[~fault_mask]
    if len(good_values) == 0:
        return scores

    center = np.median(good_values, axis=0)
    q25 = np.percentile(good_values, 25, axis=0)
    q75 = np.percentile(good_values, 75, axis=0)
    scale = q75 - q25
    scale[scale < 1e-6] = 1.0

    z = np.abs((values - center) / scale)
    top_k = min(10, z.shape[1])
    anomaly = np.partition(z, -top_k, axis=1)[:, -top_k:].mean(axis=1)

    good_reference = anomaly[~fault_mask]
    fault_reference = anomaly[fault_mask]
    normal_level = float(np.percentile(good_reference, 95))
    fault_level = float(np.percentile(fault_reference, 95))
    span = max(fault_level - normal_level, 1e-6)
    severity = np.clip((anomaly - normal_level) / span, 0.0, 1.0)
    scores[fault_mask] = 1.0 - severity[fault_mask]
    return scores.astype(np.float32)


def _health_scores_for_row(
    df: pd.DataFrame,
    columns: list[str],
    health_label_mode: str,
) -> np.ndarray:
    if "health_score" in df.columns:
        scores = df["health_score"].astype(np.float32).clip(0.0, 1.0).to_numpy()
    elif "health" in df.columns:
        scores = df["health"].astype(np.float32).clip(0.0, 1.0).to_numpy()
    elif health_label_mode == "anomaly":
        scores = _row_anomaly_health_scores(df, columns)
    elif health_label_mode == "source":
        scores = np.ones(len(df), dtype=np.float32)
        scores[df["source_label"].to_numpy() == "fault"] = 0.0
    elif health_label_mode == "time":
        scores = np.ones(len(df), dtype=np.float32)
        fault_mask = df["source_label"].to_numpy() == "fault"
        fault_positions = df.loc[fault_mask, "source_position"].to_numpy(dtype=np.float32)
        if len(fault_positions) > 1:
            fault_span = max(float(fault_positions.max() - fault_positions.min()), 1.0)
            scores[fault_mask] = 1.0 - (fault_positions - fault_positions.min()) / fault_span
        elif len(fault_positions) == 1:
            scores[fault_mask] = 0.0
    else:
        raise ValueError(f"Unknown health_label_mode: {health_label_mode}")
    return scores.astype(np.float32)


def _fault_part_labels_for_row(df: pd.DataFrame) -> np.ndarray:
    part_to_index = {name: idx for idx, name in enumerate(NASA_BEARING_FAULT_PARTS)}
    if "fault_part" in df.columns:
        normalized = df["fault_part"].fillna("normal").astype(str).str.strip().str.lower()
        return normalized.map(
            lambda value: part_to_index.get(value, part_to_index["unknown_fault"])
        ).to_numpy(dtype=np.int64)

    labels = np.full(len(df), part_to_index["normal"], dtype=np.int64)
    fault_mask = df["source_label"].to_numpy() == "fault"
    labels[fault_mask] = part_to_index["unknown_fault"]
    return labels


def _build_row_arrays(
    df: pd.DataFrame,
    health_bins: int,
    health_label_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = _row_feature_columns(df)
    features = df[columns].to_numpy(dtype=np.float32)
    scores = _health_scores_for_row(
        df=df,
        columns=columns,
        health_label_mode=health_label_mode,
    )
    return (
        features,
        _health_labels(scores, health_bins),
        _fault_part_labels_for_row(df),
        scores,
        np.zeros(len(df), dtype=np.int64),
        (df["source_label"].to_numpy() == "fault").astype(np.int64),
        df["source_position"].to_numpy(dtype=np.int64),
    )


def _stratified_split_indices(
    health_labels: np.ndarray,
    fault_labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    strata = np.stack([health_labels, fault_labels], axis=1)
    for health_label, fault_label in np.unique(strata, axis=0):
        indices = np.where((health_labels == health_label) & (fault_labels == fault_label))[0]
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction))) if len(indices) > 1 else 0
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_nasa_bearing_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    health_bins: int = 11,
    health_label_mode: str = "time",
    fault_map: str = DEFAULT_FAULT_MAP,
    include_bearing_id: bool = True,
    sample_layout: str = "per_bearing",
    val_fraction: float = 0.2,
    limit_good: int | None = None,
    limit_fault: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    data_root = Path(data_root)
    good_df = _read_feature_file(data_root / "features_good_named.csv", "good", limit_good)
    fault_df = _read_feature_file(data_root / "features_fault_named.csv", "fault", limit_fault)
    good_df["source_position"] = np.arange(len(good_df), dtype=np.int64)
    fault_df["source_position"] = np.arange(len(fault_df), dtype=np.int64)
    df = pd.concat([good_df, fault_df], ignore_index=True)
    df["window_position"] = np.arange(len(df), dtype=np.int64)
    fault_map_by_bearing = parse_fault_map(fault_map)
    if sample_layout == "per_bearing":
        (
            features_raw,
            health_labels,
            fault_labels,
            health_scores,
            bearing_indices,
            source_is_fault,
            source_positions,
        ) = _build_per_bearing_arrays(
            df=df,
            health_bins=health_bins,
            fault_map=fault_map_by_bearing,
            health_label_mode=health_label_mode,
            include_bearing_id=include_bearing_id,
        )
    elif sample_layout == "row":
        (
            features_raw,
            health_labels,
            fault_labels,
            health_scores,
            bearing_indices,
            source_is_fault,
            source_positions,
        ) = _build_row_arrays(
            df=df,
            health_bins=health_bins,
            health_label_mode=health_label_mode,
        )
    else:
        raise ValueError(f"Unknown sample_layout: {sample_layout}")

    train_indices, val_indices = _stratified_split_indices(
        health_labels=health_labels,
        fault_labels=fault_labels,
        val_fraction=val_fraction,
        seed=seed,
    )
    if include_bearing_id and sample_layout == "per_bearing":
        feature_values = features_raw[:, :-4]
        bearing_id_values = features_raw[:, -4:]
        normalizer = NasaBearingNormalizer.fit(feature_values[np.asarray(train_indices)])
        features = np.concatenate(
            [normalizer.transform(feature_values), bearing_id_values.astype(np.float32)],
            axis=1,
        )
    else:
        normalizer = NasaBearingNormalizer.fit(features_raw[np.asarray(train_indices)])
        features = normalizer.transform(features_raw)

    dataset = NasaBearingFeatureDataset(
        features=features,
        health_labels=health_labels,
        fault_part_labels=fault_labels,
        health_scores=health_scores,
        bearing_indices=bearing_indices,
        source_is_fault=source_is_fault,
        source_positions=source_positions,
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, int(features_raw.shape[1])
