from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew


FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "mad",
    "p2p",
    "skew",
    "kurt",
    "crest",
    "imp",
    "spec_centroid",
    "spec_bandwidth",
    "rolloff_95",
    "spec_flatness",
    "spec_entropy",
    "dom_freq",
    "peak_ratio",
    "band_q1",
    "band_q2",
    "band_q3",
    "band_q4",
]

XJTU_FAULT_MAP = {
    "Bearing1_1": "outer_ring",
    "Bearing1_2": "outer_ring",
    "Bearing1_3": "outer_ring",
    "Bearing1_4": "cage",
    "Bearing1_5": "unknown_fault",
    "Bearing2_1": "inner_ring",
    "Bearing2_2": "outer_ring",
    "Bearing2_3": "cage",
    "Bearing2_4": "outer_ring",
    "Bearing2_5": "outer_ring",
    "Bearing3_1": "outer_ring",
    "Bearing3_2": "unknown_fault",
    "Bearing3_3": "inner_ring",
    "Bearing3_4": "inner_ring",
    "Bearing3_5": "outer_ring",
}

def _add_cwru_entries(
    manifest: dict[str, tuple[str, int, float]],
    ids_by_load: list[int],
    fault_part: str,
    diameter: float,
) -> None:
    for load_hp, file_id in enumerate(ids_by_load):
        manifest[str(file_id)] = (fault_part, load_hp, diameter)


def build_cwru_manifest() -> dict[str, tuple[str, int, float]]:
    manifest: dict[str, tuple[str, int, float]] = {
        "97": ("normal", 0, 0.0),
        "98": ("normal", 1, 0.0),
        "99": ("normal", 2, 0.0),
        "100": ("normal", 3, 0.0),
    }

    # 12k drive-end bearing fault data.
    _add_cwru_entries(manifest, [105, 106, 107, 108], "inner_ring", 0.007)
    _add_cwru_entries(manifest, [118, 119, 120, 121], "rolling_element", 0.007)
    _add_cwru_entries(manifest, [130, 131, 132, 133], "outer_ring", 0.007)
    _add_cwru_entries(manifest, [144, 145, 146, 147], "outer_ring", 0.007)
    _add_cwru_entries(manifest, [156, 158, 159, 160], "outer_ring", 0.007)
    _add_cwru_entries(manifest, [169, 170, 171, 172], "inner_ring", 0.014)
    _add_cwru_entries(manifest, [185, 186, 187, 188], "rolling_element", 0.014)
    _add_cwru_entries(manifest, [197, 198, 199, 200], "outer_ring", 0.014)
    _add_cwru_entries(manifest, [209, 210, 211, 212], "inner_ring", 0.021)
    _add_cwru_entries(manifest, [222, 223, 224, 225], "rolling_element", 0.021)
    _add_cwru_entries(manifest, [234, 235, 236, 237], "outer_ring", 0.021)
    _add_cwru_entries(manifest, [246, 247, 248, 249], "outer_ring", 0.021)
    _add_cwru_entries(manifest, [258, 259, 260, 261], "outer_ring", 0.021)
    _add_cwru_entries(manifest, [3001, 3002, 3003, 3004], "inner_ring", 0.028)
    _add_cwru_entries(manifest, [3005, 3006, 3007, 3008], "rolling_element", 0.028)

    # 48k drive-end bearing fault data.
    _add_cwru_entries(manifest, [109, 110, 111, 112], "inner_ring", 0.007)
    _add_cwru_entries(manifest, [122, 123, 124, 125], "rolling_element", 0.007)
    _add_cwru_entries(manifest, [135, 136, 137, 138], "outer_ring", 0.007)
    _add_cwru_entries(manifest, [148, 149, 150, 151], "outer_ring", 0.007)
    _add_cwru_entries(manifest, [161, 162, 163, 164], "outer_ring", 0.007)
    _add_cwru_entries(manifest, [174, 175, 176, 177], "inner_ring", 0.014)
    _add_cwru_entries(manifest, [189, 190, 191, 192], "rolling_element", 0.014)
    _add_cwru_entries(manifest, [201, 202, 203, 204], "outer_ring", 0.014)
    _add_cwru_entries(manifest, [213, 214, 215, 217], "inner_ring", 0.021)
    _add_cwru_entries(manifest, [226, 227, 228, 229], "rolling_element", 0.021)
    _add_cwru_entries(manifest, [238, 239, 240, 241], "outer_ring", 0.021)
    _add_cwru_entries(manifest, [250, 251, 252, 253], "outer_ring", 0.021)
    _add_cwru_entries(manifest, [262, 263, 264, 265], "outer_ring", 0.021)

    # 12k fan-end bearing fault data.
    _add_cwru_entries(manifest, [278, 279, 280, 281], "inner_ring", 0.007)
    _add_cwru_entries(manifest, [282, 283, 284, 285], "rolling_element", 0.007)
    _add_cwru_entries(manifest, [294, 295, 296, 297], "outer_ring", 0.007)
    _add_cwru_entries(manifest, [298, 299, 300, 301], "outer_ring", 0.007)
    _add_cwru_entries(manifest, [302, 305, 306, 307], "outer_ring", 0.007)
    _add_cwru_entries(manifest, [274, 275, 276, 277], "inner_ring", 0.014)
    _add_cwru_entries(manifest, [286, 287, 288, 289], "rolling_element", 0.014)
    manifest["313"] = ("outer_ring", 0, 0.014)
    _add_cwru_entries(manifest, [310, 309, 311, 312], "outer_ring", 0.014)
    _add_cwru_entries(manifest, [270, 271, 272, 273], "inner_ring", 0.021)
    _add_cwru_entries(manifest, [290, 291, 292, 293], "rolling_element", 0.021)
    manifest["315"] = ("outer_ring", 0, 0.021)
    manifest["316"] = ("outer_ring", 1, 0.021)
    manifest["317"] = ("outer_ring", 2, 0.021)
    manifest["318"] = ("outer_ring", 3, 0.021)
    return manifest


CWRU_MANIFEST = build_cwru_manifest()


def signal_features(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        x = np.zeros(1, dtype=np.float64)
    abs_x = np.abs(x)
    mean_abs = float(np.mean(abs_x))
    rms = float(np.sqrt(np.mean(x * x)))
    p2p = float(np.ptp(x))
    spectrum = np.abs(np.fft.rfft(x - np.mean(x)))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    power = spectrum * spectrum
    power_sum = float(power.sum())
    if power_sum <= 1e-12:
        norm_power = np.ones_like(power) / max(len(power), 1)
    else:
        norm_power = power / power_sum
    centroid = float((freqs * norm_power).sum())
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * norm_power).sum()))
    cumulative = np.cumsum(norm_power)
    rolloff_index = int(np.searchsorted(cumulative, 0.95, side="left"))
    rolloff_index = min(rolloff_index, len(freqs) - 1)
    entropy = float(-(norm_power * np.log(norm_power + 1e-12)).sum())
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))
    dom_index = int(np.argmax(power)) if len(power) else 0
    peak_ratio = float(power[dom_index] / (power_sum + 1e-12))
    bands = np.array_split(power, 4)
    band_values = [float(band.sum() / (power_sum + 1e-12)) for band in bands]
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": rms,
        "mad": float(np.mean(np.abs(x - np.mean(x)))),
        "p2p": p2p,
        "skew": float(skew(x, bias=False)) if len(x) > 2 else 0.0,
        "kurt": float(kurtosis(x, fisher=False, bias=False)) if len(x) > 3 else 0.0,
        "crest": float(np.max(abs_x) / (rms + 1e-12)),
        "imp": float(np.max(abs_x) / (mean_abs + 1e-12)),
        "spec_centroid": centroid,
        "spec_bandwidth": bandwidth,
        "rolloff_95": float(freqs[rolloff_index]),
        "spec_flatness": flatness,
        "spec_entropy": entropy,
        "dom_freq": float(freqs[dom_index]) if len(freqs) else 0.0,
        "peak_ratio": peak_ratio,
        "band_q1": band_values[0],
        "band_q2": band_values[1],
        "band_q3": band_values[2],
        "band_q4": band_values[3],
    }


def add_channel_features(row: dict[str, object], channel: int, signal: np.ndarray, sample_rate: float) -> None:
    features = signal_features(signal, sample_rate)
    for name in FEATURE_NAMES:
        row[f"B{channel}__{name}"] = features[name]


def repeat_channels(row: dict[str, object], source_channels: int) -> None:
    if source_channels == 1:
        for name in FEATURE_NAMES:
            row[f"B2__{name}"] = row[f"B1__{name}"]
            row[f"B3__{name}"] = row[f"B1__{name}"]
            row[f"B4__{name}"] = row[f"B1__{name}"]
    elif source_channels == 2:
        for name in FEATURE_NAMES:
            row[f"B3__{name}"] = row[f"B1__{name}"]
            row[f"B4__{name}"] = row[f"B2__{name}"]
    elif source_channels == 3:
        for name in FEATURE_NAMES:
            row[f"B4__{name}"] = row[f"B1__{name}"]


def anomaly_scores(feature_frame: pd.DataFrame, baseline_count: int) -> np.ndarray:
    feature_cols = [column for column in feature_frame.columns if "__" in column]
    values = feature_frame[feature_cols].to_numpy(dtype=np.float64)
    baseline = values[:baseline_count]
    center = np.median(baseline, axis=0)
    q25 = np.percentile(baseline, 25, axis=0)
    q75 = np.percentile(baseline, 75, axis=0)
    scale = q75 - q25
    scale[scale < 1e-6] = 1.0
    z = np.abs((values - center) / scale)
    top_k = min(8, z.shape[1])
    distance = np.partition(z, -top_k, axis=1)[:, -top_k:].mean(axis=1)
    normal_level = float(np.percentile(distance[:baseline_count], 95))
    fault_level = float(np.percentile(distance, 95))
    severity = np.clip((distance - normal_level) / max(fault_level - normal_level, 1e-6), 0.0, 1.0)
    return (1.0 - severity).astype(np.float32)


def time_to_failure_scores(total_count: int, baseline_count: int) -> np.ndarray:
    scores = np.ones(total_count, dtype=np.float32)
    fault_count = total_count - baseline_count
    if fault_count <= 0:
        return scores
    if fault_count == 1:
        scores[baseline_count:] = 0.0
    else:
        scores[baseline_count:] = np.linspace(1.0, 0.0, fault_count, dtype=np.float32)
    return scores


def numeric_csv_key(path: Path) -> int:
    return int(path.stem) if path.stem.isdigit() else 0


def convert_xjtu(args: argparse.Namespace) -> None:
    root = Path(args.input_root)
    if (root / "XJTU-SY_Bearing_Datasets").exists():
        root = root / "XJTU-SY_Bearing_Datasets"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    good_rows: list[dict[str, object]] = []
    fault_rows: list[dict[str, object]] = []

    bearing_dirs = sorted(path for path in root.glob("*/*") if path.is_dir())
    for bearing_dir in bearing_dirs:
        files = sorted(bearing_dir.glob("*.csv"), key=numeric_csv_key)
        if args.max_files_per_bearing > 0:
            files = files[: args.max_files_per_bearing]
        if not files:
            continue
        bearing_name = bearing_dir.name
        fault_part = XJTU_FAULT_MAP.get(bearing_name, "unknown_fault")
        rows = []
        for position, csv_path in enumerate(files):
            data = pd.read_csv(csv_path)
            horizontal = data.iloc[:, 0].to_numpy(dtype=np.float64)
            vertical = data.iloc[:, 1].to_numpy(dtype=np.float64)
            row: dict[str, object] = {
                "dataset": "XJTU-SY",
                "condition": bearing_dir.parent.name,
                "bearing": bearing_name,
                "source_file": str(csv_path.relative_to(root)),
                "source_position": position,
                "fault_part": fault_part,
            }
            add_channel_features(row, 1, horizontal, args.sample_rate)
            add_channel_features(row, 2, vertical, args.sample_rate)
            repeat_channels(row, source_channels=2)
            rows.append(row)

        frame = pd.DataFrame(rows)
        baseline_count = max(args.min_baseline_files, int(round(len(frame) * args.baseline_fraction)))
        baseline_count = min(max(baseline_count, 1), len(frame))
        if args.health_mode == "time":
            frame["health_score"] = time_to_failure_scores(len(frame), baseline_count)
        elif args.health_mode == "anomaly":
            frame["health_score"] = anomaly_scores(frame, baseline_count)
        else:
            time_scores = time_to_failure_scores(len(frame), baseline_count)
            anomaly_score_values = anomaly_scores(frame, baseline_count)
            frame["health_score"] = np.minimum(time_scores, anomaly_score_values)
        frame["health"] = frame["health_score"]
        good_frame = frame.iloc[:baseline_count].copy()
        good_frame["fault_part"] = "normal"
        good_frame["health_score"] = 1.0
        good_frame["health"] = 1.0
        good_rows.extend(good_frame.to_dict("records"))
        fault_rows.extend(frame.iloc[baseline_count:].to_dict("records"))

    pd.DataFrame(good_rows).to_csv(output_root / "features_good_named.csv", index=False)
    pd.DataFrame(fault_rows).to_csv(output_root / "features_fault_named.csv", index=False)
    print(f"wrote {len(good_rows)} good rows and {len(fault_rows)} fault rows to {output_root}")


def load_cwru_signal(mat_path: Path, suffix: str) -> np.ndarray | None:
    data = loadmat(mat_path)
    for key, value in data.items():
        if key.endswith(suffix):
            return np.asarray(value).reshape(-1)
    return None


def cwru_file_id(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not infer CWRU file id from {path}")
    return match.group(1)


def cwru_sample_rate(mat_path: Path, default_sample_rate: float) -> float:
    if "48k_drive_end" in mat_path.parts:
        return 48000.0
    return default_sample_rate


def convert_cwru(args: argparse.Namespace) -> None:
    root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    good_rows: list[dict[str, object]] = []
    fault_rows: list[dict[str, object]] = []

    for mat_path in sorted(root.rglob("*.mat")):
        file_id = cwru_file_id(mat_path)
        if file_id not in CWRU_MANIFEST:
            continue
        fault_part, load_hp, diameter = CWRU_MANIFEST[file_id]
        signals = [
            load_cwru_signal(mat_path, "DE_time"),
            load_cwru_signal(mat_path, "FE_time"),
            load_cwru_signal(mat_path, "BA_time"),
        ]
        signals = [signal for signal in signals if signal is not None]
        if len(signals) < 1:
            continue
        total_windows = min(len(signal) for signal in signals) // args.window_size
        if args.max_windows_per_file > 0:
            total_windows = min(total_windows, args.max_windows_per_file)
        target_rows = good_rows if fault_part == "normal" else fault_rows
        for window_index in range(total_windows):
            start = window_index * args.window_stride
            end = start + args.window_size
            if end > min(len(signal) for signal in signals):
                break
            row: dict[str, object] = {
                "dataset": "CWRU",
                "source_file": str(mat_path.relative_to(root)),
                "source_position": window_index,
                "load_hp": load_hp,
                "fault_diameter_inch": diameter,
                "fault_part": fault_part,
                "health_score": 1.0 if fault_part == "normal" else max(0.0, 1.0 - diameter / 0.028),
            }
            sample_rate = cwru_sample_rate(mat_path, args.sample_rate)
            for channel, signal in enumerate(signals[:3], start=1):
                add_channel_features(row, channel, signal[start:end], sample_rate)
            repeat_channels(row, source_channels=min(len(signals), 3))
            target_rows.append(row)

    pd.DataFrame(good_rows).to_csv(output_root / "features_good_named.csv", index=False)
    pd.DataFrame(fault_rows).to_csv(output_root / "features_fault_named.csv", index=False)
    print(f"wrote {len(good_rows)} good rows and {len(fault_rows)} fault rows to {output_root}")


def load_paderborn_struct(mat_path: Path):
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    for key, value in data.items():
        if not key.startswith("__"):
            return value
    raise ValueError(f"No Paderborn measurement struct found in {mat_path}")


def paderborn_signal(measurement, name: str) -> np.ndarray | None:
    for item in np.atleast_1d(measurement.Y).reshape(-1):
        if getattr(item, "Name", None) == name:
            return np.asarray(item.Data, dtype=np.float64).reshape(-1)
    return None


def convert_paderborn(args: argparse.Namespace) -> None:
    root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    good_rows: list[dict[str, object]] = []
    fault_rows: list[dict[str, object]] = []
    fault_map = {
        "K001": "normal",
        "KA01": "outer_ring",
        "KI01": "inner_ring",
    }

    for mat_path in sorted(root.rglob("*.mat")):
        bearing = mat_path.parent.name
        fault_part = fault_map.get(bearing)
        if fault_part is None:
            continue
        measurement = load_paderborn_struct(mat_path)
        vibration = paderborn_signal(measurement, "vibration_1")
        current_1 = paderborn_signal(measurement, "phase_current_1")
        current_2 = paderborn_signal(measurement, "phase_current_2")
        signals = [signal for signal in [vibration, current_1, current_2] if signal is not None]
        if not signals:
            continue
        max_windows = min(len(signal) for signal in signals) // args.window_size
        if args.max_windows_per_file > 0:
            max_windows = min(max_windows, args.max_windows_per_file)
        target_rows = good_rows if fault_part == "normal" else fault_rows
        condition_match = re.match(r"(N\d+)_M(\d+)_F(\d+)_", mat_path.name)
        speed_code = condition_match.group(1) if condition_match else ""
        torque_code = condition_match.group(2) if condition_match else ""
        force_code = condition_match.group(3) if condition_match else ""
        for window_index in range(max_windows):
            start = window_index * args.window_stride
            end = start + args.window_size
            if end > min(len(signal) for signal in signals):
                break
            row: dict[str, object] = {
                "dataset": "Paderborn",
                "bearing": bearing,
                "condition": "_".join(part for part in [speed_code, torque_code, force_code] if part),
                "source_file": str(mat_path.relative_to(root)),
                "source_position": window_index,
                "fault_part": fault_part,
                "health_score": 1.0 if fault_part == "normal" else 0.0,
            }
            for channel, signal in enumerate(signals[:3], start=1):
                add_channel_features(row, channel, signal[start:end], args.sample_rate)
            repeat_channels(row, source_channels=min(len(signals), 3))
            target_rows.append(row)

    pd.DataFrame(good_rows).to_csv(output_root / "features_good_named.csv", index=False)
    pd.DataFrame(fault_rows).to_csv(output_root / "features_fault_named.csv", index=False)
    print(f"wrote {len(good_rows)} good rows and {len(fault_rows)} fault rows to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert external bearing datasets to NASA-style feature CSVs.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    xjtu = subparsers.add_parser("xjtu")
    xjtu.add_argument("--input-root", default="data/XJTU/extracted/XJTU-SY_Bearing_Datasets")
    xjtu.add_argument("--output-root", default="data/XJTU/converted")
    xjtu.add_argument("--sample-rate", type=float, default=25600.0)
    xjtu.add_argument("--baseline-fraction", type=float, default=0.2)
    xjtu.add_argument("--min-baseline-files", type=int, default=10)
    xjtu.add_argument("--max-files-per-bearing", type=int, default=0)
    xjtu.add_argument(
        "--health-mode",
        type=str,
        default="time",
        choices=["time", "anomaly", "hybrid"],
        help="XJTU pseudo health target. time uses run-to-failure order; hybrid clips time by anomaly.",
    )

    cwru = subparsers.add_parser("cwru")
    cwru.add_argument("--input-root", default="data/CWRU")
    cwru.add_argument("--output-root", default="data/CWRU/converted")
    cwru.add_argument("--sample-rate", type=float, default=12000.0)
    cwru.add_argument("--window-size", type=int, default=2048)
    cwru.add_argument("--window-stride", type=int, default=2048)
    cwru.add_argument("--max-windows-per-file", type=int, default=0)

    paderborn = subparsers.add_parser("paderborn")
    paderborn.add_argument("--input-root", default="data/Paderborn/extracted")
    paderborn.add_argument("--output-root", default="data/Paderborn/converted")
    paderborn.add_argument("--sample-rate", type=float, default=64000.0)
    paderborn.add_argument("--window-size", type=int, default=4096)
    paderborn.add_argument("--window-stride", type=int, default=4096)
    paderborn.add_argument("--max-windows-per-file", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "xjtu":
        convert_xjtu(args)
    elif args.dataset == "cwru":
        convert_cwru(args)
    elif args.dataset == "paderborn":
        convert_paderborn(args)
    else:
        raise ValueError(args.dataset)


if __name__ == "__main__":
    main()
