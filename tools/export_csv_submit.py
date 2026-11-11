from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


DATASETS = {
    "NASA": Path("data/NasaBearingDataset"),
    "CWRU": Path("data/CWRU/converted_full"),
    "Paderborn": Path("data/Paderborn/converted"),
}


def add_minimal_labels(frame: pd.DataFrame, dataset: str, source: str) -> pd.DataFrame:
    frame = frame.copy()
    if "dataset" not in frame.columns:
        frame.insert(0, "dataset", dataset)
    if "source_label" not in frame.columns:
        frame.insert(1, "source_label", source)
    if "health_score" not in frame.columns:
        frame.insert(2, "health_score", 1.0 if source == "good" else 0.0)
    if "fault_part" not in frame.columns:
        frame.insert(3, "fault_part", "normal" if source == "good" else "unknown_fault")
    return frame


def export_dataset(name: str, source_root: Path, output_root: Path) -> None:
    target = output_root / name
    target.mkdir(parents=True, exist_ok=True)
    for filename, source in [
        ("features_good_named.csv", "good"),
        ("features_fault_named.csv", "fault"),
    ]:
        source_path = source_root / filename
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        frame = pd.read_csv(source_path)
        frame = add_minimal_labels(frame, name, source)
        frame.to_csv(target / filename, index=False)
    readme = target / "README.txt"
    readme.write_text(
        f"{name} CSV export for submission.\n"
        "Files: features_good_named.csv, features_fault_named.csv\n"
        "These files are exported for submission and are not tied to one model input format.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export non-XJTU datasets as submission CSVs.")
    parser.add_argument("--output-root", default="exports/csv_submit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, source_root in DATASETS.items():
        export_dataset(name, source_root, output_root)
    print(f"wrote CSV exports to {output_root}")


if __name__ == "__main__":
    main()
