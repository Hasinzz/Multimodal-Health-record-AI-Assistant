from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def build_parser(dataset_name: str, default_dir: str, default_output: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Inspect and index the optional {dataset_name} dataset.")
    parser.add_argument("--data-dir", type=str, default=default_dir, help="Path to the dataset root folder.")
    parser.add_argument("--output-csv", type=str, default=default_output, help="Where to write the standardized sample index CSV.")
    parser.add_argument("--max-samples", type=int, default=20, help="Maximum number of samples to include in the sample index.")
    return parser


def discover_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.rglob("*") if path.is_file())


def summarize_files(dataset_name: str, data_dir: Path, files: list[Path]) -> None:
    print(f"Dataset: {dataset_name}")
    print(f"Root: {data_dir}")
    print(f"Files discovered: {len(files)}")
    if files:
        print(f"First file: {files[0]}")
        print(f"Last file: {files[-1]}")
    else:
        print("No files were found. Place the dataset in the expected folder and rerun.")


def write_standardized_index(
    dataset_name: str,
    data_dir: Path,
    output_csv: Path,
    files: list[Path],
    max_samples: int,
) -> Optional[Path]:
    if not files:
        return None

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for file_path in files[:max_samples]:
        rows.append(
            {
                "dataset": dataset_name,
                "source_path": str(file_path),
                "relative_path": str(file_path.relative_to(data_dir)),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_csv


def run_dataset_loader(dataset_name: str, default_dir: str, default_output: str) -> None:
    parser = build_parser(dataset_name, default_dir, default_output)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_csv = Path(args.output_csv)

    files = discover_files(data_dir)
    summarize_files(dataset_name, data_dir, files)

    output_path = write_standardized_index(
        dataset_name=dataset_name,
        data_dir=data_dir,
        output_csv=output_csv,
        files=files,
        max_samples=max(1, args.max_samples),
    )

    if output_path is not None:
        print(f"Standardized sample index written to: {output_path}")
