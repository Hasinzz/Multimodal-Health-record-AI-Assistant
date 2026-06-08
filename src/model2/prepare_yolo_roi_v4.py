from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SOURCE_CANDIDATES = [
    ROOT / "data" / "improvement" / "model2_lab_reports" / "lbmaske",
    ROOT / "data" / "documents" / "lab_reports",
]
DEST_DIR = ROOT / "data" / "roi_yolo_v4" / "unlabeled_for_annotation"
INDEX_PATH = (
    ROOT
    / "outputs"
    / "v4_advanced_improvement"
    / "yolo_roi"
    / "yolo_annotation_sample_index.csv"
)


def collect_images(source: Path) -> List[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def first_existing_source() -> Optional[Path]:
    for source in SOURCE_CANDIDATES:
        if source.exists():
            images = collect_images(source)
            if images:
                return source
    return None


def safe_name(index: int, path: Path) -> str:
    clean_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)
    clean_stem = clean_stem[:90].strip("_") or "lab_report"
    return f"{index:04d}_{clean_stem}{path.suffix.lower()}"


def image_size(path: Path) -> tuple[str, str]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return str(img.width), str(img.height)
    except Exception:
        return "", ""


def write_index(rows: Iterable[dict]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "source_file",
        "target_file",
        "width",
        "height",
        "created_at",
        "annotation_status",
    ]
    with INDEX_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy lab report pages into a V4 YOLO annotation staging folder."
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--source-dir", type=Path, default=None)
    args = parser.parse_args()

    source = args.source_dir if args.source_dir else first_existing_source()
    if source is None or not source.exists():
        raise SystemExit(
            "No lab report image source found. Expected data/improvement/model2_lab_reports/lbmaske "
            "or data/documents/lab_reports."
        )

    images = collect_images(source)
    if not images:
        raise SystemExit(f"No image files found in {source}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    selected = images[: max(0, args.max_samples)]
    rows = []
    now = datetime.now().isoformat(timespec="seconds")

    for index, src in enumerate(selected):
        target = DEST_DIR / safe_name(index, src)
        if not target.exists() or target.stat().st_size != src.stat().st_size:
            shutil.copy2(src, target)
        width, height = image_size(target)
        rows.append(
            {
                "sample_index": index,
                "source_file": str(src.relative_to(ROOT)),
                "target_file": str(target.relative_to(ROOT)),
                "width": width,
                "height": height,
                "created_at": now,
                "annotation_status": "unlabeled_manual_bbox_required",
            }
        )

    write_index(rows)

    print(f"Source folder: {source}")
    print(f"Available lab report images: {len(images)}")
    print(f"Copied/staged samples: {len(rows)}")
    print(f"Annotation folder: {DEST_DIR}")
    print(f"Index file: {INDEX_PATH}")
    print(
        "YOLO ROI V4 is prepared, but training cannot start until bounding-box labels "
        "are added under data/roi_yolo_v4/labels/train and labels/val."
    )


if __name__ == "__main__":
    main()
