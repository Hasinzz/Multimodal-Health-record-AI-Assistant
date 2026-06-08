from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "roi_yolo_v4" / "unlabeled_for_annotation"
DATA_DIR = ROOT / "data" / "roi_yolo_v4"
DEBUG_DIR = ROOT / "outputs" / "v4_advanced_improvement" / "yolo_roi" / "pseudolabel_debug"
INDEX_PATH = ROOT / "outputs" / "v4_advanced_improvement" / "yolo_roi" / "pseudolabel_index_v4.csv"
REPORT_PATH = ROOT / "outputs" / "v4_advanced_improvement" / "yolo_roi" / "yolo_pseudolabel_report_v4.md"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = {
    0: "patient_info",
    1: "test_table",
    2: "remarks",
    3: "signature_stamp",
}


@dataclass
class Box:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    reason: str

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]


def collect_images(source_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_images(images: List[Path], seed: int) -> Dict[str, List[Path]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * 0.70)
    val_count = int(total * 0.15)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def ensure_output_dirs() -> None:
    for split in ["train", "val", "test"]:
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def existing_label_files() -> List[Path]:
    files: List[Path] = []
    for split in ["train", "val", "test"]:
        files.extend((DATA_DIR / "labels" / split).glob("*.txt"))
    return files


def text_mask(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )
    cleaned = np.zeros_like(mask)
    h, w = mask.shape[:2]
    max_component_area = 0.035 * w * h
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 8:
            continue
        if area > max_component_area:
            continue
        if bw > 0.75 * w and bh > 0.05 * h:
            continue
        cv2.drawContours(cleaned, [contour], -1, 255, thickness=-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)


def dense_bbox(mask: np.ndarray, region: Tuple[int, int, int, int], min_pixels: int) -> Optional[Tuple[int, int, int, int, float]]:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = clip_box(region, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = mask[y1:y2, x1:x2]
    ys, xs = np.where(roi > 0)
    if len(xs) < min_pixels:
        return None
    bx1 = int(xs.min() + x1)
    by1 = int(ys.min() + y1)
    bx2 = int(xs.max() + x1 + 1)
    by2 = int(ys.max() + y1 + 1)
    density = float(len(xs) / max(1, (bx2 - bx1) * (by2 - by1)))
    return bx1, by1, bx2, by2, density


def clip_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(1, min(width, int(x2))),
        max(1, min(height, int(y2))),
    )


def pad_box(box: Tuple[int, int, int, int], width: int, height: int, px: int, py: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return clip_box((x1 - px, y1 - py, x2 + px, y2 + py), width, height)


def box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def valid_box(box: Tuple[int, int, int, int], width: int, height: int, min_area_ratio: float, max_area_ratio: float) -> bool:
    area_ratio = box_area(box) / float(width * height)
    x1, y1, x2, y2 = box
    return (
        min_area_ratio <= area_ratio <= max_area_ratio
        and (x2 - x1) >= 0.08 * width
        and (y2 - y1) >= 0.025 * height
    )


def detect_patient_info(mask: np.ndarray) -> Optional[Box]:
    h, w = mask.shape[:2]
    region = (int(0.02 * w), int(0.08 * h), int(0.98 * w), int(0.35 * h))
    result = dense_bbox(mask, region, min_pixels=max(120, int(0.00045 * w * h)))
    if not result:
        return None
    x1, y1, x2, y2, density = result
    box = pad_box((x1, y1, x2, y2), w, h, int(0.025 * w), int(0.015 * h))
    if not valid_box(box, w, h, 0.015, 0.33):
        return None
    return Box(0, *box, confidence=min(0.95, 0.55 + density), reason="upper_header_text_density")


def detect_test_table(mask: np.ndarray) -> Optional[Box]:
    h, w = mask.shape[:2]
    region = (int(0.03 * w), int(0.28 * h), int(0.98 * w), int(0.86 * h))
    result = dense_bbox(mask, region, min_pixels=max(450, int(0.0016 * w * h)))
    if not result:
        return None
    x1, y1, x2, y2, density = result
    box = pad_box((x1, y1, x2, y2), w, h, int(0.025 * w), int(0.012 * h))
    if not valid_box(box, w, h, 0.08, 0.62):
        return None
    return Box(1, *box, confidence=min(0.98, 0.60 + density), reason="central_dense_table_like_text")


def detect_remarks(mask: np.ndarray, table_box: Optional[Box]) -> Optional[Box]:
    h, w = mask.shape[:2]
    start_y = int(0.68 * h)
    if table_box:
        start_y = max(start_y, min(int(0.88 * h), table_box.y2))
    region = (int(0.03 * w), start_y, int(0.78 * w), int(0.96 * h))
    result = dense_bbox(mask, region, min_pixels=max(80, int(0.00025 * w * h)))
    if not result:
        return None
    x1, y1, x2, y2, density = result
    if y2 - y1 > 0.23 * h:
        return None
    box = pad_box((x1, y1, x2, y2), w, h, int(0.02 * w), int(0.012 * h))
    if not valid_box(box, w, h, 0.006, 0.18):
        return None
    return Box(2, *box, confidence=min(0.90, 0.50 + density), reason="lower_left_text_block")


def detect_signature_stamp(mask: np.ndarray) -> Optional[Box]:
    h, w = mask.shape[:2]
    region = (int(0.45 * w), int(0.72 * h), int(0.98 * w), int(0.98 * h))
    result = dense_bbox(mask, region, min_pixels=max(70, int(0.0002 * w * h)))
    if not result:
        return None
    x1, y1, x2, y2, density = result
    if y2 - y1 > 0.24 * h and x2 - x1 > 0.48 * w:
        return None
    box = pad_box((x1, y1, x2, y2), w, h, int(0.02 * w), int(0.014 * h))
    if not valid_box(box, w, h, 0.006, 0.22):
        return None
    return Box(3, *box, confidence=min(0.90, 0.50 + density), reason="lower_right_signature_stamp_density")


def remove_near_duplicate_boxes(boxes: List[Box]) -> List[Box]:
    kept: List[Box] = []
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for existing in kept:
            if box.class_id == existing.class_id and iou(box, existing) > 0.75:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return sorted(kept, key=lambda item: item.class_id)


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = box_area((a.x1, a.y1, a.x2, a.y2)) + box_area((b.x1, b.y1, b.x2, b.y2)) - intersection
    return intersection / union if union else 0.0


def scale_box(box: Box, scale: float, width: int, height: int) -> Box:
    if scale == 1.0:
        return box
    inv = 1.0 / scale
    x1, y1, x2, y2 = clip_box(
        (
            int(round(box.x1 * inv)),
            int(round(box.y1 * inv)),
            int(round(box.x2 * inv)),
            int(round(box.y2 * inv)),
        ),
        width,
        height,
    )
    return Box(box.class_id, x1, y1, x2, y2, box.confidence, box.reason)


def generate_boxes(image_bgr: np.ndarray) -> List[Box]:
    original_h, original_w = image_bgr.shape[:2]
    max_detection_dim = 1400
    scale = min(1.0, max_detection_dim / float(max(original_h, original_w)))
    if scale < 1.0:
        work_image = cv2.resize(
            image_bgr,
            (int(original_w * scale), int(original_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work_image = image_bgr

    mask = text_mask(work_image)
    boxes: List[Box] = []
    patient_info = detect_patient_info(mask)
    if patient_info:
        boxes.append(patient_info)
    test_table = detect_test_table(mask)
    if test_table:
        boxes.append(test_table)
    remarks = detect_remarks(mask, test_table)
    if remarks:
        boxes.append(remarks)
    signature_stamp = detect_signature_stamp(mask)
    if signature_stamp:
        boxes.append(signature_stamp)
    boxes = [scale_box(box, scale, original_w, original_h) for box in boxes]
    return remove_near_duplicate_boxes(boxes)


def yolo_line(box: Box, width: int, height: int) -> str:
    x_center = ((box.x1 + box.x2) / 2.0) / width
    y_center = ((box.y1 + box.y2) / 2.0) / height
    box_width = (box.x2 - box.x1) / width
    box_height = (box.y2 - box.y1) / height
    values = [x_center, y_center, box_width, box_height]
    values = [max(0.0, min(1.0, value)) for value in values]
    return f"{box.class_id} " + " ".join(f"{value:.6f}" for value in values)


def write_debug_image(image_bgr: np.ndarray, boxes: List[Box], output_path: Path) -> None:
    colors = {
        0: (255, 80, 80),
        1: (80, 200, 80),
        2: (80, 180, 255),
        3: (220, 80, 220),
    }
    debug = image_bgr.copy()
    for box in boxes:
        color = colors.get(box.class_id, (255, 255, 255))
        cv2.rectangle(debug, (box.x1, box.y1), (box.x2, box.y2), color, 4)
        label = f"{box.class_name} {box.confidence:.2f}"
        cv2.putText(
            debug,
            label,
            (box.x1, max(30, box.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def copy_and_label_image(image_path: Path, split: str, debug_limit: int, debug_count: int) -> Tuple[dict, List[Box], int]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = image_bgr.shape[:2]
    boxes = generate_boxes(image_bgr)

    target_image = DATA_DIR / "images" / split / image_path.name
    target_label = DATA_DIR / "labels" / split / f"{image_path.stem}.txt"
    shutil.copy2(image_path, target_image)

    label_lines = [yolo_line(box, w, h) for box in boxes]
    target_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    next_debug_count = debug_count
    if debug_count < debug_limit:
        debug_path = DEBUG_DIR / f"{debug_count + 1:03d}_{split}_{image_path.stem}.jpg"
        write_debug_image(image_bgr, boxes, debug_path)
        next_debug_count += 1

    row = {
        "image_path": str(target_image.relative_to(ROOT)),
        "split": split,
        "label_path": str(target_label.relative_to(ROOT)),
        "number_of_boxes": len(boxes),
        "classes_detected": ";".join(sorted({box.class_name for box in boxes})),
    }
    return row, boxes, next_debug_count


def write_index(rows: List[dict]) -> None:
    with INDEX_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "split", "label_path", "number_of_boxes", "classes_detected"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(total_images: int, split_counts: Dict[str, int], rows: List[dict], class_counts: Counter[str]) -> None:
    zero_box_count = sum(1 for row in rows if int(row["number_of_boxes"]) == 0)
    label_files_created = len(rows)
    total_boxes = sum(class_counts.values())
    lines = [
        "# YOLO ROI V4 Pseudo-Label Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This dataset is weakly supervised and pseudo-labeled. It is not manually annotated ground truth.",
        "",
        f"Total input images: {total_images}",
        f"Train images: {split_counts.get('train', 0)}",
        f"Validation images: {split_counts.get('val', 0)}",
        f"Test images: {split_counts.get('test', 0)}",
        f"Label files created: {label_files_created}",
        f"Total pseudo-boxes: {total_boxes}",
        f"Images with zero boxes: {zero_box_count}",
        "",
        "Total boxes per class:",
    ]
    for class_id, class_name in CLASS_NAMES.items():
        lines.append(f"- {class_id} {class_name}: {class_counts.get(class_name, 0)}")
    lines.extend(
        [
            "",
            "Limitations:",
            "",
            "- Boxes are generated from image-processing heuristics, not human annotation.",
            "- Header, table, remarks, and signature regions may overlap or be missed on unusual page layouts.",
            "- Redaction bars, stamps, logos, and scanner artifacts can confuse density-based detection.",
            "- The labels are suitable for a weakly supervised V4 experiment only.",
            "",
            "Warning: these are not manually verified labels.",
            "",
            "Recommendation: manually inspect at least 30 debug images before training.",
            "",
            f"Debug folder: `{DEBUG_DIR.relative_to(ROOT)}`",
            f"Index CSV: `{INDEX_PATH.relative_to(ROOT)}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak YOLO ROI pseudo-labels for V4 lab reports.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-samples", type=int, default=30)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing pseudo-label files. Do not use if manual labels are present.",
    )
    args = parser.parse_args()

    if not args.source_dir.exists():
        raise SystemExit(f"Source folder not found: {args.source_dir}")

    images = collect_images(args.source_dir)
    if not images:
        raise SystemExit(f"No staged images found in {args.source_dir}")

    existing_labels = existing_label_files()
    if existing_labels and not args.force:
        raise SystemExit(
            "Existing YOLO label files were found. Refusing to overwrite labels without --force. "
            "If these are manual labels, keep them and do not run pseudo-label generation."
        )

    ensure_output_dirs()
    splits = split_images(images, args.seed)
    rows: List[dict] = []
    class_counts: Counter[str] = Counter()
    debug_count = 0

    for split, split_images_list in splits.items():
        for image_path in split_images_list:
            row, boxes, debug_count = copy_and_label_image(
                image_path,
                split=split,
                debug_limit=args.debug_samples,
                debug_count=debug_count,
            )
            rows.append(row)
            for box in boxes:
                class_counts[box.class_name] += 1

    write_index(rows)
    write_report(
        total_images=len(images),
        split_counts={split: len(items) for split, items in splits.items()},
        rows=rows,
        class_counts=class_counts,
    )

    print(f"Input images: {len(images)}")
    print(f"Train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Label files created: {len(rows)}")
    print(f"Total pseudo-boxes: {sum(class_counts.values())}")
    print(f"Images with zero boxes: {sum(1 for row in rows if int(row['number_of_boxes']) == 0)}")
    print(f"Debug images written: {min(debug_count, args.debug_samples)}")
    print(f"Wrote index: {INDEX_PATH}")
    print(f"Wrote report: {REPORT_PATH}")
    print("These labels are weak pseudo-labels, not manually verified annotations.")


if __name__ == "__main__":
    main()
