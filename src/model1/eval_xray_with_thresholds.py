from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, XRAY_CLASSES
from src.model1.train_xray import (
    build_image_index,
    load_xray_samples,
    cap_samples,
    split_samples,
    build_transforms,
    XrayDataset,
    safe_class_auroc,
)
from src.model1.infer import TimmWithFeatures


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate X-ray val set with per-class thresholds")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "images" / "xray"))
    parser.add_argument("--metadata-csv", type=str, default=str(PROJECT_ROOT / "data" / "structured" / "Data_Entry_2017.csv"))
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--thresholds-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "training" / "xray_tuned_eval"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def save_table(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    metadata_csv = Path(args.metadata_csv)
    checkpoint_path = Path(args.checkpoint_path)
    thresholds_path = Path(args.thresholds_json)

    with thresholds_path.open("r", encoding="utf-8") as fh:
        thresholds = json.load(fh)

    image_index, _ = build_image_index(data_dir=Path(data_dir), log=lambda msg: None)
    samples, _summary, _unknown = load_xray_samples(metadata_csv=metadata_csv, image_index=image_index, class_names=XRAY_CLASSES, log=lambda msg: None)
    samples = cap_samples(samples, None, seed=args.seed)
    _, val_samples = split_samples(samples, seed=args.seed)

    _, val_transform = build_transforms(args.image_size)
    val_dataset = XrayDataset(val_samples, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmWithFeatures(backbone_name=args.backbone, num_classes=len(XRAY_CLASSES))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        # best-effort; ignore mismatch
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.numpy()
            logits, _ = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_targets.append(targets)
            all_probs.append(probs)

    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, len(XRAY_CLASSES)))
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(XRAY_CLASSES)))

    per_class_auroc, macro_auroc, micro_auroc = safe_class_auroc(y_true, y_prob)

    thr_array = np.array([float(thresholds.get(c, 0.5)) for c in XRAY_CLASSES], dtype=np.float32)
    y_pred = (y_prob >= thr_array[np.newaxis, :]).astype(int)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class_table = []
    for idx, cname in enumerate(XRAY_CLASSES):
        per_class_table.append({
            "class_name": cname,
            "threshold": float(thr_array[idx]),
            "auroc": None if per_class_auroc[idx] is None else float(per_class_auroc[idx]),
            "f1": float(per_class_f1[idx]),
            "precision": float(per_class_precision[idx]),
            "recall": float(per_class_recall[idx]),
        })

    summary = {
        "thresholds_source": str(thresholds_path),
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "per_class_table": per_class_table,
    }

    with (output_dir / "xray_tuned_eval_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    save_table(per_class_table, output_dir / "xray_tuned_eval_per_class.csv")

    print(f"Saved tuned evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
