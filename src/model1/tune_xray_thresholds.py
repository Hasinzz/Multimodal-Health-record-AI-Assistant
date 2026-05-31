from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, TextIO, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROJECT_ROOT, XRAY_CLASSES  # noqa: E402
from src.model1.infer import TimmWithFeatures, clean_state_dict_keys, extract_state_dict  # noqa: E402
from src.model1.train_xray import (  # noqa: E402
    XrayDataset,
    build_dataset_summary,
    build_image_index,
    build_transforms,
    compute_pos_weight,
    load_xray_samples,
    save_json,
    set_seed,
    split_samples,
    to_jsonable,
)


STANDARD_XRAY_CLASSES = list(XRAY_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune per-class Chest X-ray thresholds for a trained multi-label classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "images" / "xray"),
        help="Root path for the Chest X-ray images.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "structured" / "Data_Entry_2017.csv"),
        help="Path to the NIH metadata CSV.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "model1" / "xray_best_model_gpu_full.pt"),
        help="Path to the trained X-ray checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "training" / "xray_gpu_full_threshold_tuning"),
        help="Directory for tuning outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size used for validation transforms.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        help="Backbone name used if the checkpoint does not store one.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
        help="Minimum threshold to search.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.95,
        help="Maximum threshold to search.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold increment during search.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional extra log file path.",
    )
    return parser.parse_args()


def make_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def make_logger(output_log_path: Path, extra_log_file: Optional[str | Path]) -> tuple[Callable[[str], None], List[TextIO]]:
    handles: List[TextIO] = []
    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    handles.append(open(output_log_path, "w", encoding="utf-8"))

    if extra_log_file is not None:
        extra_path = Path(extra_log_file)
        if extra_path.resolve() != output_log_path.resolve():
            extra_path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(open(extra_path, "w", encoding="utf-8"))

    def log(message: str) -> None:
        print(message, flush=True)
        for handle in handles:
            print(message, file=handle, flush=True)

    return log, handles


def close_logger(handles: List[TextIO]) -> None:
    for handle in handles:
        handle.close()


def ensure_threshold_grid(threshold_min: float, threshold_max: float, threshold_step: float) -> np.ndarray:
    if threshold_step <= 0:
        raise ValueError("threshold-step must be positive.")
    if threshold_max < threshold_min:
        raise ValueError("threshold-max must be greater than or equal to threshold-min.")

    count = int(math.floor((threshold_max - threshold_min) / threshold_step)) + 1
    count = max(1, count)
    thresholds = threshold_min + (np.arange(count, dtype=np.float64) * threshold_step)
    thresholds = thresholds[thresholds <= threshold_max + (threshold_step * 0.5)]
    if thresholds.size == 0:
        thresholds = np.array([threshold_min], dtype=np.float64)
    return np.unique(np.round(thresholds, 10))


def safe_class_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[List[Optional[float]], Optional[float], Optional[float]]:
    per_class_auroc: List[Optional[float]] = []
    valid_scores: List[float] = []

    for class_index in range(y_true.shape[1]):
        class_targets = y_true[:, class_index]
        class_probs = y_prob[:, class_index]
        if np.unique(class_targets).size < 2:
            per_class_auroc.append(None)
            continue
        try:
            score = float(roc_auc_score(class_targets, class_probs))
        except ValueError:
            score = None
        per_class_auroc.append(score)
        if score is not None:
            valid_scores.append(score)

    macro_auroc = float(np.mean(valid_scores)) if valid_scores else None

    try:
        micro_auroc = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
    except ValueError:
        micro_auroc = None

    return per_class_auroc, macro_auroc, micro_auroc


def predictions_from_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if y_prob.size == 0:
        return np.zeros_like(y_prob, dtype=np.int32)
    return (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> Dict[str, object]:
    per_class_auroc, macro_auroc, micro_auroc = safe_class_auroc(y_true, y_prob)

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
    for index, class_name in enumerate(class_names):
        per_class_table.append(
            {
                "class_name": class_name,
                "auroc": None if per_class_auroc[index] is None else float(per_class_auroc[index]),
                "f1": float(per_class_f1[index]),
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
            }
        )

    return {
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "per_class_auroc": per_class_auroc,
        "per_class_f1": per_class_f1.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_table": per_class_table,
        "binary_predictions": y_pred,
    }


def collect_validation_outputs(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    class_names: Sequence[str],
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    running_total = 0
    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(images)
                loss = criterion(logits, targets)
                probabilities = torch.sigmoid(logits)

            batch_size = targets.size(0)
            running_loss += float(loss.item()) * batch_size
            running_total += batch_size

            all_targets.append(targets.detach().cpu().numpy())
            all_probs.append(probabilities.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, len(class_names)), dtype=np.float32)
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(class_names)), dtype=np.float32)

    return {
        "loss": float(running_loss / max(1, running_total)),
        "targets": y_true,
        "probabilities": y_prob,
    }


def load_checkpoint_model(
    checkpoint_path: Path,
    backbone_fallback: str,
    device: torch.device,
    log: Callable[[str], None],
) -> tuple[nn.Module, Dict[str, object], List[str], str, int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_class_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    class_names = list(checkpoint_class_names) if checkpoint_class_names else list(STANDARD_XRAY_CLASSES)
    backbone = str(checkpoint.get("backbone", backbone_fallback)) if isinstance(checkpoint, dict) else backbone_fallback
    image_size = int(checkpoint.get("image_size", 224)) if isinstance(checkpoint, dict) else 224

    model = TimmWithFeatures(backbone_name=backbone, num_classes=len(class_names))
    state_dict = extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")
    state_dict = clean_state_dict_keys(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    log(f"[Checkpoint] Loaded: {checkpoint_path}")
    log(f"[Checkpoint] Backbone: {backbone}")
    log(f"[Checkpoint] Image size: {image_size}")
    log(f"[Checkpoint] Class count: {len(class_names)}")
    log(f"[Checkpoint] Missing keys: {len(missing_keys)}")
    log(f"[Checkpoint] Unexpected keys: {len(unexpected_keys)}")

    return model, checkpoint if isinstance(checkpoint, dict) else {}, class_names, backbone, image_size


def build_validation_split(
    data_dir: Path,
    metadata_csv: Path,
    class_names: Sequence[str],
    seed: int,
    log: Callable[[str], None],
) -> tuple[List[object], List[object], Dict[str, object]]:
    image_index, _ = build_image_index(data_dir=data_dir, log=log)
    samples, data_summary, _unknown_labels = load_xray_samples(
        metadata_csv=metadata_csv,
        image_index=image_index,
        class_names=class_names,
        log=log,
    )

    if not samples:
        raise ValueError("No valid X-ray samples were found after matching metadata to image files.")

    train_samples, val_samples = split_samples(samples, seed=seed)
    return train_samples, val_samples, data_summary


def build_loader(samples: Sequence[object], image_size: int, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    _train_transform, val_transform = build_transforms(image_size)
    dataset = XrayDataset(samples, transform=val_transform)
    worker_count = max(0, min(num_workers, os.cpu_count() or 0))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )


def search_thresholds_per_class(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    threshold_grid: np.ndarray,
    log: Callable[[str], None],
) -> tuple[np.ndarray, List[Dict[str, object]]]:
    tuned_thresholds = np.full(len(class_names), 0.5, dtype=np.float32)
    per_class_search_rows: List[Dict[str, object]] = []

    for class_index, class_name in enumerate(class_names):
        class_targets = y_true[:, class_index]
        class_probs = y_prob[:, class_index]

        default_pred = (class_probs >= 0.5).astype(np.int32)
        default_f1 = float(f1_score(class_targets, default_pred, zero_division=0))
        default_precision = float(precision_score(class_targets, default_pred, zero_division=0))
        default_recall = float(recall_score(class_targets, default_pred, zero_division=0))

        if np.sum(class_targets) <= 0:
            log(f"[Threshold] Warning: no positive validation samples for {class_name}; keeping threshold 0.5")
            per_class_search_rows.append(
                {
                    "class_name": class_name,
                    "threshold": 0.5,
                    "default_f1": default_f1,
                    "tuned_f1": default_f1,
                    "default_precision": default_precision,
                    "tuned_precision": default_precision,
                    "default_recall": default_recall,
                    "tuned_recall": default_recall,
                }
            )
            continue

        best_threshold = 0.5
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0

        for threshold in threshold_grid:
            class_pred = (class_probs >= threshold).astype(np.int32)
            class_f1 = float(f1_score(class_targets, class_pred, zero_division=0))
            if class_f1 > best_f1:
                best_f1 = class_f1
                best_threshold = float(threshold)
                best_precision = float(precision_score(class_targets, class_pred, zero_division=0))
                best_recall = float(recall_score(class_targets, class_pred, zero_division=0))

        tuned_thresholds[class_index] = best_threshold
        per_class_search_rows.append(
            {
                "class_name": class_name,
                "threshold": float(best_threshold),
                "default_f1": default_f1,
                "tuned_f1": float(best_f1),
                "default_precision": default_precision,
                "tuned_precision": best_precision,
                "default_recall": default_recall,
                "tuned_recall": best_recall,
            }
        )
        log(
            f"[Threshold] {class_name}: threshold={best_threshold:.2f} | tuned_f1={best_f1:.4f} | "
            f"precision={best_precision:.4f} | recall={best_recall:.4f}"
        )

    return tuned_thresholds, per_class_search_rows


def save_table_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_thresholds_json(class_names: Sequence[str], thresholds: Sequence[float], output_path: Path) -> None:
    payload = {
        "class_names": list(class_names),
        "thresholds": {class_names[index]: float(thresholds[index]) for index in range(len(class_names))},
    }
    save_json(payload, output_path)


def save_f1_comparison_plot(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    classes = [row["class_name"] for row in rows]
    default_f1 = [float(row["default_f1"]) for row in rows]
    tuned_f1 = [float(row["tuned_f1"]) for row in rows]

    y = np.arange(len(classes))
    height = 0.38

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y - height / 2, default_f1, height=height, label="Default 0.5")
    ax.barh(y + height / 2, tuned_f1, height=height, label="Tuned")
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlabel("F1 score")
    ax.set_title("Chest X-ray Per-Class F1: Default vs Tuned Thresholds")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_threshold_bar_plot(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    classes = [row["class_name"] for row in rows]
    thresholds = [float(row["threshold"]) for row in rows]

    y = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y, thresholds, color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Threshold")
    ax.set_title("Chest X-ray Tuned Thresholds")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = make_output_dir(args.output_dir)
    log_path = output_dir / "xray_threshold_tuning_log.txt"
    log, handles = make_logger(log_path, args.log_file)

    try:
        log("[Startup] Chest X-ray threshold tuning started.")
        log("[Startup] No retraining will be performed.")

        data_dir = Path(args.data_dir)
        metadata_csv = Path(args.metadata_csv)
        checkpoint_path = Path(args.checkpoint_path)

        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda"
        if use_amp:
            torch.backends.cudnn.benchmark = True

        log(f"[Device] Using {device}")
        log(f"[AMP] {'Enabled' if use_amp else 'Disabled'}")

        model, _checkpoint, class_names, backbone, image_size = load_checkpoint_model(
            checkpoint_path=checkpoint_path,
            backbone_fallback=args.backbone,
            device=device,
            log=log,
        )

        train_samples, val_samples, data_summary = build_validation_split(
            data_dir=data_dir,
            metadata_csv=metadata_csv,
            class_names=class_names,
            seed=args.seed,
            log=log,
        )

        log(f"[Data] Validation samples: {len(val_samples)}")
        log(f"[Data] Training samples: {len(train_samples)}")
        log(f"[Data] Class names: {', '.join(class_names)}")

        val_loader = build_loader(
            samples=val_samples,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        pos_weight = compute_pos_weight(train_samples, class_names).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        val_output = collect_validation_outputs(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            class_names=class_names,
        )

        y_true = val_output["targets"]
        y_prob = val_output["probabilities"]
        default_pred = (y_prob >= 0.5).astype(np.int32)
        default_metrics = compute_metrics(y_true, y_prob, default_pred, class_names)

        log("[Default] Threshold=0.50 metrics:")
        log(
            f"[Default] macro_auroc={default_metrics['macro_auroc']:.4f} | micro_auroc={default_metrics['micro_auroc']:.4f} | "
            f"macro_f1={default_metrics['macro_f1']:.4f} | micro_f1={default_metrics['micro_f1']:.4f} | "
            f"macro_precision={default_metrics['macro_precision']:.4f} | macro_recall={default_metrics['macro_recall']:.4f}"
        )

        threshold_grid = ensure_threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
        log(
            f"[Threshold] Searching {len(threshold_grid)} thresholds from {args.threshold_min:.2f} to {args.threshold_max:.2f} "
            f"with step {args.threshold_step:.2f}"
        )

        tuned_thresholds, tuned_rows = search_thresholds_per_class(
            y_true=y_true,
            y_prob=y_prob,
            class_names=class_names,
            threshold_grid=threshold_grid,
            log=log,
        )

        tuned_pred = predictions_from_thresholds(y_prob, tuned_thresholds)
        tuned_metrics = compute_metrics(y_true, y_prob, tuned_pred, class_names)

        improvement = {
            "macro_f1": float(tuned_metrics["macro_f1"]) - float(default_metrics["macro_f1"]),
            "micro_f1": float(tuned_metrics["micro_f1"]) - float(default_metrics["micro_f1"]),
            "macro_precision": float(tuned_metrics["macro_precision"]) - float(default_metrics["macro_precision"]),
            "micro_precision": float(tuned_metrics["micro_precision"]) - float(default_metrics["micro_precision"]),
            "macro_recall": float(tuned_metrics["macro_recall"]) - float(default_metrics["macro_recall"]),
            "micro_recall": float(tuned_metrics["micro_recall"]) - float(default_metrics["micro_recall"]),
        }

        log("[Tuned] Per-class thresholds applied.")
        log(
            f"[Tuned] macro_f1={tuned_metrics['macro_f1']:.4f} | micro_f1={tuned_metrics['micro_f1']:.4f} | "
            f"macro_precision={tuned_metrics['macro_precision']:.4f} | macro_recall={tuned_metrics['macro_recall']:.4f}"
        )
        log(
            f"[Improve] macro_f1={improvement['macro_f1']:+.4f} | micro_f1={improvement['micro_f1']:+.4f} | "
            f"macro_precision={improvement['macro_precision']:+.4f} | macro_recall={improvement['macro_recall']:+.4f}"
        )

        default_vs_tuned_summary_rows = [
            {
                "metric": "macro_auroc",
                "default": default_metrics["macro_auroc"],
                "tuned": tuned_metrics["macro_auroc"],
                "change": float(tuned_metrics["macro_auroc"]) - float(default_metrics["macro_auroc"]),
            },
            {
                "metric": "micro_auroc",
                "default": default_metrics["micro_auroc"],
                "tuned": tuned_metrics["micro_auroc"],
                "change": float(tuned_metrics["micro_auroc"]) - float(default_metrics["micro_auroc"]),
            },
            {
                "metric": "macro_f1",
                "default": default_metrics["macro_f1"],
                "tuned": tuned_metrics["macro_f1"],
                "change": improvement["macro_f1"],
            },
            {
                "metric": "micro_f1",
                "default": default_metrics["micro_f1"],
                "tuned": tuned_metrics["micro_f1"],
                "change": improvement["micro_f1"],
            },
            {
                "metric": "macro_precision",
                "default": default_metrics["macro_precision"],
                "tuned": tuned_metrics["macro_precision"],
                "change": improvement["macro_precision"],
            },
            {
                "metric": "micro_precision",
                "default": default_metrics["micro_precision"],
                "tuned": tuned_metrics["micro_precision"],
                "change": improvement["micro_precision"],
            },
            {
                "metric": "macro_recall",
                "default": default_metrics["macro_recall"],
                "tuned": tuned_metrics["macro_recall"],
                "change": improvement["macro_recall"],
            },
            {
                "metric": "micro_recall",
                "default": default_metrics["micro_recall"],
                "tuned": tuned_metrics["micro_recall"],
                "change": improvement["micro_recall"],
            },
        ]

        per_class_rows = []
        for index, class_name in enumerate(class_names):
            default_row = default_metrics["per_class_table"][index]
            tuned_row = tuned_metrics["per_class_table"][index]
            per_class_rows.append(
                {
                    "class_name": class_name,
                    "threshold": float(tuned_thresholds[index]),
                    "default_f1": float(default_row["f1"]),
                    "tuned_f1": float(tuned_row["f1"]),
                    "f1_change": float(tuned_row["f1"]) - float(default_row["f1"]),
                    "default_precision": float(default_row["precision"]),
                    "tuned_precision": float(tuned_row["precision"]),
                    "precision_change": float(tuned_row["precision"]) - float(default_row["precision"]),
                    "default_recall": float(default_row["recall"]),
                    "tuned_recall": float(tuned_row["recall"]),
                    "recall_change": float(tuned_row["recall"]) - float(default_row["recall"]),
                    "auroc": default_row["auroc"],
                }
            )

        threshold_table_rows = []
        for index, row in enumerate(tuned_rows):
            threshold_table_rows.append(
                {
                    "class_name": row["class_name"],
                    "threshold": float(tuned_thresholds[index]),
                    "tuned_f1": float(row["tuned_f1"]),
                    "tuned_precision": float(row["tuned_precision"]),
                    "tuned_recall": float(row["tuned_recall"]),
                    "default_f1": float(row["default_f1"]),
                    "default_precision": float(row["default_precision"]),
                    "default_recall": float(row["default_recall"]),
                    "auroc": default_metrics["per_class_table"][index]["auroc"],
                }
            )

        default_vs_tuned_summary_rows.sort(key=lambda item: item["metric"])

        save_table_csv(threshold_table_rows, output_dir / "xray_tuned_thresholds.csv")
        save_table_csv(default_vs_tuned_summary_rows, output_dir / "xray_default_vs_tuned_summary.csv")
        save_table_csv(per_class_rows, output_dir / "xray_per_class_default_vs_tuned.csv")

        write_thresholds_json(class_names, tuned_thresholds, output_dir / "xray_tuned_thresholds.json")

        metrics_payload = to_jsonable(
            {
                "dataset_summary": build_dataset_summary(train_samples + val_samples, class_names),
                "metadata_summary": data_summary,
                "checkpoint_path": str(checkpoint_path),
                "backbone": backbone,
                "image_size": image_size,
                "seed": args.seed,
                "validation_samples": len(val_samples),
                "threshold_grid": {
                    "min": args.threshold_min,
                    "max": args.threshold_max,
                    "step": args.threshold_step,
                    "count": int(len(threshold_grid)),
                },
                "default_metrics": default_metrics,
                "tuned_metrics": tuned_metrics,
                "improvement": improvement,
                "best_thresholds": {
                    class_names[index]: float(tuned_thresholds[index]) for index in range(len(class_names))
                },
                "per_class_threshold_search": tuned_rows,
                "per_class_default_vs_tuned": per_class_rows,
            }
        )
        save_json(metrics_payload, output_dir / "xray_threshold_tuning_metrics.json")

        save_f1_comparison_plot(per_class_rows, output_dir / "xray_default_vs_tuned_f1_bar.png")
        save_threshold_bar_plot(threshold_table_rows, output_dir / "xray_thresholds_bar.png")

        log("[Output] Files created:")
        for file_name in [
            "xray_threshold_tuning_metrics.json",
            "xray_tuned_thresholds.csv",
            "xray_tuned_thresholds.json",
            "xray_default_vs_tuned_summary.csv",
            "xray_per_class_default_vs_tuned.csv",
            "xray_threshold_tuning_log.txt",
            "xray_default_vs_tuned_f1_bar.png",
            "xray_thresholds_bar.png",
        ]:
            log(f"[Output] {output_dir / file_name}")

        log("[Done] Threshold tuning finished.")
    finally:
        close_logger(handles)


if __name__ == "__main__":
    main()