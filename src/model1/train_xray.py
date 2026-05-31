from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, TextIO, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROJECT_ROOT, XRAY_CLASSES  # noqa: E402
from src.model1.infer import TimmWithFeatures, clean_state_dict_keys, extract_state_dict  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_METADATA_COLUMNS = {
    "image_index": "Image Index",
    "finding_labels": "Finding Labels",
}


@dataclass(frozen=True)
class XraySample:
    image_path: Path
    labels: np.ndarray
    image_name: str
    raw_labels: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Chest X-ray multi-label classifier locally using a timm backbone."
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
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "training" / "xray_gpu"),
        help="Directory for CSV, JSON, and plot outputs.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "model1" / "xray_best_model_gpu.pt"),
        help="Path where the best X-ray checkpoint will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size used for training and validation.",
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
        help="timm backbone name. DenseNet-121 is the default.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5,
        help="Approximate number of batch progress updates to print per epoch.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional file path for duplicating important training logs.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path used to initialize model weights before training.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "plateau"],
        default="cosine",
        help="Learning-rate scheduler to use during training.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-6,
        help="Minimum learning rate used by cosine or plateau scheduling.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop if macro AUROC does not improve for this many epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Optional gradient clipping value. Set 0 to disable.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of valid samples to use before splitting.",
    )
    return parser.parse_args()


def make_logger(log_file: Optional[str | Path]) -> tuple[Callable[[str], None], Optional[TextIO]]:
    log_handle: Optional[TextIO] = None
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        if log_handle is not None:
            print(message, file=log_handle, flush=True)

    return log, log_handle


def close_logger(log_handle: Optional[TextIO]) -> None:
    if log_handle is not None:
        log_handle.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, val_transform


def normalize_column_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_")


def resolve_metadata_column(fieldnames: Sequence[str], expected_name: str) -> str:
    normalized_expected = normalize_column_name(expected_name)
    for fieldname in fieldnames:
        if normalize_column_name(fieldname) == normalized_expected:
            return fieldname
    raise KeyError(f"Required metadata column not found: {expected_name}")


def build_image_index(data_dir: Path, log: Callable[[str], None]) -> Tuple[Dict[str, Path], int]:
    image_index: Dict[str, Path] = {}
    duplicate_count = 0

    image_paths = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    for image_path in image_paths:
        key = image_path.name.lower()
        if key in image_index:
            duplicate_count += 1
            continue
        image_index[key] = image_path

    log(f"[Data] Indexed {len(image_index)} image files under {data_dir}")
    if duplicate_count:
        log(f"[Data] Duplicate image filenames ignored during indexing: {duplicate_count}")

    return image_index, duplicate_count


def parse_xray_labels(
    raw_labels: str,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    label_vector = np.zeros(len(class_names), dtype=np.float32)
    if not raw_labels:
        return label_vector, []

    stripped = raw_labels.strip()
    if stripped == "No Finding":
        return label_vector, []

    known_labels = []
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    for label in (part.strip() for part in stripped.split("|")):
        if not label:
            continue
        if label == "No Finding":
            return np.zeros(len(class_names), dtype=np.float32), []
        if label in class_to_index:
            label_vector[class_to_index[label]] = 1.0
            known_labels.append(label)
        else:
            continue

    return label_vector, known_labels


def load_xray_samples(
    metadata_csv: Path,
    image_index: Dict[str, Path],
    class_names: Sequence[str],
    log: Callable[[str], None],
) -> Tuple[List[XraySample], Dict[str, int], List[str]]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Unable to read metadata header from {metadata_csv}")

        image_column = resolve_metadata_column(reader.fieldnames, DEFAULT_METADATA_COLUMNS["image_index"])
        labels_column = resolve_metadata_column(reader.fieldnames, DEFAULT_METADATA_COLUMNS["finding_labels"])

        log(f"[Data] Metadata columns detected: {image_column}, {labels_column}")

        samples: List[XraySample] = []
        missing_images = 0
        unknown_labels: set[str] = set()
        raw_label_counts: Dict[str, int] = {}

        for row in reader:
            image_name = (row.get(image_column) or "").strip()
            raw_labels = (row.get(labels_column) or "").strip()
            if not image_name:
                continue

            labels, _known_labels = parse_xray_labels(raw_labels, class_names)
            if raw_labels and raw_labels != "No Finding":
                raw_label_counts[raw_labels] = raw_label_counts.get(raw_labels, 0) + 1
                for label in (part.strip() for part in raw_labels.split("|")):
                    if label and label not in class_names and label != "No Finding":
                        unknown_labels.add(label)

            image_path = image_index.get(image_name.lower())
            if image_path is None:
                missing_images += 1
                continue

            samples.append(
                XraySample(
                    image_path=image_path,
                    labels=labels,
                    image_name=image_name,
                    raw_labels=raw_labels,
                )
            )

    log(f"[Data] Valid samples found: {len(samples)}")
    log(f"[Data] Missing images skipped: {missing_images}")
    if unknown_labels:
        log(f"[Data] Unknown labels encountered and ignored: {', '.join(sorted(unknown_labels))}")

    summary = {
        "missing_images": missing_images,
        "unknown_label_count": len(unknown_labels),
        "unique_raw_label_strings": len(raw_label_counts),
    }

    return samples, summary, sorted(unknown_labels)


def cap_samples(samples: Sequence[XraySample], max_samples: Optional[int], seed: int) -> List[XraySample]:
    sample_list = list(samples)
    if max_samples is None or max_samples >= len(sample_list):
        return sample_list

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(sample_list), generator=generator).tolist()
    selected_indices = sorted(order[:max_samples])
    return [sample_list[index] for index in selected_indices]


class XrayDataset(Dataset):
    def __init__(self, samples: Sequence[XraySample], transform: transforms.Compose):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image = self.transform(image)
        target = torch.from_numpy(sample.labels.astype(np.float32))
        return image, target


def split_samples(
    samples: Sequence[XraySample],
    seed: int,
) -> Tuple[List[XraySample], List[XraySample]]:
    total = len(samples)
    if total < 2:
        raise ValueError("Not enough valid X-ray samples to build a train/validation split.")

    val_size = max(1, int(round(total * 0.2)))
    train_size = total - val_size
    if train_size < 1:
        train_size = 1
        val_size = total - train_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(list(samples), [train_size, val_size], generator=generator)
    train_samples = [samples[index] for index in train_subset.indices]
    val_samples = [samples[index] for index in val_subset.indices]
    return train_samples, val_samples


def build_dataloaders(
    train_samples: Sequence[XraySample],
    val_samples: Sequence[XraySample],
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_transform, val_transform = build_transforms(image_size)
    train_dataset = XrayDataset(train_samples, transform=train_transform)
    val_dataset = XrayDataset(val_samples, transform=val_transform)

    worker_count = max(0, min(num_workers, os.cpu_count() or 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def build_dataset_summary(
    samples: Sequence[XraySample],
    class_names: Sequence[str],
) -> Dict[str, object]:
    labels = np.stack([sample.labels for sample in samples], axis=0) if samples else np.zeros((0, len(class_names)))
    positive_counts = labels.sum(axis=0) if len(labels) else np.zeros(len(class_names))
    return {
        "total_samples": int(len(samples)),
        "class_names": list(class_names),
        "positive_counts": {
            class_names[index]: int(positive_counts[index]) for index in range(len(class_names))
        },
    }


def compute_pos_weight(train_samples: Sequence[XraySample], class_names: Sequence[str]) -> torch.Tensor:
    labels = np.stack([sample.labels for sample in train_samples], axis=0).astype(np.float32)
    positive_counts = labels.sum(axis=0)
    total = float(len(train_samples))
    negative_counts = total - positive_counts
    denominator = np.maximum(positive_counts, 1.0)
    pos_weight = negative_counts / denominator
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_binary_predictions(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probabilities >= threshold).astype(np.int32)


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


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    class_names: Sequence[str],
) -> Dict[str, object]:
    y_pred = compute_binary_predictions(y_prob, threshold=threshold)

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
        "threshold": float(threshold),
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


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    class_names: Sequence[str],
    threshold: float = 0.5,
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

    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, len(class_names)))
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(class_names)))

    metrics = compute_multilabel_metrics(y_true, y_prob, threshold=threshold, class_names=class_names)

    return {
        "loss": float(running_loss / max(1, running_total)),
        "targets": y_true,
        "probabilities": y_prob,
        "metrics": metrics,
    }


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    epoch_index: int,
    total_epochs: int,
    progress_interval: int,
    log: Callable[[str], None],
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_total = 0
    total_batches = len(data_loader)
    progress_every = max(1, math.ceil(total_batches / max(1, progress_interval)))

    log(f"[Train] Epoch {epoch_index}/{total_epochs}")

    for batch_index, (images, targets) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

        batch_size = targets.size(0)
        running_loss += float(loss.item()) * batch_size
        running_total += batch_size

        if batch_index % progress_every == 0 or batch_index == total_batches:
            average_loss = running_loss / max(1, running_total)
            log(f"  Batch {batch_index:>4}/{total_batches} | loss={average_loss:.4f}")

    return {"train_loss": float(running_loss / max(1, running_total))}


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    epoch: int,
    class_names: Sequence[str],
    backbone: str,
    image_size: int,
    best_metrics: Dict[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "backbone": backbone,
            "image_size": image_size,
            "class_names": list(class_names),
            "model_state_dict": model.state_dict(),
            "best_metrics": best_metrics,
        },
        checkpoint_path,
    )


def save_history_csv(history: Sequence[Dict[str, object]], output_dir: Path) -> Path:
    csv_path = output_dir / "xray_training_history.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "macro_auroc",
        "micro_auroc",
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "micro_precision",
        "macro_recall",
        "micro_recall",
        "learning_rate",
        "best_score",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return csv_path


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


def save_json(data: Dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64, np.float16)):
        return float(value)
    if isinstance(value, (np.int32, np.int64, np.int16, np.int8, np.uint32, np.uint64)):
        return int(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_curve_plot(
    history: Sequence[Dict[str, object]],
    train_key: str,
    val_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    epochs = [row["epoch"] for row in history]
    train_values = [np.nan if row.get(train_key) is None else row.get(train_key) for row in history]
    val_values = [np.nan if row.get(val_key) is None else row.get(val_key) for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_values, marker="o", label="Train")
    ax.plot(epochs, val_values, marker="o", label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metric_curve_plot(
    history: Sequence[Dict[str, object]],
    key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    epochs = [row["epoch"] for row in history]
    values = [np.nan if row.get(key) is None else row.get(key) for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_optional_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def load_initial_checkpoint(model: nn.Module, init_checkpoint: Optional[str | Path], device: torch.device, log: Callable[[str], None]) -> None:
    if init_checkpoint is None:
        return

    checkpoint_path = Path(init_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported init checkpoint format: {checkpoint_path}")

    state_dict = clean_state_dict_keys(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    log(f"[Init] Loaded init checkpoint from {checkpoint_path}")
    log(f"[Init] Missing keys: {len(missing_keys)}")
    log(f"[Init] Unexpected keys: {len(unexpected_keys)}")


def build_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    min_learning_rate: float,
) -> Optional[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau]:
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=min_learning_rate,
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=min_learning_rate,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    log, log_handle = make_logger(args.log_file)

    try:
        log(
            "[Startup] For live GPU output, use: conda run --no-capture-output -n thesis_gpu python -m src.model1.train_xray ..."
        )

        output_dir = make_output_dir(args.output_dir)
        checkpoint_path = Path(args.checkpoint_path)
        data_dir = Path(args.data_dir)
        metadata_csv = Path(args.metadata_csv)

        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

        image_index, _ = build_image_index(data_dir=data_dir, log=log)
        samples, data_summary, unknown_labels = load_xray_samples(
            metadata_csv=metadata_csv,
            image_index=image_index,
            class_names=XRAY_CLASSES,
            log=log,
        )

        if not samples:
            raise ValueError("No valid X-ray samples were found after matching metadata to image files.")

        if args.max_samples is not None:
            samples = cap_samples(samples, args.max_samples, seed=args.seed)
            log(f"[Data] Max samples applied: {len(samples)}")

        train_samples, val_samples = split_samples(samples, seed=args.seed)

        dataset_summary = build_dataset_summary(samples, XRAY_CLASSES)
        log(f"[Data] Training samples: {len(train_samples)}")
        log(f"[Data] Validation samples: {len(val_samples)}")
        log(f"[Data] Class names: {', '.join(XRAY_CLASSES)}")

        train_transform, val_transform = build_transforms(args.image_size)
        train_dataset = XrayDataset(train_samples, train_transform)
        val_dataset = XrayDataset(val_samples, val_transform)

        worker_count = max(0, min(args.num_workers, os.cpu_count() or 0))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=worker_count,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=torch.cuda.is_available(),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda"
        if use_amp:
            torch.backends.cudnn.benchmark = True

        log(f"[Device] Using {device}")
        log(f"[AMP] {'Enabled' if use_amp else 'Disabled'}")

        model = TimmWithFeatures(backbone_name=args.backbone, num_classes=len(XRAY_CLASSES))
        model.to(device)
        load_initial_checkpoint(model=model, init_checkpoint=args.init_checkpoint, device=device, log=log)

        train_label_matrix = np.stack([sample.labels for sample in train_samples], axis=0)
        train_positive_counts = train_label_matrix.sum(axis=0)
        pos_weight = compute_pos_weight(train_samples, XRAY_CLASSES).to(device)
        zero_positive_classes = [
            XRAY_CLASSES[index]
            for index, count in enumerate(train_positive_counts)
            if count <= 0
        ]
        if zero_positive_classes:
            log(f"[Data] Classes with zero positives in training split: {', '.join(zero_positive_classes)}")
        log("[Loss] BCEWithLogitsLoss pos_weight prepared from training labels")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = build_scheduler(
            scheduler_name=args.scheduler,
            optimizer=optimizer,
            total_epochs=args.epochs,
            min_learning_rate=args.min_learning_rate,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        history: List[Dict[str, object]] = []
        best_score = -float("inf")
        best_score_name = "macro_auroc"
        best_epoch = 0
        best_metrics: Dict[str, object] = {}
        best_val_metrics: Dict[str, object] = {}
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            log(f"[Train] Current learning rate: {current_lr:.8f}")
            train_result = train_one_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                epoch_index=epoch,
                total_epochs=args.epochs,
                progress_interval=args.progress_interval,
                log=log,
                grad_clip=args.grad_clip,
            )

            val_result = evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                class_names=XRAY_CLASSES,
                threshold=0.5,
            )

            metrics = val_result["metrics"]
            val_loss = float(val_result["loss"])
            macro_auroc = metrics["macro_auroc"]
            macro_f1 = metrics["macro_f1"]

            if macro_auroc is not None:
                current_score = float(macro_auroc)
                current_score_name = "macro_auroc"
            else:
                current_score = float(macro_f1)
                current_score_name = "macro_f1"

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_result["train_loss"]),
                    "val_loss": val_loss,
                    "macro_auroc": macro_auroc,
                    "micro_auroc": metrics["micro_auroc"],
                    "macro_f1": macro_f1,
                    "micro_f1": metrics["micro_f1"],
                    "macro_precision": metrics["macro_precision"],
                    "micro_precision": metrics["micro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "micro_recall": metrics["micro_recall"],
                    "learning_rate": args.learning_rate,
                    "best_score": current_score,
                    "best_score_name": current_score_name,
                }
            )

            log(
                f"[Validation] Epoch {epoch}/{args.epochs} | loss={val_loss:.4f} | "
                f"macro_auroc={format_optional_metric(macro_auroc)} | micro_auroc={format_optional_metric(metrics['micro_auroc'])} | "
                f"macro_f1={macro_f1:.4f} | micro_f1={metrics['micro_f1']:.4f} | "
                f"macro_precision={metrics['macro_precision']:.4f} | macro_recall={metrics['macro_recall']:.4f}"
            )

            if current_score > best_score:
                best_score = current_score
                best_score_name = current_score_name
                best_epoch = epoch
                best_metrics = to_jsonable(
                    {
                        "val_loss": val_loss,
                        **metrics,
                    }
                )
                best_val_metrics = best_metrics
                save_checkpoint(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    class_names=XRAY_CLASSES,
                    backbone=args.backbone,
                    image_size=args.image_size,
                    best_metrics=best_metrics,
                )
                log(
                    f"[Checkpoint] Saved best model to {checkpoint_path} using {current_score_name}={current_score:.4f}"
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(current_score)
                else:
                    scheduler.step()

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                log(
                    f"[EarlyStopping] No macro AUROC improvement for {epochs_without_improvement} epoch(s); stopping early at epoch {epoch}."
                )
                break

        history_csv = save_history_csv(history, output_dir)

        final_eval = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            class_names=XRAY_CLASSES,
            threshold=0.5,
        )
        final_metrics = to_jsonable(
            {
                "dataset_summary": dataset_summary,
                "metadata_summary": data_summary,
                "unknown_labels": unknown_labels,
                "training_history_csv": str(history_csv),
                "checkpoint_path": str(checkpoint_path),
                "best_epoch": best_epoch,
                "best_score_name": best_score_name,
                "best_score_value": None if best_score == -float("inf") else best_score,
                "best_metrics": best_val_metrics,
                "final_validation": {
                    "loss": final_eval["loss"],
                    **final_eval["metrics"],
                },
            }
        )
        save_json(final_metrics, output_dir / "xray_metrics.json")

        auc_rows = []
        f1_rows = []
        threshold_rows = []
        for class_index, class_name in enumerate(XRAY_CLASSES):
            auc_rows.append(
                {
                    "class_name": class_name,
                    "auroc": None if final_eval["metrics"]["per_class_auroc"][class_index] is None else float(final_eval["metrics"]["per_class_auroc"][class_index]),
                }
            )
            f1_rows.append(
                {
                    "class_name": class_name,
                    "f1": float(final_eval["metrics"]["per_class_f1"][class_index]),
                    "precision": float(final_eval["metrics"]["per_class_precision"][class_index]),
                    "recall": float(final_eval["metrics"]["per_class_recall"][class_index]),
                }
            )
            threshold_rows.append(
                {
                    "class_name": class_name,
                    "threshold": 0.5,
                    "auroc": None if final_eval["metrics"]["per_class_auroc"][class_index] is None else float(final_eval["metrics"]["per_class_auroc"][class_index]),
                    "f1": float(final_eval["metrics"]["per_class_f1"][class_index]),
                    "precision": float(final_eval["metrics"]["per_class_precision"][class_index]),
                    "recall": float(final_eval["metrics"]["per_class_recall"][class_index]),
                }
            )

        save_table_csv(auc_rows, output_dir / "xray_auc_table.csv")
        save_table_csv(f1_rows, output_dir / "xray_f1_table.csv")
        save_table_csv(threshold_rows, output_dir / "xray_threshold_table.csv")

        save_curve_plot(
            history=history,
            train_key="train_loss",
            val_key="val_loss",
            title="Chest X-ray Training Loss",
            ylabel="Loss",
            output_path=output_dir / "xray_loss_curve.png",
        )
        save_metric_curve_plot(
            history=history,
            key="macro_auroc",
            title="Chest X-ray Macro AUROC",
            ylabel="Macro AUROC",
            output_path=output_dir / "xray_macro_auc_curve.png",
        )
        save_metric_curve_plot(
            history=history,
            key="macro_f1",
            title="Chest X-ray Macro F1",
            ylabel="Macro F1",
            output_path=output_dir / "xray_macro_f1_curve.png",
        )

        log(f"[Output] Training history saved to {output_dir / 'xray_training_history.csv'}")
        log(f"[Output] Metrics JSON saved to {output_dir / 'xray_metrics.json'}")
        log(f"[Output] AUROC table saved to {output_dir / 'xray_auc_table.csv'}")
        log(f"[Output] F1 table saved to {output_dir / 'xray_f1_table.csv'}")
        log(f"[Output] Threshold table saved to {output_dir / 'xray_threshold_table.csv'}")
        log(f"[Output] Loss curve saved to {output_dir / 'xray_loss_curve.png'}")
        log(f"[Output] Macro AUROC curve saved to {output_dir / 'xray_macro_auc_curve.png'}")
        log(f"[Output] Macro F1 curve saved to {output_dir / 'xray_macro_f1_curve.png'}")
        log(f"[Done] Chest X-ray training finished. Best score: {best_score_name}={best_score:.4f} at epoch {best_epoch}")
    finally:
        close_logger(log_handle)


if __name__ == "__main__":
    main()