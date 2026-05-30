from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import BRAIN_CLASSES, PROJECT_ROOT  # noqa: E402
from src.model1.infer import TimmWithFeatures  # noqa: E402


CLASS_FOLDER_TO_LABEL = {
    "glioma": "glioma_tumor",
    "meningioma": "meningioma_tumor",
    "notumor": "no_tumor",
    "pituitary": "pituitary_tumor",
}

DISPLAY_CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Brain MRI classifier locally using DenseNet-121."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "images" / "brain_mri"),
        help="Root path for the Brain MRI dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
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
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "training" / "brain_mri"),
        help="Directory for CSV, JSON, and plot outputs.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        help="Torchvision/timm backbone name. DenseNet-121 is the default.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "model1" / "brain_best_model_retrained.pt"),
        help="Path where the best retrained checkpoint will be saved.",
    )
    return parser.parse_args()


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
            transforms.RandomRotation(degrees=10),
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


def infer_split_dirs(data_dir: Path) -> Tuple[Path, Optional[Path]]:
    training_dir = data_dir / "Training"
    testing_dir = data_dir / "Testing"

    if training_dir.exists() and testing_dir.exists():
        return training_dir, testing_dir

    return data_dir, None


def validate_class_folders(root_dir: Path) -> None:
    available = {item.name for item in root_dir.iterdir() if item.is_dir()}
    expected = set(CLASS_FOLDER_TO_LABEL.keys())
    missing = sorted(expected - available)

    if missing:
        raise FileNotFoundError(
            f"Missing brain MRI class folders under {root_dir}: {', '.join(missing)}"
        )


def build_datasets(
    data_dir: Path,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    seed: int,
) -> Tuple[Dataset, Dataset, List[str]]:
    train_root, val_root = infer_split_dirs(data_dir)

    if val_root is not None:
        validate_class_folders(train_root)
        validate_class_folders(val_root)

        train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_root, transform=val_transform)

        if train_dataset.classes != val_dataset.classes:
            raise ValueError(
                f"Training and validation class folders do not match: {train_dataset.classes} vs {val_dataset.classes}"
            )

        class_names = [CLASS_FOLDER_TO_LABEL[class_name] for class_name in train_dataset.classes]
        return train_dataset, val_dataset, class_names

    validate_class_folders(train_root)
    full_dataset_train = datasets.ImageFolder(train_root, transform=train_transform)
    full_dataset_val = datasets.ImageFolder(train_root, transform=val_transform)

    total_length = len(full_dataset_train)
    if total_length < 2:
        raise ValueError("Not enough images to build a training/validation split.")

    val_length = max(1, int(round(total_length * 0.2)))
    train_length = total_length - val_length

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset_train,
        [train_length, val_length],
        generator=generator,
    )

    val_subset = Subset(full_dataset_val, val_subset.indices)

    class_names = [CLASS_FOLDER_TO_LABEL[class_name] for class_name in full_dataset_train.classes]
    return train_subset, val_subset, class_names


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    num_workers = min(4, os.cpu_count() or 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def save_history_csv(history: Sequence[Dict[str, float]], output_dir: Path) -> Path:
    csv_path = output_dir / "brain_training_history.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "learning_rate",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return csv_path


def save_json(data: Dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def compute_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class = {
        class_names[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index in range(len(class_names))
    }

    return {
        "accuracy": float(accuracy),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Brain MRI Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            ax.text(
                col_index,
                row_index,
                format(int(cm[row_index, col_index])),
                ha="center",
                va="center",
                color="white" if cm[row_index, col_index] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_curve_plot(
    history: Sequence[Dict[str, float]],
    train_key: str,
    val_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    epochs = [row["epoch"] for row in history]
    train_values = [row[train_key] for row in history]
    val_values = [row[val_key] for row in history]

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


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
    use_amp: bool,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(images)
                loss = criterion(logits, targets)

            predictions = torch.argmax(logits, dim=1)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (predictions == targets).sum().item()
            running_total += batch_size

            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = running_loss / max(1, running_total)
    accuracy = running_correct / max(1, running_total)
    metrics = compute_class_metrics(all_targets, all_predictions, class_names)

    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy),
        "targets": all_targets,
        "predictions": all_predictions,
        "metrics": metrics,
    }


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    epoch_index: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    progress_interval = max(1, len(data_loader) // 5)

    print(f"[Train] Epoch {epoch_index}/{total_epochs}")

    for batch_index, (images, targets) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == targets).sum().item()
        running_total += batch_size

        if batch_index % progress_interval == 0 or batch_index == len(data_loader):
            batch_accuracy = running_correct / max(1, running_total)
            batch_loss = running_loss / max(1, running_total)
            print(
                f"  Batch {batch_index:>4}/{len(data_loader)} | loss={batch_loss:.4f} | acc={batch_accuracy:.4f}"
            )

    average_loss = running_loss / max(1, running_total)
    accuracy = running_correct / max(1, running_total)

    return {
        "train_loss": float(average_loss),
        "train_accuracy": float(accuracy),
    }


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = make_output_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint_path)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_transform, val_transform = build_transforms(args.image_size)
    train_dataset, val_dataset, class_names = build_datasets(
        data_dir=data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        seed=args.seed,
    )

    print("[Data] Training samples:", len(train_dataset))
    print("[Data] Validation samples:", len(val_dataset))
    print("[Data] Classes:", ", ".join(class_names))

    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True

    print(f"[Device] Using {device}")
    print(f"[AMP] {'Enabled' if use_amp else 'Disabled'}")

    model = TimmWithFeatures(backbone_name=args.backbone, num_classes=len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: List[Dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    best_metrics: Dict[str, object] = {}
    best_predictions: List[int] = []
    best_targets: List[int] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
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
        )

        val_result = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            use_amp=use_amp,
        )

        train_loss = train_result["train_loss"]
        train_accuracy = train_result["train_accuracy"]
        val_loss = val_result["loss"]
        val_accuracy = val_result["accuracy"]

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )

        print(
            f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        improved = val_accuracy > best_val_accuracy or (
            np.isclose(val_accuracy, best_val_accuracy) and val_loss < best_val_loss
        )

        if improved:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_result["metrics"]
            best_predictions = list(val_result["predictions"])
            best_targets = list(val_result["targets"])
            save_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                class_names=class_names,
                backbone=args.backbone,
                image_size=args.image_size,
                best_metrics=best_metrics,
            )
            print(f"[Checkpoint] Saved best model to {checkpoint_path}")

    history_csv_path = save_history_csv(history, output_dir)
    save_curve_plot(
        history=history,
        train_key="train_loss",
        val_key="val_loss",
        title="Brain MRI Loss Curve",
        ylabel="Loss",
        output_path=output_dir / "brain_loss_curve.png",
    )
    save_curve_plot(
        history=history,
        train_key="train_accuracy",
        val_key="val_accuracy",
        title="Brain MRI Accuracy Curve",
        ylabel="Accuracy",
        output_path=output_dir / "brain_accuracy_curve.png",
    )

    cm = np.asarray(best_metrics.get("confusion_matrix", []), dtype=np.int64)
    if cm.size == 0:
        cm = confusion_matrix(
            best_targets,
            best_predictions,
            labels=list(range(len(class_names))),
        )

    save_confusion_matrix_plot(
        cm=cm,
        class_names=class_names,
        output_path=output_dir / "brain_confusion_matrix.png",
    )

    metrics_output = {
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_loss": float(best_val_loss),
        "class_names": list(class_names),
        "checkpoint_path": str(checkpoint_path),
        "history_csv": str(history_csv_path),
        "metrics": best_metrics,
    }
    save_json(metrics_output, output_dir / "brain_metrics.json")

    print("[Done] Training complete")
    print(f"[Done] Best epoch: {best_epoch}")
    print(f"[Done] Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"[Done] History CSV: {history_csv_path}")
    print(f"[Done] Metrics JSON: {output_dir / 'brain_metrics.json'}")
    print(f"[Done] Confusion matrix: {output_dir / 'brain_confusion_matrix.png'}")
    print(f"[Done] Loss curve: {output_dir / 'brain_loss_curve.png'}")
    print(f"[Done] Accuracy curve: {output_dir / 'brain_accuracy_curve.png'}")


if __name__ == "__main__":
    main()