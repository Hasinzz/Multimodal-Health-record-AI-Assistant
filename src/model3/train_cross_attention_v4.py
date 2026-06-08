from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def load_vector(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    if path.suffix.lower() == ".pt":
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ["embedding", "features", "vector"]:
                if key in value:
                    value = value[key]
                    break
        return torch.as_tensor(value, dtype=torch.float32).flatten()
    if path.suffix.lower() == ".npy":
        import numpy as np

        return torch.as_tensor(np.load(path), dtype=torch.float32).flatten()
    raise ValueError(f"Unsupported embedding format: {path}")


class FusionCsvDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_id: Dict[str, int] | None = None):
        if not csv_path.exists():
            raise FileNotFoundError(f"Fusion CSV not found: {csv_path}")
        self.rows = []
        with csv_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            required = {"image_embedding_path", "text_embedding_path", "label"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{csv_path} is missing {sorted(missing)}. Precompute Model-1 image embeddings "
                    "and BioBERT/SentenceTransformer text embeddings before training."
                )
            for row in reader:
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows found in {csv_path}")

        if label_to_id is None:
            labels = sorted({row["label"] for row in self.rows})
            self.label_to_id = {label: index for index, label in enumerate(labels)}
        else:
            self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = load_vector(Path(row["image_embedding_path"]))
        text = load_vector(Path(row["text_embedding_path"]))
        label = torch.tensor(self.label_to_id[row["label"]], dtype=torch.long)
        return image, text, label


def collate(batch):
    image, text, label = zip(*batch)
    dim = min(min(item.numel() for item in image), min(item.numel() for item in text))
    return (
        torch.stack([item[:dim] for item in image]),
        torch.stack([item[:dim] for item in text]),
        torch.stack(label),
    )


class CrossAttentionClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dim: int = 256, heads: int = 4):
        super().__init__()
        safe_heads = heads if embedding_dim % heads == 0 and embedding_dim >= heads else 1
        self.image_projection = nn.Linear(embedding_dim, hidden_dim)
        self.text_projection = nn.Linear(embedding_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, safe_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image_token = self.image_projection(image).unsqueeze(1)
        text_token = self.text_projection(text).unsqueeze(1)
        sequence = torch.cat([image_token, text_token], dim=1)
        attended, _ = self.attention(sequence, sequence, sequence)
        fused = attended.mean(dim=1)
        return self.classifier(fused)


def macro_f1(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    scores = []
    for label in range(num_classes):
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(model, loader, device, num_classes: int) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for image, text, label in loader:
            image, text, label = image.to(device), text.to(device), label.to(device)
            logits = model(image, text)
            loss = criterion(logits, label)
            pred = logits.argmax(dim=1)
            total_loss += float(loss.item()) * label.size(0)
            total += label.size(0)
            correct += int((pred == label).sum().item())
            y_true.extend(label.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V4 cross-modal attention from precomputed embeddings.")
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v4_advanced_improvement/cross_attention"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/model3/cross_attention_v4"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    train_dataset = FusionCsvDataset(args.train_csv)
    val_dataset = FusionCsvDataset(args.val_csv, label_to_id=train_dataset.label_to_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    first_image, first_text, _ = next(iter(train_loader))
    embedding_dim = first_image.shape[1]
    num_classes = len(train_dataset.label_to_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrossAttentionClassifier(embedding_dim=embedding_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_f1 = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "cross_attention_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for image, text, label in train_loader:
            image, text, label = image.to(device), text.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image, text)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * label.size(0)
            seen += label.size(0)

        metrics = evaluate(model, val_loader, device, num_classes)
        row = {"epoch": epoch, "train_loss": train_loss / seen if seen else 0.0, **metrics}
        history.append(row)
        print(row)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_id": train_dataset.label_to_id,
                    "embedding_dim": embedding_dim,
                    "num_classes": num_classes,
                },
                checkpoint_path,
            )

    metrics_path = args.output_dir / "cross_attention_metrics_v4.json"
    history_path = args.output_dir / "cross_attention_training_history_v4.csv"
    metrics_path.write_text(json.dumps(history[-1], indent=2), encoding="utf-8")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
