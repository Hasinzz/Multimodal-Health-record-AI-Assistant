from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


ENTITY_TYPES = [
    "DRUG",
    "DOSAGE",
    "FREQUENCY",
    "TEST",
    "VALUE",
    "UNIT",
    "REFERENCE_RANGE",
    "PATIENT_INFO",
    "DATE",
    "CLINICAL_FINDING",
]
LABELS = ["O"] + [prefix + "-" + entity for entity in ENTITY_TYPES for prefix in ("B", "I")]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def dependency_error() -> str:
    return (
        "Missing BioBERT training dependencies. Install with:\n"
        "C:\\Users\\T2520824\\Miniconda3\\envs\\thesis_gpu\\python.exe -m pip install "
        "transformers datasets evaluate seqeval accelerate"
    )


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"NER JSONL file not found: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if len(record.get("tokens", [])) != len(record.get("labels", [])):
                    raise ValueError(f"Token/label length mismatch in {path}")
                records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V4 BioBERT token-classification NER.")
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--model-name", default="dmis-lab/biobert-base-cased-v1.1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v4_advanced_improvement/biobert_ner"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/model2/biobert_ner_v4"))
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    try:
        import numpy as np
        from datasets import Dataset
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except Exception as error:
        raise SystemExit(f"{dependency_error()}\nOriginal import error: {error}") from error

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def align_labels(batch):
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
        )
        all_labels = []
        for item_index, labels in enumerate(batch["labels"]):
            word_ids = tokenized.word_ids(batch_index=item_index)
            previous_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(LABEL_TO_ID.get(labels[word_id], 0))
                else:
                    label = labels[word_id]
                    if label.startswith("B-"):
                        label = "I-" + label[2:]
                    label_ids.append(LABEL_TO_ID.get(label, 0))
                previous_word_id = word_id
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    train_dataset = Dataset.from_list(train_records).map(align_labels, batched=True)
    val_dataset = Dataset.from_list(val_records).map(align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        predictions = np.argmax(predictions, axis=2)
        true_predictions = []
        true_labels = []
        for prediction, label in zip(predictions, labels):
            pred_row = []
            label_row = []
            for pred_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue
                pred_row.append(ID_TO_LABEL[int(pred_id)])
                label_row.append(ID_TO_LABEL[int(label_id)])
            true_predictions.append(pred_row)
            true_labels.append(label_row)
        report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
        return {
            "precision": precision_score(true_labels, true_predictions, zero_division=0),
            "recall": recall_score(true_labels, true_predictions, zero_division=0),
            "entity_f1": f1_score(true_labels, true_predictions, zero_division=0),
            "classification_report": report,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(args.output_dir / "logs"),
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(args.checkpoint_dir))
    tokenizer.save_pretrained(str(args.checkpoint_dir))

    metrics_path = args.output_dir / "ner_metrics_v4.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "train.log").write_text(
        "BioBERT NER V4 training completed. See trainer logs and ner_metrics_v4.json.\n",
        encoding="utf-8",
    )

    print(f"Saved checkpoint: {args.checkpoint_dir}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
