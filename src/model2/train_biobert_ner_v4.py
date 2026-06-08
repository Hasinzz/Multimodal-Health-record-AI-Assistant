from __future__ import annotations

import argparse
import inspect
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


def load_tokenizer(model_name: str, fallback_model_name: str):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        return tokenizer, model_name, None
    except Exception as error:
        print(f"[Tokenizer] Could not load tokenizer for {model_name}: {error}")
        print(f"[Tokenizer] Falling back to {fallback_model_name} tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=False)
        return tokenizer, fallback_model_name, str(error)


def load_token_classification_model(model_name: str, fallback_model_name: str, num_labels: int):
    from transformers import AutoModelForTokenClassification

    common_kwargs = {
        "num_labels": num_labels,
        "id2label": ID_TO_LABEL,
        "label2id": LABEL_TO_ID,
    }
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_name, **common_kwargs)
        return model, model_name, None
    except Exception as error:
        print(f"[Model] Could not load model for {model_name}: {error}")
        print(f"[Model] Falling back to {fallback_model_name}.")
        model = AutoModelForTokenClassification.from_pretrained(fallback_model_name, **common_kwargs)
        return model, fallback_model_name, str(error)


def _continuation_label(label: str) -> str:
    if label.startswith("B-"):
        return "I-" + label[2:]
    return label


def encode_record(record: Dict, tokenizer, max_length: int) -> Dict:
    tokens = record["tokens"]
    labels = record["labels"]
    input_ids: List[int] = []
    label_ids: List[int] = []

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id

    if cls_id is not None:
        input_ids.append(cls_id)
        label_ids.append(-100)

    max_body_length = max_length - (1 if sep_id is not None else 0)

    for token, label in zip(tokens, labels):
        piece_ids = tokenizer.encode(str(token), add_special_tokens=False)
        if not piece_ids and unk_id is not None:
            piece_ids = [unk_id]
        if not piece_ids:
            continue

        if len(input_ids) + len(piece_ids) > max_body_length:
            break

        for piece_index, piece_id in enumerate(piece_ids):
            input_ids.append(piece_id)
            piece_label = label if piece_index == 0 else _continuation_label(label)
            label_ids.append(LABEL_TO_ID.get(piece_label, 0))

    if sep_id is not None and len(input_ids) < max_length:
        input_ids.append(sep_id)
        label_ids.append(-100)

    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": label_ids,
    }


def encode_records(records: List[Dict], tokenizer, max_length: int) -> List[Dict]:
    encoded = [encode_record(record, tokenizer=tokenizer, max_length=max_length) for record in records]
    return [record for record in encoded if record["input_ids"]]


def build_metrics(predictions, labels, include_report: bool = False) -> Dict:
    import numpy as np
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

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

    metrics = {
        "precision": precision_score(true_labels, true_predictions, zero_division=0),
        "recall": recall_score(true_labels, true_predictions, zero_division=0),
        "entity_f1": f1_score(true_labels, true_predictions, zero_division=0),
    }
    if include_report:
        metrics["classification_report"] = classification_report(
            true_labels,
            true_predictions,
            output_dict=True,
            zero_division=0,
        )
    return metrics


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def make_training_args(args):
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    strategy_key = (
        "evaluation_strategy"
        if "evaluation_strategy" in signature.parameters
        else "eval_strategy"
    )
    kwargs = {
        "output_dir": str(args.output_dir / "trainer_checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        strategy_key: "epoch",
        "save_strategy": "epoch",
        "logging_dir": str(args.output_dir / "logs"),
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "entity_f1",
        "greater_is_better": True,
        "report_to": [],
    }
    return TrainingArguments(**kwargs)


def make_trainer(Trainer, model, training_args, train_dataset, val_dataset, tokenizer, data_collator, compute_metrics):
    signature = inspect.signature(Trainer.__init__)
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    if "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    return Trainer(**kwargs)


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
    parser.add_argument("--fallback-model-name", default="bert-base-cased")
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from transformers import (
            DataCollatorForTokenClassification,
            Trainer,
        )
    except Exception as error:
        raise SystemExit(f"{dependency_error()}\nOriginal import error: {error}") from error

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)

    tokenizer, tokenizer_name_used, tokenizer_error = load_tokenizer(
        args.model_name,
        args.fallback_model_name,
    )

    train_dataset = Dataset.from_list(
        encode_records(train_records, tokenizer=tokenizer, max_length=args.max_length)
    )
    val_dataset = Dataset.from_list(
        encode_records(val_records, tokenizer=tokenizer, max_length=args.max_length)
    )

    model, model_name_used, model_error = load_token_classification_model(
        args.model_name,
        args.fallback_model_name,
        num_labels=len(LABELS),
    )

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        return build_metrics(predictions, labels, include_report=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = make_training_args(args)

    trainer = make_trainer(
        Trainer=Trainer,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    prediction_output = trainer.predict(val_dataset)
    metrics = dict(prediction_output.metrics)
    metrics.update(
        {
            "model_requested": args.model_name,
            "model_used": model_name_used,
            "model_fallback_used": model_name_used != args.model_name,
            "tokenizer_requested": args.model_name,
            "tokenizer_used": tokenizer_name_used,
            "tokenizer_fallback_used": tokenizer_name_used != args.model_name,
            "tokenizer_error": tokenizer_error,
            "model_error": model_error,
            "train_records": len(train_dataset),
            "validation_records": len(val_dataset),
        }
    )
    metrics.update(
        {
            "entity_metrics": build_metrics(
                prediction_output.predictions,
                prediction_output.label_ids,
                include_report=True,
            )
        }
    )
    trainer.save_model(str(args.checkpoint_dir))
    tokenizer.save_pretrained(str(args.checkpoint_dir))

    metrics_path = args.output_dir / "ner_metrics_v4.json"
    metrics_path.write_text(json.dumps(make_json_safe(metrics), indent=2), encoding="utf-8")
    (args.output_dir / "train.log").write_text(
        "BioBERT/BERT NER V4 training completed. See trainer logs and ner_metrics_v4.json.\n",
        encoding="utf-8",
    )

    print(f"Saved checkpoint: {args.checkpoint_dir}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
