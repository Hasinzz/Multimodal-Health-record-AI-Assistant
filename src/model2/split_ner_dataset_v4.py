from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "ner_biobert_v4" / "weak_labels" / "weak_ner_dataset_v4.jsonl"
TRAIN_PATH = ROOT / "data" / "ner_biobert_v4" / "train" / "train.jsonl"
VAL_PATH = ROOT / "data" / "ner_biobert_v4" / "val" / "val.jsonl"
TEST_PATH = ROOT / "data" / "ner_biobert_v4" / "test" / "test.jsonl"
REPORT_PATH = (
    ROOT
    / "outputs"
    / "v4_advanced_improvement"
    / "biobert_ner"
    / "ner_split_report_v4.md"
)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Weak NER dataset not found: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if len(record.get("tokens", [])) != len(record.get("labels", [])):
                    raise ValueError(f"Token/label mismatch in {path}")
                records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split V4 weak NER JSONL into train/val/test files.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    records = read_jsonl(args.input_file)
    rng = random.Random(args.seed)
    rng.shuffle(records)

    total = len(records)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)
    test_count = total - train_count - val_count

    train_records = records[:train_count]
    val_records = records[train_count : train_count + val_count]
    test_records = records[train_count + val_count :]

    write_jsonl(TRAIN_PATH, train_records)
    write_jsonl(VAL_PATH, val_records)
    write_jsonl(TEST_PATH, test_records)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# NER Split Report V4",
                "",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                f"Input file: `{args.input_file.relative_to(ROOT)}`",
                f"Seed: {args.seed}",
                f"Total records: {total}",
                f"Train records: {len(train_records)}",
                f"Validation records: {len(val_records)}",
                f"Test records: {len(test_records)}",
                "",
                f"Train output: `{TRAIN_PATH.relative_to(ROOT)}`",
                f"Validation output: `{VAL_PATH.relative_to(ROOT)}`",
                f"Test output: `{TEST_PATH.relative_to(ROOT)}`",
                "",
                "These splits are based on weak labels. Manual correction is still required before making reliable Entity-F1 claims.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {TRAIN_PATH}")
    print(f"Wrote {VAL_PATH}")
    print(f"Wrote {TEST_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
