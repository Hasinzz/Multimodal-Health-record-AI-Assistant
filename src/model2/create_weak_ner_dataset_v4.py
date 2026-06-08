from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSONL = ROOT / "data" / "ner_biobert_v4" / "weak_labels" / "weak_ner_dataset_v4.jsonl"
MANUAL_REVIEW_JSONL = (
    ROOT / "data" / "ner_biobert_v4" / "manual_review" / "weak_ner_manual_review_v4.jsonl"
)
REPORT_PATH = (
    ROOT
    / "outputs"
    / "v4_advanced_improvement"
    / "biobert_ner"
    / "weak_ner_dataset_v4_report.md"
)

TOKEN_RE = re.compile(r"\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?|[A-Za-z]+(?:[-'][A-Za-z]+)*|[^\w\s]")
DATE_RE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\d{4}[/-]\d{1,2}[/-]\d{1,2}$")
RANGE_RE = re.compile(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?$")
DOSE_RE = re.compile(r"^\d+(?:\.\d+)?(?:mg|ml|mcg|g|iu|%)$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")

LAB_TESTS = {
    "hb",
    "haemoglobin",
    "hemoglobin",
    "wbc",
    "rbc",
    "platelet",
    "platelets",
    "glucose",
    "cholesterol",
    "hdl",
    "ldl",
    "triglyceride",
    "triglycerides",
    "creatinine",
    "bun",
    "crp",
    "esr",
    "bilirubin",
    "alt",
    "ast",
    "neutrophils",
    "lymphocytes",
    "eosinophils",
    "monocytes",
    "basophils",
}
UNITS = {"mg", "ml", "mcg", "g", "iu", "unit", "units", "gm", "g/dl", "mg/dl", "fl", "pg", "%", "/ul"}
FREQUENCY = {"od", "bd", "tds", "qid", "daily", "twice", "thrice", "nightly", "morning", "evening"}
CLINICAL = {"fever", "cough", "pain", "infection", "tumor", "diabetes", "hypertension", "asthma"}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def add_term(terms: set[str], value: str) -> None:
    cleaned = normalize(value)
    if len(cleaned) >= 2:
        terms.add(cleaned)
        for part in cleaned.split():
            if len(part) >= 4:
                terms.add(part)


def load_drug_terms(max_rows_per_file: int = 5000) -> set[str]:
    terms: set[str] = set()

    mapping_path = ROOT / "data" / "improvement" / "model2_ocr_prescriptions" / "mapping.json"
    if mapping_path.exists():
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            for key in mapping:
                add_term(terms, str(key))
        except Exception:
            pass

    label_files = [
        ROOT / "data" / "improvement" / "model2_bd_prescriptions" / "Training" / "training_labels.csv",
        ROOT / "data" / "improvement" / "model2_bd_prescriptions" / "Validation" / "validation_labels.csv",
        ROOT / "data" / "improvement" / "model2_bd_prescriptions" / "Testing" / "testing_labels.csv",
    ]
    for path in label_files:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                if index >= max_rows_per_file:
                    break
                add_term(terms, row.get("MEDICINE_NAME", ""))
                add_term(terms, row.get("GENERIC_NAME", ""))

    medicine_csvs = sorted((ROOT / "data" / "improvement" / "model3_medicine_datasets").rglob("*.csv"))
    name_columns = [
        "Medicine Name",
        "Name",
        "product_name",
        "name",
        "short_composition1",
        "short_composition2",
        "Composition",
        "salt_composition",
    ]
    for path in medicine_csvs:
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                if index >= max_rows_per_file:
                    break
                for column in name_columns:
                    add_term(terms, row.get(column, ""))

    return terms


def tokenize(text: str) -> List[str]:
    return [match.group(0) for match in TOKEN_RE.finditer(text)]


def classify_token(token: str, drug_terms: set[str]) -> str:
    lower = normalize(token)
    if not lower:
        return "O"
    if lower in drug_terms:
        return "DRUG"
    if DATE_RE.match(token):
        return "DATE"
    if RANGE_RE.match(token):
        return "REFERENCE_RANGE"
    if DOSE_RE.match(token):
        return "DOSAGE"
    if lower in FREQUENCY:
        return "FREQUENCY"
    if lower in LAB_TESTS:
        return "TEST"
    if token.lower() in UNITS or lower in UNITS:
        return "UNIT"
    if NUMERIC_RE.match(token):
        return "VALUE"
    if lower in CLINICAL:
        return "CLINICAL_FINDING"
    return "O"


def to_bio(entity_labels: Sequence[str]) -> List[str]:
    bio: List[str] = []
    previous = "O"
    for label in entity_labels:
        if label == "O":
            bio.append("O")
        elif label == previous:
            bio.append(f"I-{label}")
        else:
            bio.append(f"B-{label}")
        previous = label
    return bio


def make_record(text: str, source_file: str, source_type: str, drug_terms: set[str]) -> Dict:
    tokens = tokenize(text)
    entity_labels = [classify_token(token, drug_terms) for token in tokens]
    return {
        "tokens": tokens,
        "labels": to_bio(entity_labels),
        "text": text,
        "source_file": source_file,
        "source_type": source_type,
        "weak_label_source": "rule_based_patterns_plus_medicine_dictionaries",
    }


def collect_existing_texts(max_files: int) -> List[Tuple[str, str, str]]:
    roots = [
        ROOT / "outputs" / "final_run_100_tuned_v2",
        ROOT / "outputs" / "final_run_100_retrain_v3",
        ROOT / "outputs" / "main_run_100",
    ]
    text_keys = ["ocr_text", "document_text", "cleaned_text", "text", "summary", "patient_summary_text"]
    collected: List[Tuple[str, str, str]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if len(collected) >= max_files:
                return collected
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            stack = [data]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, str) and key in text_keys and len(value.split()) >= 3:
                            collected.append((value, str(path.relative_to(ROOT)), "existing_model2_output"))
                            break
                        if isinstance(value, (dict, list)):
                            stack.append(value)
                elif isinstance(item, list):
                    stack.extend(item)

    return collected


def collect_medicine_label_texts(max_rows: int) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    path = ROOT / "data" / "improvement" / "model2_bd_prescriptions" / "Training" / "training_labels.csv"
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            text = " ".join(
                value for value in [row.get("MEDICINE_NAME", ""), row.get("GENERIC_NAME", "")] if value
            )
            if text.strip():
                rows.append((text, f"{path.relative_to(ROOT)}:{index + 2}", "medicine_label_csv"))
    return rows


def ocr_lab_samples(max_files: int) -> List[Tuple[str, str, str]]:
    source = ROOT / "data" / "improvement" / "model2_lab_reports" / "lbmaske"
    if not source.exists():
        return []
    images = sorted(path for path in source.rglob("*.png") if path.is_file())[:max_files]
    collected: List[Tuple[str, str, str]] = []
    try:
        from src.model2.ocr import extract_text_from_file
    except Exception as error:
        print(f"Could not import OCR pipeline: {error}")
        return collected

    for path in images:
        try:
            text = extract_text_from_file(str(path))
        except Exception as error:
            print(f"OCR skipped for {path}: {error}")
            continue
        if len(text.split()) >= 3:
            collected.append((text, str(path.relative_to(ROOT)), "lab_report_ocr_sample"))
    return collected


def write_jsonl(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_report(records: Sequence[Dict], drug_term_count: int, sources: Dict[str, int]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled_tokens = sum(
        1 for record in records for label in record["labels"] if label != "O"
    )
    total_tokens = sum(len(record["tokens"]) for record in records)
    lines = [
        "# Weak NER Dataset V4 Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Records written: {len(records)}",
        f"Total tokens: {total_tokens}",
        f"Weak labeled tokens: {labeled_tokens}",
        f"Medicine dictionary terms loaded: {drug_term_count}",
        "",
        "Source counts:",
    ]
    for source, count in sorted(sources.items()):
        lines.append(f"- {source}: {count}")
    lines.extend(
        [
            "",
            "These are weak labels, not gold labels. Use 50 to 100 manually reviewed samples before reporting reliable Entity-F1.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weak BIO labels for V4 BioBERT NER preparation.")
    parser.add_argument("--max-existing-output-files", type=int, default=100)
    parser.add_argument("--max-ocr-files", type=int, default=10)
    parser.add_argument("--max-medicine-label-rows", type=int, default=500)
    parser.add_argument("--max-dictionary-rows-per-file", type=int, default=5000)
    args = parser.parse_args()

    drug_terms = load_drug_terms(args.max_dictionary_rows_per_file)
    texts = collect_existing_texts(args.max_existing_output_files)
    if not texts:
        texts.extend(ocr_lab_samples(args.max_ocr_files))
    texts.extend(collect_medicine_label_texts(args.max_medicine_label_rows))

    records = [
        make_record(text=text, source_file=source_file, source_type=source_type, drug_terms=drug_terms)
        for text, source_file, source_type in texts
        if text.strip()
    ]
    records = [record for record in records if record["tokens"]]

    write_jsonl(OUTPUT_JSONL, records)
    write_jsonl(MANUAL_REVIEW_JSONL, records[:100])

    source_counts: Dict[str, int] = {}
    for record in records:
        source_counts[record["source_type"]] = source_counts.get(record["source_type"], 0) + 1
    write_report(records, len(drug_terms), source_counts)

    print(f"Wrote weak labels: {OUTPUT_JSONL}")
    print(f"Wrote manual review sample: {MANUAL_REVIEW_JSONL}")
    print(f"Wrote report: {REPORT_PATH}")
    print("Weak labels are not gold labels. Manually review validation samples before training claims.")


if __name__ == "__main__":
    main()
