from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Optional

from src.model2.advanced_ner import extract_entities
from src.model2.clean_text import clean_ocr_text, make_preview
from src.model2.ocr_engines import extract_text_with_engine


def build_structured_json(entities: list[dict]) -> dict:
    structured = {
        "patient_info": {},
        "dates": [],
        "lab_values": [],
        "medications_or_dosage_lines": [],
        "clinical_keywords": [],
        "other_entities": [],
    }

    for entity in entities:
        entity_type = entity.get("type")
        if entity_type == "PATIENT_INFO":
            field = entity.get("field", "name")
            value = entity.get("text")
            if field and value:
                structured["patient_info"][field] = value
        elif entity_type == "DATE":
            structured["dates"].append(entity)
        elif entity_type == "LAB_VALUE":
            structured["lab_values"].append(entity)
        elif entity_type == "MEDICATION_OR_DOSAGE_LINE":
            structured["medications_or_dosage_lines"].append(entity)
        elif entity_type == "CLINICAL_KEYWORD":
            structured["clinical_keywords"].append(entity)
        else:
            structured["other_entities"].append(entity)

    return structured


def generate_document_summary(structured_json: Dict) -> str:
    patient_info = structured_json.get("patient_info", {})
    lab_values = structured_json.get("lab_values", [])
    medications = structured_json.get("medications_or_dosage_lines", [])
    keywords = structured_json.get("clinical_keywords", [])
    other_entities = structured_json.get("other_entities", [])

    summary_parts = []

    if patient_info:
        info_text = ", ".join([f"{key}: {value}" for key, value in patient_info.items()])
        summary_parts.append(f"Patient information found: {info_text}.")

    if lab_values:
        summary_parts.append(f"{len(lab_values)} lab value(s) were detected from the document.")

    if medications:
        summary_parts.append(f"{len(medications)} medication or dosage-related line(s) were detected.")

    if keywords:
        keyword_text = ", ".join(sorted(set([item.get("text", "") for item in keywords if item.get("text")])) )
        if keyword_text:
            summary_parts.append(f"Clinical keywords found: {keyword_text}.")

    if other_entities:
        summary_parts.append(f"{len(other_entities)} additional biomedical entity(ies) were detected.")

    if not summary_parts:
        summary_parts.append("The document was processed, but no strong structured clinical entities were detected.")

    return " ".join(summary_parts)


def run_advanced_document_pipeline(
    document_path: str,
    case_id: str = "case_001",
    ocr_engine: str = "tesseract",
    ner_engine: str = "rule",
    roi_mode: str = "opencv",
    yolo_weights: Optional[str] = None,
    max_pages: int = 5,
    log: Optional[Callable[[str], None]] = None,
) -> Dict:
    ocr_result = extract_text_with_engine(
        document_path=document_path,
        engine=ocr_engine,
        roi_mode=roi_mode,
        yolo_weights=yolo_weights,
        max_pages=max_pages,
        log=log,
    )
    cleaned_text = clean_ocr_text(ocr_result["text"])
    ner_result = extract_entities(cleaned_text, ner_engine=ner_engine, log=log)
    entities = ner_result["entities"]
    structured_json = build_structured_json(entities)
    patient_summary = generate_document_summary(structured_json)

    fallback_used = bool(ocr_result.get("fallback_used") or ner_result.get("fallback_used"))

    return {
        "case_id": case_id,
        "file": str(document_path),
        "raw_text": cleaned_text,
        "raw_text_preview": make_preview(cleaned_text),
        "entities": entities,
        "structured_json": structured_json,
        "patient_summary": patient_summary,
        "ocr_engine_requested": ocr_engine,
        "ocr_engine_used": ocr_result.get("engine_used", ocr_engine),
        "roi_mode_requested": roi_mode,
        "roi_mode_used": ocr_result.get("roi_mode_used", roi_mode),
        "ner_engine_requested": ner_engine,
        "ner_engine_used": ner_result.get("ner_engine_used", ner_engine),
        "fallback_used": fallback_used,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the experimental advanced Model-2 OCR pipeline.")
    parser.add_argument("--case-id", type=str, default="case_001", help="Unique case ID.")
    parser.add_argument("--document", type=str, required=False, help="Path to a document image, PDF, or TXT file.")
    parser.add_argument("--ocr-engine", type=str, choices=["tesseract", "trocr", "paddle"], default="tesseract")
    parser.add_argument("--ner-engine", type=str, choices=["rule", "biobert"], default="rule")
    parser.add_argument("--roi-mode", type=str, choices=["none", "opencv", "yolo"], default="opencv")
    parser.add_argument("--yolo-weights", type=str, default=None, help="Optional path to YOLO weights for ROI detection.")
    parser.add_argument("--max-pages", type=int, default=5, help="Maximum PDF pages to OCR.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output path for the resulting JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.document:
        print("No --document was provided. Nothing to process.")
        return

    result = run_advanced_document_pipeline(
        document_path=args.document,
        case_id=args.case_id,
        ocr_engine=args.ocr_engine,
        ner_engine=args.ner_engine,
        roi_mode=args.roi_mode,
        yolo_weights=args.yolo_weights,
        max_pages=args.max_pages,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Saved] {output_path}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
