from typing import Dict, List

from src.model2.clean_text import clean_ocr_text, make_preview
from src.model2.ner import extract_entities
from src.model2.ocr import extract_text_from_file


def build_structured_json(entities: List[Dict]) -> Dict:
    structured = {
        "patient_info": {},
        "dates": [],
        "lab_values": [],
        "medications_or_dosage_lines": [],
        "clinical_keywords": [],
    }

    for entity in entities:
        entity_type = entity.get("type")

        if entity_type == "PATIENT_INFO":
            field = entity.get("field")
            value = entity.get("text")

            if field:
                structured["patient_info"][field] = value

        elif entity_type == "DATE":
            structured["dates"].append(entity)

        elif entity_type == "LAB_VALUE":
            structured["lab_values"].append(entity)

        elif entity_type == "MEDICATION_OR_DOSAGE_LINE":
            structured["medications_or_dosage_lines"].append(entity)

        elif entity_type == "CLINICAL_KEYWORD":
            structured["clinical_keywords"].append(entity)

    return structured


def generate_document_summary(structured_json: Dict) -> str:
    patient_info = structured_json.get("patient_info", {})
    lab_values = structured_json.get("lab_values", [])
    medications = structured_json.get("medications_or_dosage_lines", [])
    keywords = structured_json.get("clinical_keywords", [])

    summary_parts = []

    if patient_info:
        info_text = ", ".join(
            [f"{key}: {value}" for key, value in patient_info.items()]
        )
        summary_parts.append(f"Patient information found: {info_text}.")

    if lab_values:
        summary_parts.append(
            f"{len(lab_values)} lab value(s) were detected from the document."
        )

    if medications:
        summary_parts.append(
            f"{len(medications)} medication or dosage-related line(s) were detected."
        )

    if keywords:
        keyword_text = ", ".join(
            sorted(set([item["text"] for item in keywords]))
        )
        summary_parts.append(f"Clinical keywords found: {keyword_text}.")

    if not summary_parts:
        summary_parts.append(
            "The document was processed, but no strong structured clinical entities were detected."
        )

    return " ".join(summary_parts)


def run_document_pipeline(
    document_path: str,
    case_id: str = "case_001",
) -> Dict:
    raw_text = extract_text_from_file(document_path)
    cleaned_text = clean_ocr_text(raw_text)
    entities = extract_entities(cleaned_text)
    structured_json = build_structured_json(entities)
    patient_summary = generate_document_summary(structured_json)

    return {
        "case_id": case_id,
        "file": str(document_path),
        "raw_text": cleaned_text,
        "raw_text_preview": make_preview(cleaned_text),
        "entities": entities,
        "structured_json": structured_json,
        "patient_summary": patient_summary,
    }