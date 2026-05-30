import re
from typing import Dict, List


MEDICAL_KEYWORDS = [
    "fever",
    "cough",
    "pain",
    "headache",
    "vomiting",
    "nausea",
    "diabetes",
    "hypertension",
    "asthma",
    "infection",
    "tumor",
    "chest pain",
    "shortness of breath",
    "dizziness",
]


DRUG_HINTS = [
    "tab",
    "tablet",
    "cap",
    "capsule",
    "syrup",
    "inj",
    "injection",
    "mg",
    "ml",
    "dose",
    "daily",
    "bd",
    "tds",
    "od",
]


LAB_TEST_NAMES = [
    "hb",
    "hemoglobin",
    "wbc",
    "rbc",
    "platelet",
    "glucose",
    "cholesterol",
    "hdl",
    "ldl",
    "tg",
    "creatinine",
    "bun",
    "crp",
    "esr",
    "bilirubin",
    "alt",
    "ast",
]


def extract_patient_info(text: str) -> Dict:
    patient = {}

    name_match = re.search(
        r"(patient\s*name|name)\s*[:\-]\s*([A-Za-z .]+)",
        text,
        flags=re.IGNORECASE,
    )

    if name_match:
        patient["name"] = name_match.group(2).strip()

    age_match = re.search(
        r"(age)\s*[:\-]?\s*(\d{1,3})",
        text,
        flags=re.IGNORECASE,
    )

    if age_match:
        patient["age"] = age_match.group(2)

    gender_match = re.search(
        r"(sex|gender)\s*[:\-]?\s*(male|female|m|f)",
        text,
        flags=re.IGNORECASE,
    )

    if gender_match:
        patient["gender"] = gender_match.group(2)

    return patient


def extract_dates(text: str) -> List[Dict]:
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    ]

    entities = []

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            entities.append(
                {
                    "type": "DATE",
                    "text": match.group(0),
                }
            )

    return entities


def extract_lab_values(text: str) -> List[Dict]:
    entities = []

    for test_name in LAB_TEST_NAMES:
        pattern = rf"\b({test_name})\b\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z/%]+)?"

        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            entities.append(
                {
                    "type": "LAB_VALUE",
                    "name": match.group(1),
                    "value": match.group(2),
                    "unit": match.group(3) or "",
                    "text": match.group(0),
                }
            )

    return entities


def extract_drug_lines(text: str) -> List[Dict]:
    entities = []

    for line in text.splitlines():
        lower_line = line.lower()

        if any(hint in lower_line for hint in DRUG_HINTS):
            if len(line.strip()) > 3:
                entities.append(
                    {
                        "type": "MEDICATION_OR_DOSAGE_LINE",
                        "text": line.strip(),
                    }
                )

    return entities


def extract_symptoms(text: str) -> List[Dict]:
    entities = []
    lower_text = text.lower()

    for keyword in MEDICAL_KEYWORDS:
        if keyword in lower_text:
            entities.append(
                {
                    "type": "CLINICAL_KEYWORD",
                    "text": keyword,
                }
            )

    return entities


def extract_entities(text: str) -> List[Dict]:
    entities = []

    patient_info = extract_patient_info(text)

    for key, value in patient_info.items():
        entities.append(
            {
                "type": "PATIENT_INFO",
                "field": key,
                "text": value,
            }
        )

    entities.extend(extract_dates(text))
    entities.extend(extract_lab_values(text))
    entities.extend(extract_drug_lines(text))
    entities.extend(extract_symptoms(text))

    return entities