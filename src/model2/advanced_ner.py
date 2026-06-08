from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional

from src.model2.ner import extract_entities as extract_rule_entities

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIOBERT_V4_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "model2" / "biobert_ner_v4"


def _log(log: Optional[Callable[[str], None]], message: str) -> None:
    if log is not None:
        log(message)


def _normalize_biobert_entity(entity: dict) -> dict:
    text = str(entity.get("text") or entity.get("word") or "").strip()
    label = str(entity.get("label", entity.get("entity_group", entity.get("entity", "ENTITY")))).upper()
    score = entity.get("score")
    if hasattr(score, "item"):
        score = score.item()

    if not text:
        return {}

    date_match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", text)
    if date_match:
        return {"type": "DATE", "text": date_match.group(0)}

    if label in {"PER", "PERSON", "PATIENT", "NAME"}:
        return {"type": "PATIENT_INFO", "field": "name", "text": text}

    if label in {"DATE", "TIME"}:
        return {"type": "DATE", "text": text}

    if label in {"ORG", "HOSPITAL", "FACILITY"}:
        return {"type": "CLINICAL_KEYWORD", "text": text}

    return {"type": "BIOBERT_ENTITY", "label": label, "text": text, "score": score}


def _is_usable_transformers_checkpoint(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / "config.json").exists()


def _run_biobert_ner(text: str, biobert_checkpoint_path: Optional[str] = None) -> tuple[list[dict], Optional[str]]:
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(f"transformers unavailable: {exc}") from exc

    model_candidates: list[tuple[str, Optional[str]]] = []

    explicit_checkpoint = Path(biobert_checkpoint_path) if biobert_checkpoint_path else None
    if explicit_checkpoint and _is_usable_transformers_checkpoint(explicit_checkpoint):
        model_candidates.append((str(explicit_checkpoint), str(explicit_checkpoint)))

    env_local = os.environ.get("BIOBERT_NER_MODEL_PATH")
    if env_local and _is_usable_transformers_checkpoint(Path(env_local)):
        model_candidates.append((env_local, env_local))

    if not explicit_checkpoint:
        if _is_usable_transformers_checkpoint(DEFAULT_BIOBERT_V4_CHECKPOINT):
            model_candidates.append(
                (str(DEFAULT_BIOBERT_V4_CHECKPOINT), str(DEFAULT_BIOBERT_V4_CHECKPOINT))
            )

    env_model = os.environ.get("BIOBERT_NER_MODEL")
    if env_model:
        model_candidates.append((env_model, None))

    model_candidates.extend(
        [
            ("d4data/biomedical-ner-all", None),
            ("Clinical-AI-Apollo/Medical-NER", None),
        ]
    )

    last_error: Optional[Exception] = None
    for model_name, checkpoint_used in model_candidates:
        try:
            ner_pipeline = pipeline(
                task="token-classification",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
            )
            raw_entities = ner_pipeline(text)
            normalized = []
            for entity in raw_entities:
                normalized_entity = _normalize_biobert_entity(entity)
                if normalized_entity:
                    normalized.append(normalized_entity)
            return normalized, checkpoint_used
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"No biomedical NER model could be loaded: {last_error}")


def extract_entities(
    text: str,
    ner_engine: str = "rule",
    biobert_checkpoint_path: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    ner_engine = (ner_engine or "rule").lower()

    if ner_engine == "rule":
        return {
            "entities": extract_rule_entities(text),
            "ner_engine_used": "rule",
            "biobert_checkpoint_used": None,
            "fallback_used": False,
        }

    if ner_engine == "biobert":
        try:
            entities, checkpoint_used = _run_biobert_ner(
                text,
                biobert_checkpoint_path=biobert_checkpoint_path,
            )
            return {
                "entities": entities,
                "ner_engine_used": "biobert",
                "biobert_checkpoint_used": checkpoint_used,
                "fallback_used": False,
            }
        except Exception as exc:
            _log(log, f"[NER] BioBERT NER unavailable; falling back to rule-based extraction. Reason: {exc}")
            return {
                "entities": extract_rule_entities(text),
                "ner_engine_used": "rule",
                "biobert_checkpoint_used": None,
                "fallback_used": True,
            }

    _log(log, f"[NER] Unknown NER engine '{ner_engine}'. Falling back to rule-based extraction.")
    return {
        "entities": extract_rule_entities(text),
        "ner_engine_used": "rule",
        "biobert_checkpoint_used": None,
        "fallback_used": True,
    }
