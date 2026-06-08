from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.config import KB_DIR, PROJECT_ROOT
from src.model2.advanced_pipeline import run_advanced_document_pipeline
from src.model2.pipeline import run_document_pipeline
from src.model3.pipeline import run_fusion_pipeline


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v4_advanced_improvement" / "comparison"
V4_YOLO_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "model2"
    / "yolo_roi_v4"
    / "yolov8n_roi_v4_pseudolabel_best.pt"
)
V4_BERT_NER_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "model2" / "biobert_ner_v4"
V4_RAG_KB_DIR = PROJECT_ROOT / "data" / "rag_kb_v4"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def first_image(paths: Iterable[Path]) -> Optional[Path]:
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                return path
    return None


def default_lab_report() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "roi_yolo_v4" / "unlabeled_for_annotation",
        PROJECT_ROOT / "data" / "improvement" / "model2_lab_reports" / "lbmaske",
        PROJECT_ROOT / "data" / "documents" / "lab_reports",
    ]
    path = first_image(candidates)
    if path is None:
        raise FileNotFoundError("No lab report image found for V4 comparison.")
    return path


def entity_types(output: Dict) -> Dict[str, int]:
    entities = output.get("extracted_entities") or output.get("entities") or []
    return dict(Counter(str(entity.get("type", "UNKNOWN")) for entity in entities))


def run_stable(document_path: Path) -> tuple[Dict, Dict, float]:
    start = time.perf_counter()
    model2_output = run_document_pipeline(
        document_path=str(document_path),
        case_id="v4_comparison_stable",
    )
    fusion_output = run_fusion_pipeline(
        case_id="v4_comparison_stable",
        model2_output=model2_output,
        kb_dir=str(KB_DIR),
    )
    runtime = time.perf_counter() - start
    return model2_output, fusion_output, runtime


def run_v4(document_path: Path) -> tuple[Dict, Dict, float]:
    start = time.perf_counter()
    yolo_weights = str(V4_YOLO_CHECKPOINT) if V4_YOLO_CHECKPOINT.exists() else None
    bert_checkpoint = str(V4_BERT_NER_CHECKPOINT) if V4_BERT_NER_CHECKPOINT.exists() else None
    model2_output = run_advanced_document_pipeline(
        document_path=str(document_path),
        case_id="v4_comparison_advanced",
        ocr_engine="tesseract",
        ner_engine="biobert" if bert_checkpoint else "rule",
        roi_mode="yolo" if yolo_weights else "opencv",
        yolo_weights=yolo_weights,
        biobert_checkpoint_path=bert_checkpoint,
        mode_used="v4_advanced",
    )
    fusion_output = run_fusion_pipeline(
        case_id="v4_comparison_advanced",
        model2_output=model2_output,
        kb_dir=str(V4_RAG_KB_DIR if V4_RAG_KB_DIR.exists() else KB_DIR),
    )
    runtime = time.perf_counter() - start
    return model2_output, fusion_output, runtime


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(
    document_path: Path,
    stable_model2: Dict,
    stable_fusion: Dict,
    stable_runtime: float,
    v4_model2: Dict,
    v4_fusion: Dict,
    v4_runtime: float,
) -> None:
    stable_entities = stable_model2.get("entities", [])
    v4_entities = v4_model2.get("extracted_entities") or v4_model2.get("entities", [])
    lines = [
        "# V4 Comparison Summary",
        "",
        f"Document: `{document_path}`",
        "",
        "## Stable Pipeline",
        "",
        f"- OCR text length: {len(stable_model2.get('raw_text', ''))}",
        f"- Number of entities extracted: {len(stable_entities)}",
        f"- Entity types detected: {entity_types(stable_model2)}",
        f"- KB used: `{stable_fusion.get('kb_used')}`",
        f"- Retrieved evidence count: {len(stable_fusion.get('retrieved_evidence', []))}",
        f"- Runtime seconds: {stable_runtime:.2f}",
        "",
        "## V4 Advanced Pipeline",
        "",
        f"- OCR text length: {len(v4_model2.get('raw_text', ''))}",
        f"- Number of entities extracted: {len(v4_entities)}",
        f"- Entity types detected: {entity_types(v4_model2)}",
        f"- Mode used: {v4_model2.get('mode_used')}",
        f"- ROI mode used: {v4_model2.get('roi_mode_used')}",
        f"- YOLO checkpoint used: `{v4_model2.get('yolo_checkpoint_used')}`",
        f"- NER engine used: {v4_model2.get('ner_engine_used')}",
        f"- BERT checkpoint used: `{v4_model2.get('bert_checkpoint_used')}`",
        f"- KB used: `{v4_fusion.get('kb_used')}`",
        f"- Retrieved evidence count: {len(v4_fusion.get('retrieved_evidence', []))}",
        f"- Runtime seconds: {v4_runtime:.2f}",
        "",
        "## Summary Quality Notes",
        "",
        "- Stable pipeline is the thesis baseline and remains unchanged.",
        "- V4 uses pseudo-labeled YOLO ROI and weak-label BERT NER as optional experimental improvements.",
        "- Compare OCR length and entity count carefully; more entities does not automatically mean better clinical extraction.",
        "",
        "## Limitations",
        "",
        "- YOLO ROI V4 was trained on pseudo-labels, not manual ground-truth boxes.",
        "- BERT NER V4 was trained on weak labels, not expert clinical annotations.",
        "- RAG evidence quality should be checked manually query by query.",
        "- This is technical validation, not clinical validation.",
    ]
    (OUTPUT_DIR / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stable document pipeline against V4 advanced mode.")
    parser.add_argument("--lab-report", type=Path, default=None)
    parser.add_argument("--prescription", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document_path = args.lab_report or args.prescription or default_lab_report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stable_model2, stable_fusion, stable_runtime = run_stable(document_path)
    v4_model2, v4_fusion, v4_runtime = run_v4(document_path)

    write_json(
        OUTPUT_DIR / "stable_output.json",
        {
            "document_path": str(document_path),
            "model2_output": stable_model2,
            "fusion_output": stable_fusion,
            "runtime_seconds": stable_runtime,
        },
    )
    write_json(
        OUTPUT_DIR / "v4_output.json",
        {
            "document_path": str(document_path),
            "model2_output": v4_model2,
            "fusion_output": v4_fusion,
            "runtime_seconds": v4_runtime,
        },
    )
    write_summary(
        document_path=document_path,
        stable_model2=stable_model2,
        stable_fusion=stable_fusion,
        stable_runtime=stable_runtime,
        v4_model2=v4_model2,
        v4_fusion=v4_fusion,
        v4_runtime=v4_runtime,
    )

    print(f"Document: {document_path}")
    print(f"Wrote {OUTPUT_DIR / 'stable_output.json'}")
    print(f"Wrote {OUTPUT_DIR / 'v4_output.json'}")
    print(f"Wrote {OUTPUT_DIR / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
