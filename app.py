from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import DEFAULT_BRAIN_CHECKPOINT, DEFAULT_XRAY_CHECKPOINT, DEFAULT_XRAY_THRESHOLDS, KB_DIR, PROJECT_ROOT, create_required_folders
from src.model1.infer import predict_image
from src.model2.advanced_pipeline import run_advanced_document_pipeline
from src.model2.pipeline import run_document_pipeline
from src.model3.advanced_fusion import run_advanced_fusion_pipeline
from src.model3.pipeline import run_fusion_pipeline

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional dependency
    st = None


def _save_upload(uploaded_file, prefix: str) -> Path:
    upload_dir = PROJECT_ROOT / "outputs" / "streamlit_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=f"{prefix}_", dir=upload_dir) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def _maybe_get_streamlit():
    if st is None:
        print("Streamlit is not installed. Add it to requirements.txt and run `streamlit run app.py` after installation.")
        return None
    return st


def _render_value(label: str, value: Any) -> None:
    if value is None:
        return
    st.subheader(label)
    if isinstance(value, (dict, list)):
        st.json(value)
    else:
        st.write(value)


def _run_document_analysis(
    document_path: Path,
    case_id: str,
    document_kind: str,
    ocr_engine: str,
    ner_engine: str,
    roi_mode: str,
    yolo_weights: Optional[str],
) -> Dict[str, Any]:
    if ocr_engine == "tesseract" and ner_engine == "rule" and roi_mode == "none":
        return run_document_pipeline(document_path=str(document_path), case_id=case_id)

    return run_advanced_document_pipeline(
        document_path=str(document_path),
        case_id=case_id,
        ocr_engine=ocr_engine,
        ner_engine=ner_engine,
        roi_mode=roi_mode,
        yolo_weights=yolo_weights,
    )


def main() -> None:
    streamlit = _maybe_get_streamlit()
    if streamlit is None:
        return

    create_required_folders()
    st.set_page_config(page_title="Multimodal AI Assistant", layout="wide")
    st.title("Multimodal AI Assistant for Automated Health Record Summarization and Feedback Analysis")
    st.caption("Stable baseline first, with optional experimental components for comparison.")

    with st.sidebar:
        st.header("Analysis Options")
        case_id = st.text_input("Case ID", value="case_001")
        image_modality = st.selectbox("Image modality", ["brain_mri", "xray"])
        document_kind = st.selectbox("Document type", ["optional", "prescription", "lab_report"])
        ocr_engine = st.selectbox("OCR engine", ["tesseract", "trocr", "paddle"], index=0)
        ner_engine = st.selectbox("NER engine", ["rule", "biobert"], index=0)
        roi_mode = st.selectbox("ROI mode", ["none", "opencv", "yolo"], index=1)
        fusion_mode = st.selectbox("Fusion mode", ["stable", "advanced"], index=0)
        yolo_weights = st.text_input("YOLO weights path (optional)", value="")
        use_clahe = st.checkbox("Use CLAHE for X-ray", value=False)
        use_n4 = st.checkbox("Use N4 for brain MRI", value=False)

    image_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    document_file = st.file_uploader("Upload document", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "txt"])

    run_button = st.button("Run analysis", type="primary")

    if not run_button:
        st.info("Upload an image and/or document, choose options, then run the analysis.")
        return

    case_output_dir = PROJECT_ROOT / "outputs" / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    model1_output: Optional[Dict[str, Any]] = None
    model2_output: Optional[Dict[str, Any]] = None

    if image_file is not None:
        image_path = _save_upload(image_file, f"{case_id}_image")
        checkpoint_path = DEFAULT_XRAY_CHECKPOINT if image_modality == "xray" else DEFAULT_BRAIN_CHECKPOINT
        thresholds_path = str(DEFAULT_XRAY_THRESHOLDS) if image_modality == "xray" else None
        embedding_output_path = case_output_dir / f"{case_id}_{image_modality}_embedding.npy"
        model1_output = predict_image(
            image_path=str(image_path),
            modality=image_modality,
            checkpoint_path=str(checkpoint_path),
            case_id=case_id,
            embedding_output_path=str(embedding_output_path),
            thresholds_path=thresholds_path,
            use_clahe=use_clahe,
            use_n4=use_n4,
        )
        _render_value("Image prediction", model1_output.get("patient_summary_text"))
        if image_modality == "xray":
            _render_value("X-ray tuned threshold results", model1_output.get("xray_positive_labels"))
        _render_value("Model-1 output", model1_output)

    if document_file is not None:
        document_path = _save_upload(document_file, f"{case_id}_document")
        yolo_weights_path = yolo_weights.strip() or None
        model2_output = _run_document_analysis(
            document_path=document_path,
            case_id=case_id,
            document_kind=document_kind,
            ocr_engine=ocr_engine,
            ner_engine=ner_engine,
            roi_mode=roi_mode,
            yolo_weights=yolo_weights_path,
        )
        _render_value("OCR extracted text", model2_output.get("raw_text_preview"))
        _render_value("Extracted entities", model2_output.get("entities"))
        _render_value("Patient summary", model2_output.get("patient_summary"))
        _render_value("Model-2 output", model2_output)

    if fusion_mode == "advanced":
        fusion_output = run_advanced_fusion_pipeline(
            case_id=case_id,
            model1_output=model1_output,
            model2_output=model2_output,
            kb_dir=str(KB_DIR),
            fusion_mode="advanced",
        )
    else:
        fusion_output = run_fusion_pipeline(
            case_id=case_id,
            model1_output=model1_output,
            model2_output=model2_output,
            kb_dir=str(KB_DIR),
        )

    _render_value("Doctor feedback", fusion_output.get("doctor_feedback"))
    _render_value("Retrieved evidence", fusion_output.get("retrieved_evidence"))
    _render_value("Fusion output", fusion_output)

    download_payload = json.dumps(
        {
            "model1_output": model1_output,
            "model2_output": model2_output,
            "model3_output": fusion_output,
        },
        indent=2,
        ensure_ascii=False,
    )
    st.download_button(
        label="Download JSON",
        data=download_payload,
        file_name=f"{case_id}_multimodal_output.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
