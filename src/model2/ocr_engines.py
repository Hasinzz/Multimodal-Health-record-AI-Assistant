from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Optional

import fitz
import numpy as np
import pytesseract
from PIL import Image

from src.model2.roi import detect_roi


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _log(log: Optional[Callable[[str], None]], message: str) -> None:
    if log is not None:
        log(message)


def configure_tesseract() -> None:
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    which_cmd = shutil.which("tesseract")
    if which_cmd:
        pytesseract.pytesseract.tesseract_cmd = which_cmd
        return

    windows_default = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    if windows_default.exists():
        pytesseract.pytesseract.tesseract_cmd = str(windows_default)


configure_tesseract()


def _load_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _ocr_tesseract_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image, config="--psm 6")


def _ocr_trocr_image(image: Image.Image) -> str:
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except Exception as exc:
        raise RuntimeError(f"TrOCR dependencies unavailable: {exc}") from exc

    model_name = os.environ.get("TROCR_MODEL_NAME", "microsoft/trocr-base-printed")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def _ocr_paddle_image(image: Image.Image) -> str:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(f"PaddleOCR unavailable: {exc}") from exc

    ocr = PaddleOCR(use_angle_cls=True, lang=os.environ.get("PADDLE_OCR_LANG", "en"), show_log=False)
    result = ocr.ocr(np.array(image), cls=True)
    lines = []
    for page in result or []:
        for line in page or []:
            text = line[1][0] if line and len(line) > 1 and line[1] else ""
            if text:
                lines.append(text)
    return "\n".join(lines)


def _render_pdf(pdf_path: Path, max_pages: int = 5) -> list[Image.Image]:
    document = fitz.open(str(pdf_path))
    images: list[Image.Image] = []
    try:
        for page_index in range(min(len(document), max_pages)):
            page = document[page_index]
            pixmap = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            images.append(image)
    finally:
        document.close()
    return images


def _extract_text_from_txt(text_path: Path) -> str:
    return text_path.read_text(encoding="utf-8", errors="ignore")


def _run_engine_on_image(image: Image.Image, engine: str) -> tuple[str, str, bool]:
    engine = (engine or "tesseract").lower()
    if engine == "tesseract":
        return _ocr_tesseract_image(image), "tesseract", False

    if engine == "trocr":
        try:
            return _ocr_trocr_image(image), "trocr", False
        except Exception:
            return _ocr_tesseract_image(image), "tesseract", True

    if engine == "paddle":
        try:
            return _ocr_paddle_image(image), "paddle", False
        except Exception:
            return _ocr_tesseract_image(image), "tesseract", True

    return _ocr_tesseract_image(image), "tesseract", True


def extract_text_with_engine(
    document_path: str | Path,
    engine: str = "tesseract",
    roi_mode: str = "opencv",
    yolo_weights: Optional[str] = None,
    max_pages: int = 5,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    document_path = Path(document_path)
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    suffix = document_path.suffix.lower()
    fallback_used = False
    page_texts: list[str] = []
    used_engine = (engine or "tesseract").lower()
    used_roi_mode = (roi_mode or "none").lower()
    yolo_weights_used = None

    if suffix == ".txt":
        raw_text = _extract_text_from_txt(document_path)
        return {
            "text": raw_text,
            "engine_used": "text",
            "roi_mode_used": "none",
            "yolo_weights_used": None,
            "fallback_used": False,
        }

    if suffix == ".pdf":
        images = _render_pdf(document_path, max_pages=max_pages)
    elif suffix in IMAGE_EXTENSIONS:
        images = [_load_image(Image.open(document_path))]
    else:
        raise ValueError(f"Unsupported document type: {suffix}")

    for image in images:
        roi_result = detect_roi(image, mode=used_roi_mode, yolo_weights=yolo_weights, log=log)
        used_roi_mode = roi_result.roi_mode_used if roi_result.roi_mode_used else used_roi_mode
        if roi_result.yolo_weights_used:
            yolo_weights_used = roi_result.yolo_weights_used
        fallback_used = fallback_used or roi_result.fallback_used
        text, engine_used, engine_fallback = _run_engine_on_image(roi_result.image, used_engine)
        used_engine = engine_used
        fallback_used = fallback_used or engine_fallback
        page_texts.append(text)

    combined_text = []
    for index, page_text in enumerate(page_texts, start=1):
        combined_text.append(f"\n--- Page {index} ---\n{page_text}")

    return {
        "text": "\n".join(combined_text).strip(),
        "engine_used": used_engine,
        "roi_mode_used": used_roi_mode,
        "yolo_weights_used": yolo_weights_used,
        "fallback_used": fallback_used,
    }


def extract_text_from_file(
    file_path: str | Path,
    engine: str = "tesseract",
    roi_mode: str = "none",
    yolo_weights: Optional[str] = None,
    max_pages: int = 5,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    result = extract_text_with_engine(
        document_path=file_path,
        engine=engine,
        roi_mode=roi_mode,
        yolo_weights=yolo_weights,
        max_pages=max_pages,
        log=log,
    )
    return result["text"]
