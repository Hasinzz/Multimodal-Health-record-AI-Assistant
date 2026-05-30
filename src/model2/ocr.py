from pathlib import Path
from typing import List
import os
import shutil

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def _configure_tesseract_cmd() -> None:
    # Priority: explicit env override -> PATH -> common Windows install path.
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


_configure_tesseract_cmd()


def preprocess_image_for_ocr(image_path: str) -> Image.Image:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(
        gray,
        d=5,
        sigmaColor=75,
        sigmaSpace=75,
    )

    thresholded = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    return Image.fromarray(thresholded)


def ocr_image(image_path: str) -> str:
    processed_image = preprocess_image_for_ocr(image_path)

    text = pytesseract.image_to_string(
        processed_image,
        config="--psm 6",
    )

    return text


def pdf_to_images(pdf_path: str, max_pages: int = 5) -> List[Image.Image]:
    document = fitz.open(pdf_path)

    images = []

    for page_index in range(min(len(document), max_pages)):
        page = document[page_index]
        pixmap = page.get_pixmap(dpi=200)

        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )

        images.append(image)

    document.close()

    return images


def ocr_pdf(pdf_path: str, max_pages: int = 5) -> str:
    images = pdf_to_images(pdf_path, max_pages=max_pages)

    all_text = []

    for index, image in enumerate(images):
        text = pytesseract.image_to_string(
            image,
            config="--psm 6",
        )

        all_text.append(f"\n--- Page {index + 1} ---\n{text}")

    return "\n".join(all_text)


def extract_text_from_file(file_path: str) -> str:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        return ocr_pdf(str(file_path))

    if suffix in IMAGE_EXTENSIONS:
        return ocr_image(str(file_path))

    raise ValueError(
        f"Unsupported document type: {suffix}. "
        "Use .txt, .pdf, .png, .jpg, .jpeg, .tif, or .tiff"
    )