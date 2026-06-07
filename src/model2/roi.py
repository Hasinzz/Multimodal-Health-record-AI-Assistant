from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class ROIResult:
    image: Image.Image
    mode_used: str
    fallback_used: bool
    message: str


def _log(log: Optional[Callable[[str], None]], message: str) -> None:
    if log is not None:
        log(message)


def _load_pil_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _opencv_roi(image: Image.Image) -> Tuple[Image.Image, str]:
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, "opencv_no_contours"

    boxes = []
    height, width = gray.shape[:2]
    min_area = max(300, int(0.002 * width * height))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area and w > 10 and h > 10:
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        return image, "opencv_no_boxes"

    x1 = max(0, min(box[0] for box in boxes))
    y1 = max(0, min(box[1] for box in boxes))
    x2 = min(width, max(box[2] for box in boxes))
    y2 = min(height, max(box[3] for box in boxes))

    if x2 <= x1 or y2 <= y1:
        return image, "opencv_invalid_box"

    return image.crop((x1, y1, x2, y2)), f"opencv_crop_{x1}_{y1}_{x2}_{y2}"


def _yolo_roi(image: Image.Image, yolo_weights: Optional[str], log: Optional[Callable[[str], None]]) -> Tuple[Image.Image, str, bool]:
    if not yolo_weights:
        _log(log, "[ROI] YOLO weights were not provided; falling back to OpenCV ROI.")
        cropped, detail = _opencv_roi(image)
        return cropped, f"opencv_fallback_{detail}", True

    weights_path = Path(yolo_weights)
    if not weights_path.exists():
        _log(log, f"[ROI] YOLO weights not found at {weights_path}; falling back to OpenCV ROI.")
        cropped, detail = _opencv_roi(image)
        return cropped, f"opencv_fallback_{detail}", True

    try:
        from ultralytics import YOLO
    except Exception as exc:
        _log(log, f"[ROI] ultralytics is unavailable ({exc}); falling back to OpenCV ROI.")
        cropped, detail = _opencv_roi(image)
        return cropped, f"opencv_fallback_{detail}", True

    try:
        model = YOLO(str(weights_path))
        result = model.predict(image, verbose=False)
    except Exception as exc:
        _log(log, f"[ROI] YOLO inference failed ({exc}); falling back to OpenCV ROI.")
        cropped, detail = _opencv_roi(image)
        return cropped, f"opencv_fallback_{detail}", True

    if not result:
        _log(log, "[ROI] YOLO returned no detections; falling back to OpenCV ROI.")
        cropped, detail = _opencv_roi(image)
        return cropped, f"opencv_fallback_{detail}", True

    boxes = result[0].boxes
    if boxes is None or len(boxes) == 0:
        _log(log, "[ROI] YOLO returned empty boxes; falling back to OpenCV ROI.")
        cropped, detail = _opencv_roi(image)
        return cropped, f"opencv_fallback_{detail}", True

    xyxy = boxes.xyxy.cpu().numpy()
    width, height = image.size
    x1 = max(0, int(np.floor(np.min(xyxy[:, 0]))))
    y1 = max(0, int(np.floor(np.min(xyxy[:, 1]))))
    x2 = min(width, int(np.ceil(np.max(xyxy[:, 2]))))
    y2 = min(height, int(np.ceil(np.max(xyxy[:, 3]))))

    if x2 <= x1 or y2 <= y1:
        _log(log, "[ROI] YOLO produced an invalid crop; falling back to OpenCV ROI.")
        cropped, detail = _opencv_roi(image)
        return cropped, f"opencv_fallback_{detail}", True

    return image.crop((x1, y1, x2, y2)), f"yolo_crop_{x1}_{y1}_{x2}_{y2}", False


def detect_roi(
    image: Image.Image | str | Path,
    mode: str = "opencv",
    yolo_weights: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> ROIResult:
    pil_image = _load_pil_image(image)
    mode = (mode or "none").lower()

    if mode == "none":
        return ROIResult(
            image=pil_image,
            mode_used="none",
            fallback_used=False,
            message="Full image used without ROI cropping.",
        )

    if mode == "opencv":
        cropped, detail = _opencv_roi(pil_image)
        fallback_used = detail.startswith("opencv_no") or detail.startswith("opencv_invalid")
        return ROIResult(
            image=cropped,
            mode_used="opencv",
            fallback_used=fallback_used,
            message=detail,
        )

    if mode == "yolo":
        cropped, detail, fallback_used = _yolo_roi(pil_image, yolo_weights, log)
        return ROIResult(
            image=cropped,
            mode_used="yolo" if not fallback_used else "opencv",
            fallback_used=fallback_used,
            message=detail,
        )

    _log(log, f"[ROI] Unknown ROI mode '{mode}'. Falling back to full image.")
    return ROIResult(
        image=pil_image,
        mode_used="none",
        fallback_used=True,
        message=f"unknown_mode:{mode}",
    )
