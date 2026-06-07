import json
import importlib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

from src.config import BRAIN_CLASSES, XRAY_CLASSES


_N4_WARNING_EMITTED = False


class TimmWithFeatures(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        prefixes = [
            "module.",
            "model.",
            "net.",
        ]

        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        if new_key.startswith("head.1."):
            new_key = "classifier." + new_key[len("head.1."):]
        elif new_key.startswith("backbone.head.1."):
            new_key = "classifier." + new_key[len("backbone.head.1."):]
        elif new_key.startswith("fc."):
            new_key = "classifier." + new_key[len("fc."):]
        elif new_key.startswith("backbone.fc."):
            new_key = "classifier." + new_key[len("backbone.fc."):]

        if not new_key.startswith("backbone.") and not new_key.startswith("classifier."):
            new_key = "backbone." + new_key

        cleaned[new_key] = value

    return cleaned


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        possible_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
        ]

        for key in possible_keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def load_image_model(
    checkpoint_path: str,
    modality: str,
    backbone_name: str = "densenet121",
    device: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if modality == "xray":
        num_classes = len(XRAY_CLASSES)
    elif modality == "brain_mri":
        num_classes = len(BRAIN_CLASSES)
    else:
        raise ValueError("modality must be either 'xray' or 'brain_mri'")

    model = TimmWithFeatures(backbone_name=backbone_name, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    state_dict = clean_state_dict_keys(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"[Model-1] Loaded checkpoint: {checkpoint_path}")
    print(f"[Model-1] Backbone: {backbone_name}")
    print(f"[Model-1] Device: {device}")
    print(f"[Model-1] Missing keys: {len(missing_keys)}")
    print(f"[Model-1] Unexpected keys: {len(unexpected_keys)}")

    if len(missing_keys) > 20 or len(unexpected_keys) > 20:
        print("[Warning] Large checkpoint mismatch found. Prediction may not be scientifically reliable.")

    model.to(device)
    model.eval()

    return model, device


def load_thresholds(thresholds_path: str) -> Dict[str, float]:
    path = Path(thresholds_path)

    if not path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict) and isinstance(data.get("thresholds"), dict):
        data = data["thresholds"]

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported threshold file format: {path}")

    cleaned: Dict[str, float] = {}
    for k, v in data.items():
        try:
            cleaned[str(k)] = float(v)
        except Exception:
            continue

    return cleaned


def build_xray_thresholds(
    class_names: List[str],
    thresholds_path: Optional[str],
) -> Tuple[Dict[str, float], str]:
    default_thresholds = {class_name: 0.5 for class_name in class_names}
    if thresholds_path is None:
        return default_thresholds, "default_0.5"

    try:
        loaded_thresholds = load_thresholds(thresholds_path)
    except Exception:
        return default_thresholds, "default_0.5"

    if not loaded_thresholds:
        return default_thresholds, "default_0.5"

    thresholds_used = {
        class_name: float(loaded_thresholds.get(class_name, 0.5))
        for class_name in class_names
    }

    return thresholds_used, "tuned"


def apply_clahe_rgb(image: Image.Image) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced_rgb)


def apply_n4_bias_correction_rgb(image: Image.Image) -> Image.Image:
    global _N4_WARNING_EMITTED

    try:
        sitk = importlib.import_module("SimpleITK")
    except Exception:
        if not _N4_WARNING_EMITTED:
            warnings.warn(
                "SimpleITK is not installed; skipping N4 bias correction.",
                RuntimeWarning,
            )
            _N4_WARNING_EMITTED = True
        return image

    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    sitk_image = sitk.GetImageFromArray(gray.astype(np.float32))
    mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    try:
        corrected = corrector.Execute(sitk_image, mask)
    except Exception:
        if not _N4_WARNING_EMITTED:
            warnings.warn(
                "N4 bias correction failed; continuing without correction.",
                RuntimeWarning,
            )
            _N4_WARNING_EMITTED = True
        return image

    corrected_np = sitk.GetArrayFromImage(corrected)
    corrected_np = cv2.normalize(corrected_np, None, 0, 255, cv2.NORM_MINMAX)
    corrected_np = corrected_np.astype(np.uint8)
    corrected_rgb = cv2.cvtColor(corrected_np, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(corrected_rgb)


def preprocess_image(
    image_path: str,
    modality: str,
    use_clahe: bool = False,
    use_n4: bool = False,
    image_size: int = 224,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")

    if modality == "xray" and use_clahe:
        image = apply_clahe_rgb(image)

    if modality == "brain_mri" and use_n4:
        image = apply_n4_bias_correction_rgb(image)

    image = image.resize((image_size, image_size))

    image_np = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image_np = (image_np - mean) / std

    tensor = torch.from_numpy(image_np)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return tensor


def format_prediction_summary(
    modality: str,
    top_predictions: List[Dict[str, float]],
    positive_labels: Optional[List[str]] = None,
    probabilities: Optional[Dict[str, float]] = None,
) -> str:
    if not top_predictions:
        return "No image prediction was generated."

    if modality == "xray":
        if positive_labels:
            selected = positive_labels
            if probabilities:
                findings = ", ".join(
                    [f"{label} ({probabilities.get(label, 0.0):.2f})" for label in selected]
                )
            else:
                findings = ", ".join(selected)
            return f"Chest X-ray suggests possible findings: {findings}."

        return "No X-ray disease label exceeded the tuned decision threshold."

    if modality == "brain_mri":
        top = top_predictions[0]
        return (
            f"Brain MRI suggests: {top['label']} "
            f"with confidence {top['probability']:.2f}."
        )

    return "Image summary unavailable."


def predict_image(
    image_path: str,
    modality: str,
    checkpoint_path: str,
    backbone_name: str = "densenet121",
    case_id: str = "case_001",
    embedding_output_path: Optional[str] = None,
    thresholds_path: Optional[str] = None,
    use_clahe: bool = False,
    use_n4: bool = False,
    image_size: int = 224,
) -> Dict:
    model, device = load_image_model(
        checkpoint_path=checkpoint_path,
        modality=modality,
        backbone_name=backbone_name,
    )

    tensor = preprocess_image(
        image_path=image_path,
        modality=modality,
        use_clahe=use_clahe,
        use_n4=use_n4,
        image_size=image_size,
    )
    tensor = tensor.to(device)

    with torch.no_grad():
        logits, features = model(tensor)

        if modality == "xray":
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
            class_names = XRAY_CLASSES
            top_indices = np.argsort(probs)[::-1][:5]

        elif modality == "brain_mri":
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            class_names = BRAIN_CLASSES
            top_indices = np.argsort(probs)[::-1]

        else:
            raise ValueError("modality must be either 'xray' or 'brain_mri'")

    xray_probabilities = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }

    xray_thresholds_used: Dict[str, float] = {}
    xray_positive_labels: List[str] = []
    xray_binary_predictions: Dict[str, int] = {}
    xray_threshold_mode = "default_0.5"

    if modality == "xray":
        xray_thresholds_used, xray_threshold_mode = build_xray_thresholds(
            class_names=list(class_names),
            thresholds_path=thresholds_path,
        )

        for i, name in enumerate(class_names):
            thr = float(xray_thresholds_used.get(name, 0.5))
            is_positive = float(probs[i]) >= thr
            xray_binary_predictions[name] = int(is_positive)
            if is_positive:
                xray_positive_labels.append(name)

    predicted_labels = list(xray_positive_labels)

    top_predictions = [
        {
            "label": class_names[i],
            "probability": float(probs[i]),
        }
        for i in top_indices
    ]

    embedding_path_value = None

    if embedding_output_path is not None:
        embedding_output_path = Path(embedding_output_path)
        embedding_output_path.parent.mkdir(parents=True, exist_ok=True)

        features_np = features[0].detach().cpu().numpy()
        np.save(embedding_output_path, features_np)

        embedding_path_value = str(embedding_output_path)

    patient_summary_text = format_prediction_summary(
        modality=modality,
        top_predictions=top_predictions,
        positive_labels=xray_positive_labels if modality == "xray" else None,
        probabilities=xray_probabilities if modality == "xray" else None,
    )

    return {
        "case_id": case_id,
        "modality": modality,
        "image_path": str(image_path),
        "top_predictions": top_predictions,
        "probabilities": xray_probabilities if modality == "xray" else {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        },
        "embedding_path": embedding_path_value,
        "patient_summary_text": patient_summary_text,
        "predicted_labels": predicted_labels,
        "thresholds_used": xray_thresholds_used if modality == "xray" else None,
        "xray_threshold_mode": xray_threshold_mode if modality == "xray" else None,
        "xray_thresholds_used": xray_thresholds_used if modality == "xray" else None,
        "xray_positive_labels": xray_positive_labels if modality == "xray" else None,
        "xray_probabilities": xray_probabilities if modality == "xray" else None,
        "xray_binary_predictions": xray_binary_predictions if modality == "xray" else None,
        "use_clahe": use_clahe,
        "use_n4": use_n4,
    }