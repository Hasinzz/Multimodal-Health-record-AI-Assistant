from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

from src.config import BRAIN_CLASSES, XRAY_CLASSES


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

    # Ensure keys are strings and values are floats
    cleaned: Dict[str, float] = {}
    for k, v in data.items():
        try:
            cleaned[str(k)] = float(v)
        except Exception:
            continue

    return cleaned


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


def preprocess_image(image_path: str, modality: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")

    if modality == "xray":
        image = apply_clahe_rgb(image)

    image = image.resize((224, 224))

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
) -> str:
    if not top_predictions:
        return "No image prediction was generated."

    if modality == "xray":
        selected = top_predictions[:3]
        findings = ", ".join(
            [f"{item['label']} ({item['probability']:.2f})" for item in selected]
        )
        return f"Chest X-ray suggests possible findings: {findings}."

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
) -> Dict:
    model, device = load_image_model(
        checkpoint_path=checkpoint_path,
        modality=modality,
        backbone_name=backbone_name,
    )

    tensor = preprocess_image(image_path=image_path, modality=modality)
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

    probabilities = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }

    thresholds_used: Optional[Dict[str, float]] = None
    predicted_labels: List[str] = []

    if modality == "xray":
        if thresholds_path is not None:
            try:
                thresholds_used = load_thresholds(thresholds_path)
            except Exception:
                thresholds_used = None

        # Apply thresholds (default 0.5 when threshold for a class is missing)
        for i, name in enumerate(class_names):
            thr = 0.5
            if thresholds_used and name in thresholds_used:
                thr = float(thresholds_used[name])

            if float(probs[i]) >= thr:
                predicted_labels.append(name)

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
    )

    return {
        "case_id": case_id,
        "modality": modality,
        "image_path": str(image_path),
        "top_predictions": top_predictions,
        "probabilities": probabilities,
        "embedding_path": embedding_path_value,
        "patient_summary_text": patient_summary_text,
        "predicted_labels": predicted_labels,
        "thresholds_used": thresholds_used,
    }