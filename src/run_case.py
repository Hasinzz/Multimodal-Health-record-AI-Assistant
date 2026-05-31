import argparse
import json
from pathlib import Path
from typing import Any, Optional

from src.config import (
    DEFAULT_BRAIN_CHECKPOINT,
    DEFAULT_XRAY_CHECKPOINT,
    DEFAULT_XRAY_THRESHOLDS,
    KB_DIR,
    OUTPUT_DIR,
    create_required_folders,
)
from src.model1.infer import predict_image
from src.model2.pipeline import run_document_pipeline
from src.model3.pipeline import run_fusion_pipeline


def resolve_checkpoint_path(checkpoint_value: str, modality: str) -> Path:
    checkpoint_path = Path(checkpoint_value)

    if checkpoint_path.exists():
        return checkpoint_path

    if checkpoint_path.is_dir():
        candidates = sorted(checkpoint_path.glob("*.pt"))
        if len(candidates) == 1:
            return candidates[0]

    if checkpoint_path.parent.exists():
        candidates = sorted(checkpoint_path.parent.glob("*.pt"))
        if len(candidates) == 1:
            return candidates[0]

    expected_name = "xray_best_model.pt" if modality == "xray" else "brain_best_model.pt"
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. Expected {expected_name} under checkpoints/model1, or pass an explicit existing path with --{modality}_checkpoint."
    )


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local multimodal thesis pipeline.")
    parser.add_argument("--case_id", type=str, default="case_001", help="Unique case ID.")
    parser.add_argument("--image", type=str, default=None, help="Path to X-ray or Brain MRI image.")
    parser.add_argument("--image_modality", type=str, choices=["xray", "brain_mri"], default=None, help="Image modality.")
    parser.add_argument("--document", type=str, default=None, help="Path to prescription/lab report image, PDF, or TXT file.")
    parser.add_argument("--xray_checkpoint", type=str, default=str(DEFAULT_XRAY_CHECKPOINT), help="Path to X-ray checkpoint.")
    parser.add_argument(
        "--xray-thresholds",
        "--xray_thresholds",
        dest="xray_thresholds",
        type=str,
        default=str(DEFAULT_XRAY_THRESHOLDS),
        help="Path to JSON file with per-class X-ray thresholds (optional).",
    )
    parser.add_argument("--brain_checkpoint", type=str, default=str(DEFAULT_BRAIN_CHECKPOINT), help="Path to Brain MRI checkpoint.")
    parser.add_argument("--backbone", type=str, default="densenet121", help="Backbone name used during training, for example densenet121 or resnet50.")
    parser.add_argument("--kb_dir", type=str, default=str(KB_DIR), help="Path to knowledge-base folder.")
    return parser.parse_args()


def main() -> None:
    create_required_folders()
    args = parse_args()

    model1_output = None
    model2_output = None

    case_output_dir = OUTPUT_DIR / args.case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        if not args.image_modality:
            raise ValueError("If --image is provided, you must also provide --image_modality xray or brain_mri.")

        if args.image_modality == "xray":
            checkpoint_path = resolve_checkpoint_path(args.xray_checkpoint, "xray")
            thresholds_path: Optional[str] = args.xray_thresholds
        else:
            checkpoint_path = resolve_checkpoint_path(args.brain_checkpoint, "brain_mri")
            thresholds_path = None

        embedding_output_path = case_output_dir / f"{args.case_id}_{args.image_modality}_embedding.npy"

        model1_output = predict_image(
            image_path=args.image,
            modality=args.image_modality,
            checkpoint_path=str(checkpoint_path),
            backbone_name=args.backbone,
            case_id=args.case_id,
            embedding_output_path=str(embedding_output_path),
            thresholds_path=thresholds_path,
        )

        save_json(model1_output, case_output_dir / "model1_output.json")
        print("[Saved] Model-1 output")

    if args.document:
        model2_output = run_document_pipeline(document_path=args.document, case_id=args.case_id)
        save_json(model2_output, case_output_dir / "model2_output.json")
        print("[Saved] Model-2 output")

    model3_output = run_fusion_pipeline(
        case_id=args.case_id,
        model1_output=model1_output,
        model2_output=model2_output,
        kb_dir=args.kb_dir,
    )

    save_json(model3_output, case_output_dir / "model3_output.json")

    print("[Saved] Model-3 output")
    print()
    print("Final Summary:")
    print(model3_output["final_summary"])
    print()
    print("Doctor Feedback:")
    print(model3_output["doctor_feedback"])


if __name__ == "__main__":
    main()
