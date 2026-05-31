import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.config import (
    DEFAULT_BRAIN_CHECKPOINT,
    DEFAULT_XRAY_CHECKPOINT,
    DEFAULT_XRAY_THRESHOLDS,
    PROJECT_ROOT,
    create_required_folders,
)
from src.model1.infer import predict_image
from src.model2.pipeline import run_document_pipeline
from src.model3.pipeline import run_fusion_pipeline


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DOCUMENT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configurable local thesis batch using existing inference pipelines."
    )
    parser.add_argument("--brain-count", type=int, default=5)
    parser.add_argument("--xray-count", type=int, default=5)
    parser.add_argument("--prescription-count", type=int, default=5)
    parser.add_argument("--lab-report-count", type=int, default=5)
    parser.add_argument("--brain-fusion-count", type=int, default=3)
    parser.add_argument("--xray-fusion-count", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs/main_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brain-checkpoint", type=str, default=str(DEFAULT_BRAIN_CHECKPOINT))
    parser.add_argument("--xray-checkpoint", type=str, default=str(DEFAULT_XRAY_CHECKPOINT))
    parser.add_argument("--xray-thresholds", type=str, default=str(DEFAULT_XRAY_THRESHOLDS))
    return parser.parse_args()


def resolve_output_dir(output_dir_arg: str) -> Path:
    output_dir = Path(output_dir_arg)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def save_json(data: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def discover_files(base_dir: Path, extensions: Sequence[str]) -> List[Path]:
    if not base_dir.exists():
        return []

    allowed = {ext.lower() for ext in extensions}
    files = [
        path
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]
    return sorted(files)


def select_sample(paths: Sequence[Path], count: int, rng: random.Random) -> List[Path]:
    if count <= 0 or not paths:
        return []

    sample_size = min(count, len(paths))
    return rng.sample(list(paths), sample_size)


def interleave_documents(prescriptions: Sequence[Path], lab_reports: Sequence[Path]) -> List[Path]:
    interleaved: List[Path] = []
    max_len = max(len(prescriptions), len(lab_reports), 0)

    for index in range(max_len):
        if index < len(prescriptions):
            interleaved.append(prescriptions[index])
        if index < len(lab_reports):
            interleaved.append(lab_reports[index])

    return interleaved


def case_id_for(case_type: str, index: int) -> str:
    prefixes = {
        "brain_image_only": "main_brain",
        "xray_image_only": "main_xray",
        "prescription_doc_only": "main_prescription",
        "lab_doc_only": "main_lab",
        "brain_fusion": "main_fusion_brain",
        "xray_fusion": "main_fusion_xray",
    }

    return f"{prefixes[case_type]}_{index:03d}"


def run_case(
    *,
    case_id: str,
    case_type: str,
    output_dir: Path,
    image_path: Optional[Path] = None,
    image_modality: Optional[str] = None,
    document_path: Optional[Path] = None,
    brain_checkpoint: Optional[str] = None,
    xray_checkpoint: Optional[str] = None,
    xray_thresholds: Optional[str] = None,
) -> Dict[str, object]:
    case_output_dir = output_dir / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model1_output = None
    model2_output = None
    model3_output = None

    model1_output_path = ""
    model2_output_path = ""
    model3_output_path = ""
    embedding_path = ""

    try:
        if image_path is not None:
            if image_modality not in {"brain_mri", "xray"}:
                raise ValueError(f"Invalid image modality for {case_id}: {image_modality}")

            if image_modality == "xray":
                checkpoint_path = Path(xray_checkpoint) if xray_checkpoint else Path(DEFAULT_XRAY_CHECKPOINT)
            else:
                checkpoint_path = Path(brain_checkpoint) if brain_checkpoint else Path(DEFAULT_BRAIN_CHECKPOINT)

            embedding_output_path = case_output_dir / f"{case_id}_{image_modality}_embedding.npy"
            thresholds_path = xray_thresholds if image_modality == "xray" else None

            model1_output = predict_image(
                image_path=str(image_path),
                modality=image_modality,
                checkpoint_path=str(checkpoint_path),
                backbone_name="densenet121",
                case_id=case_id,
                embedding_output_path=str(embedding_output_path),
                thresholds_path=thresholds_path,
            )

            model1_output_path = str(case_output_dir / "model1_output.json")
            embedding_path = str(embedding_output_path)
            save_json(model1_output, Path(model1_output_path))

        if document_path is not None:
            model2_output = run_document_pipeline(
                document_path=str(document_path),
                case_id=case_id,
            )

            model2_output_path = str(case_output_dir / "model2_output.json")
            save_json(model2_output, Path(model2_output_path))

        model3_output = run_fusion_pipeline(
            case_id=case_id,
            model1_output=model1_output,
            model2_output=model2_output,
            kb_dir=str(PROJECT_ROOT / "data" / "kb"),
        )

        model3_output_path = str(case_output_dir / "model3_output.json")
        save_json(model3_output, Path(model3_output_path))

        runtime_seconds = round(time.perf_counter() - started, 3)
        print(f"[SUCCESS] {case_id}")

        return {
            "case_id": case_id,
            "case_type": case_type,
            "image_modality": image_modality or "",
            "image_path": str(image_path) if image_path else "",
            "document_path": str(document_path) if document_path else "",
            "model1_output_path": model1_output_path,
            "model2_output_path": model2_output_path,
            "model3_output_path": model3_output_path,
            "embedding_path": embedding_path,
            "status": "success",
            "error_message": "",
            "runtime_seconds": runtime_seconds,
        }

    except Exception as error:
        runtime_seconds = round(time.perf_counter() - started, 3)
        print(f"[FAILED] {case_id}: {error}")

        return {
            "case_id": case_id,
            "case_type": case_type,
            "image_modality": image_modality or "",
            "image_path": str(image_path) if image_path else "",
            "document_path": str(document_path) if document_path else "",
            "model1_output_path": model1_output_path,
            "model2_output_path": model2_output_path,
            "model3_output_path": model3_output_path,
            "embedding_path": embedding_path,
            "status": "failed",
            "error_message": str(error),
            "runtime_seconds": runtime_seconds,
        }


def write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "case_id",
        "case_type",
        "image_modality",
        "image_path",
        "document_path",
        "model1_output_path",
        "model2_output_path",
        "model3_output_path",
        "embedding_path",
        "status",
        "error_message",
        "runtime_seconds",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_report(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    requested_total: int,
    output_dir: Path,
    seed: int,
) -> None:
    total_completed = sum(1 for row in rows if row.get("status") == "success")
    total_failed = sum(1 for row in rows if row.get("status") == "failed")
    case_type_counts = Counter(row.get("case_type", "unknown") for row in rows)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "Main Run Report",
        f"Timestamp: {timestamp}",
        f"Output folder: {output_dir}",
        f"Seed: {seed}",
        f"Total cases requested: {requested_total}",
        f"Total cases completed: {total_completed}",
        f"Total cases failed: {total_failed}",
        "",
        "Counts by case type:",
    ]

    for case_type, count in sorted(case_type_counts.items()):
        lines.append(f"- {case_type}: {count}")

    lines.extend([
        "",
        "Note: inference-only run; no training was performed.",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_case_plan(
    args: argparse.Namespace,
    rng: random.Random,
) -> List[Tuple[str, str, Optional[Path], Optional[str], Optional[Path]]]:
    brain_images = discover_files(PROJECT_ROOT / "data" / "images" / "brain_mri", IMAGE_EXTENSIONS)
    xray_images = discover_files(PROJECT_ROOT / "data" / "images" / "xray", IMAGE_EXTENSIONS)
    prescriptions = discover_files(PROJECT_ROOT / "data" / "documents" / "prescriptions", DOCUMENT_EXTENSIONS)
    lab_reports = discover_files(PROJECT_ROOT / "data" / "documents" / "lab_reports", DOCUMENT_EXTENSIONS)

    selected_brain = select_sample(brain_images, args.brain_count, rng)
    selected_xray = select_sample(xray_images, args.xray_count, rng)
    selected_prescriptions = select_sample(prescriptions, args.prescription_count, rng)
    selected_lab_reports = select_sample(lab_reports, args.lab_report_count, rng)

    brain_fusion_images = select_sample(brain_images, args.brain_fusion_count, rng)
    xray_fusion_images = select_sample(xray_images, args.xray_fusion_count, rng)

    fusion_documents = interleave_documents(selected_prescriptions, selected_lab_reports)
    if not fusion_documents:
        fusion_documents = list(selected_prescriptions or selected_lab_reports)

    case_plan: List[Tuple[str, str, Optional[Path], Optional[str], Optional[Path]]] = []

    for index, image_path in enumerate(selected_brain, start=1):
        case_plan.append(
            (
                case_id_for("brain_image_only", index),
                "brain_image_only",
                image_path,
                "brain_mri",
                None,
            )
        )

    for index, image_path in enumerate(selected_xray, start=1):
        case_plan.append(
            (
                case_id_for("xray_image_only", index),
                "xray_image_only",
                image_path,
                "xray",
                None,
            )
        )

    for index, document_path in enumerate(selected_prescriptions, start=1):
        case_plan.append(
            (
                case_id_for("prescription_doc_only", index),
                "prescription_doc_only",
                None,
                None,
                document_path,
            )
        )

    for index, document_path in enumerate(selected_lab_reports, start=1):
        case_plan.append(
            (
                case_id_for("lab_doc_only", index),
                "lab_doc_only",
                None,
                None,
                document_path,
            )
        )

    for index, image_path in enumerate(brain_fusion_images, start=1):
        document_path = fusion_documents[(index - 1) % len(fusion_documents)] if fusion_documents else None
        case_plan.append(
            (
                case_id_for("brain_fusion", index),
                "brain_fusion",
                image_path,
                "brain_mri",
                document_path,
            )
        )

    for index, image_path in enumerate(xray_fusion_images, start=1):
        document_path = fusion_documents[(index - 1) % len(fusion_documents)] if fusion_documents else None
        case_plan.append(
            (
                case_id_for("xray_fusion", index),
                "xray_fusion",
                image_path,
                "xray",
                document_path,
            )
        )

    return case_plan


def main() -> None:
    create_required_folders()
    args = parse_args()
    rng = random.Random(args.seed)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    brain_checkpoint = str(Path(args.brain_checkpoint))
    xray_checkpoint = str(Path(args.xray_checkpoint))
    xray_thresholds = str(Path(args.xray_thresholds))

    case_plan = build_case_plan(args, rng)
    total_requested = len(case_plan)
    rows: List[Dict[str, object]] = []

    for index, (case_id, case_type, image_path, image_modality, document_path) in enumerate(case_plan, start=1):
        print(f"[{index}/{total_requested}] Running {case_id}...")
        row = run_case(
            case_id=case_id,
            case_type=case_type,
            output_dir=output_dir,
            image_path=image_path,
            image_modality=image_modality,
            document_path=document_path,
            brain_checkpoint=brain_checkpoint,
            xray_checkpoint=xray_checkpoint,
            xray_thresholds=xray_thresholds,
        )
        rows.append(row)

    write_csv(rows, output_dir / "main_run_summary.csv")
    write_report(rows, output_dir / "main_run_report.txt", total_requested, output_dir, args.seed)

    completed = sum(1 for row in rows if row.get("status") == "success")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    print(f"[DONE] Completed={completed} Failed={failed} Output={output_dir}")


if __name__ == "__main__":
    main()