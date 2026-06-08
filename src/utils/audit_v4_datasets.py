from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "v4_advanced_improvement" / "reports"

SCAN_TARGETS = [
    "data/improvement",
    "data/improvement/model2_lab_reports",
    "data/improvement/model2_lab_reports/lbmaske",
    "data/improvement/model2_bd_prescriptions",
    "data/improvement/model2_ocr_prescriptions",
    "data/improvement/model3_medicine_datasets",
    "data/improvement/model3_medical_information",
    "data/improvement/medocr_vision_optional",
    "data/documents/lab_reports",
    "data/documents/prescriptions",
    "data/ocr_word_datasets",
    "data/kb",
]


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_file():
            yield child


def likely_use(path_text: str, extension_counts: Dict[str, int]) -> str:
    lower = path_text.replace("\\", "/").lower()
    image_count = sum(
        extension_counts.get(ext, 0)
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    )

    if "medocr_vision_optional" in lower:
        return "not useful right now; folder is currently only a placeholder"
    if "model2_lab_reports" in lower or "documents/lab_reports" in lower:
        return "YOLO ROI, OCR improvement, BioBERT NER source text"
    if "model2_bd_prescriptions" in lower:
        return "BioBERT NER drug dictionary and medicine-word recognition support"
    if "model2_ocr_prescriptions" in lower:
        return "BioBERT NER drug-name support and medicine-word recognition support"
    if "model3_medicine_datasets" in lower or "model3_medical_information" in lower:
        return "RAG KB upgrade and medicine dictionary support"
    if "data/kb" in lower:
        return "existing baseline RAG KB"
    if "documents/prescriptions" in lower:
        return "OCR evaluation and possible NER source text"
    if "ocr_word_datasets" in lower:
        return "medicine-word OCR support"
    if image_count > 0:
        return "image data; inspect labels before training"
    return "not enough evidence from file names alone"


def scan_folder(relative_path: str, sample_limit: int = 10) -> Dict:
    path = ROOT / relative_path
    samples: List[str] = []
    extensions: Counter[str] = Counter()
    file_count = 0
    total_bytes = 0

    for file_path in iter_files(path):
        file_count += 1
        total_bytes += file_path.stat().st_size
        extensions[file_path.suffix.lower() or "[no_ext]"] += 1
        if len(samples) < sample_limit:
            samples.append(str(file_path.relative_to(ROOT)))

    extension_counts = dict(sorted(extensions.items()))
    return {
        "path": relative_path,
        "exists": path.exists(),
        "is_file": path.is_file(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "extension_counts": extension_counts,
        "sample_files": samples,
        "likely_use": likely_use(relative_path, extension_counts),
    }


def write_markdown(results: List[Dict], output_path: Path) -> None:
    lines = [
        "# V4 Dataset Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This audit inspects likely V4 data folders without modifying existing datasets.",
        "",
        "| Folder | Exists | Files | Extensions | Likely use | Sample files |",
        "|---|---:|---:|---|---|---|",
    ]
    for item in results:
        ext_summary = ", ".join(
            f"{ext}: {count}" for ext, count in item["extension_counts"].items()
        )
        sample_summary = "<br>".join(item["sample_files"][:5]) or "-"
        lines.append(
            "| {path} | {exists} | {file_count} | {exts} | {use} | {samples} |".format(
                path=item["path"],
                exists="yes" if item["exists"] else "no",
                file_count=item["file_count"],
                exts=ext_summary or "-",
                use=item["likely_use"],
                samples=sample_summary,
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = [scan_folder(path) for path in SCAN_TARGETS]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "results": results,
    }

    json_path = REPORT_DIR / "v4_dataset_audit.json"
    md_path = REPORT_DIR / "v4_dataset_audit.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(results, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
