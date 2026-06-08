from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "outputs" / "v4_advanced_improvement" / "reports" / "v4_requirements_check.md"

PACKAGES = [
    ("torch", "torch", "core training"),
    ("torchvision", "torchvision", "image models"),
    ("timm", "timm", "image backbones"),
    ("opencv-python / cv2", "cv2", "OCR preprocessing"),
    ("pytesseract", "pytesseract", "OCR"),
    ("ultralytics", "ultralytics", "YOLO ROI"),
    ("transformers", "transformers", "BioBERT NER"),
    ("datasets", "datasets", "BioBERT NER"),
    ("evaluate", "evaluate", "BioBERT NER metrics"),
    ("seqeval", "seqeval", "NER Entity-F1"),
    ("accelerate", "accelerate", "HuggingFace training"),
    ("sentence-transformers", "sentence_transformers", "text embeddings"),
    ("streamlit", "streamlit", "UI demo"),
    ("pandas", "pandas", "tables"),
    ("openpyxl", "openpyxl", "Excel KB ingestion"),
]

INSTALL_HINTS = {
    "ultralytics": "C:\\Users\\T2520824\\Miniconda3\\envs\\thesis_gpu\\python.exe -m pip install ultralytics",
    "transformers": "C:\\Users\\T2520824\\Miniconda3\\envs\\thesis_gpu\\python.exe -m pip install transformers datasets evaluate seqeval accelerate",
    "datasets": "C:\\Users\\T2520824\\Miniconda3\\envs\\thesis_gpu\\python.exe -m pip install transformers datasets evaluate seqeval accelerate",
    "evaluate": "C:\\Users\\T2520824\\Miniconda3\\envs\\thesis_gpu\\python.exe -m pip install transformers datasets evaluate seqeval accelerate",
    "seqeval": "C:\\Users\\T2520824\\Miniconda3\\envs\\thesis_gpu\\python.exe -m pip install transformers datasets evaluate seqeval accelerate",
    "accelerate": "C:\\Users\\T2520824\\Miniconda3\\envs\\thesis_gpu\\python.exe -m pip install transformers datasets evaluate seqeval accelerate",
    "sentence-transformers": "C:\\Users\\T2520824\\Miniconda3\\envs\\thesis_gpu\\python.exe -m pip install sentence-transformers",
    "openpyxl": "C:\\Users\\T2520824\\Miniconda3\\envs\\thesis_gpu\\python.exe -m pip install openpyxl",
}


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for display_name, module_name, purpose in PACKAGES:
        available = importlib.util.find_spec(module_name) is not None
        rows.append(
            {
                "display_name": display_name,
                "module_name": module_name,
                "purpose": purpose,
                "available": available,
                "install_hint": INSTALL_HINTS.get(display_name, ""),
            }
        )

    lines = [
        "# V4 Requirements Check",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Package | Module | Purpose | Status |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | `{row['module_name']}` | {row['purpose']} | "
            f"{'installed' if row['available'] else 'missing'} |"
        )

    missing_hints = sorted(
        {row["install_hint"] for row in rows if not row["available"] and row["install_hint"]}
    )
    lines.extend(["", "Missing package install commands:"])
    if missing_hints:
        lines.extend(f"- `{hint}`" for hint in missing_hints)
    else:
        lines.append("- None for the checked V4 packages.")
    lines.append("")
    lines.append("No packages were installed by this check.")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
