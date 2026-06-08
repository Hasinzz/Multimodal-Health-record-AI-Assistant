from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = (
    ROOT
    / "outputs"
    / "v4_advanced_improvement"
    / "cross_attention"
    / "cross_attention_v4_readiness_report.md"
)
SEARCH_ROOTS = [
    ROOT / "data" / "fusion_pairs_v4",
    ROOT / "data" / "fusion_pairs",
    ROOT / "data" / "external" / "openi",
    ROOT / "data" / "external" / "multicare",
    ROOT / "data" / "external" / "pmc_patients",
    ROOT / "data" / "external" / "synthea",
]
REQUIRED_COLUMNS = {"image_path", "text", "label"}
CASE_COLUMNS = {"patient_or_case_id", "case_id", "patient_id"}


def read_header(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def inspect_csv(path: Path) -> Dict:
    header = read_header(path)
    header_set = set(header)
    has_required = REQUIRED_COLUMNS.issubset(header_set)
    has_case_id = bool(CASE_COLUMNS.intersection(header_set))
    has_split = "split" in header_set or any(part in str(path).lower() for part in ["train", "val", "test"])
    return {
        "path": str(path.relative_to(ROOT)),
        "columns": header,
        "has_required_columns": has_required,
        "has_case_id": has_case_id,
        "has_split": has_split,
        "ready": has_required and has_case_id and has_split,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inspected = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            try:
                inspected.append(inspect_csv(path))
            except Exception as error:
                inspected.append({"path": str(path.relative_to(ROOT)), "error": str(error), "ready": False})

    ready = [item for item in inspected if item.get("ready")]
    lines = [
        "# Cross-Attention V4 Readiness Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    if ready:
        lines.append("True paired image-text-label CSV files were found:")
        lines.extend(f"- `{item['path']}`" for item in ready)
    else:
        lines.append(
            "No true paired image-text-label dataset found. Cross-modal attention V4 training cannot start honestly."
        )
    lines.extend(
        [
            "",
            "Required columns: `image_path`, `text`, `label`, `split`, and `patient_or_case_id` or equivalent.",
            "",
            "Inspected CSV files:",
        ]
    )
    if inspected:
        for item in inspected:
            columns = ", ".join(item.get("columns", []))
            status = "ready" if item.get("ready") else "not ready"
            lines.append(f"- `{item['path']}`: {status}; columns: {columns}")
    else:
        lines.append("- No CSV files found in the configured paired-data folders.")
    lines.extend(
        [
            "",
            "Pseudo-pairs may be used only as experimental demonstrations, not as real clinical multimodal training evidence.",
        ]
    )
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
