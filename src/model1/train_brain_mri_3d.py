from __future__ import annotations

import argparse
import sys
from pathlib import Path


VOLUME_SUFFIXES = {".nii", ".gz", ".mha", ".mhd", ".nrrd", ".npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold for experimental 3D brain MRI training.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "images" / "brain_mri"),
        help="Path to the brain MRI dataset root.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Placeholder epoch count for future 3D training.")
    parser.add_argument("--batch-size", type=int, default=2, help="Placeholder batch size for future 3D training.")
    parser.add_argument("--image-size", type=int, default=224, help="Placeholder image size for future 3D training.")
    return parser.parse_args()


def has_volume_data(data_dir: Path) -> bool:
    if not data_dir.exists():
        return False
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VOLUME_SUFFIXES:
            return True
    return False


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not has_volume_data(data_dir):
        print("3D CNN requires 3D volume data; current dataset is 2D images.")
        raise SystemExit(1)

    print(f"3D volume data detected under {data_dir}, but training is intentionally disabled in this scaffold.")
    print("This script is a guardrail only and does not start large training.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
