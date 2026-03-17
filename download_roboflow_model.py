#!/usr/bin/env python3
"""
download_roboflow_model.py
--------------------------
One-shot script to download the barbell-detector YOLO model from Roboflow.
Run this once before using barbell_tracker.py.

Usage:
    python download_roboflow_model.py --api-key YOUR_ROBOFLOW_API_KEY

The model will be saved to:
    ./models/barbells-detector/weights/best.pt

Roboflow project details:
    Workspace : yolo-project-c2bfs
    Project   : barbells-detector
    Classes   : plate, bar
"""

import argparse
import sys
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Roboflow project config
# ---------------------------------------------------------------------------

ROBOFLOW_WORKSPACE = "yolo-project-c2bfs"
ROBOFLOW_PROJECT   = "barbells-detector"
ROBOFLOW_VERSION   = 1          # bump this if you want a newer model version
ROBOFLOW_FORMAT    = "yolov8"

MODEL_OUTPUT_DIR   = Path("./models/barbells-detector")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download barbells-detector YOLO model from Roboflow."
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Your Roboflow API key (find it at roboflow.com → Settings → API)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=ROBOFLOW_VERSION,
        help=f"Model version to download (default: {ROBOFLOW_VERSION})",
    )
    parser.add_argument(
        "--output",
        default=str(MODEL_OUTPUT_DIR),
        help=f"Directory to save model (default: {MODEL_OUTPUT_DIR})",
    )
    return parser.parse_args()


def download_model(api_key: str, version: int, output_dir: Path):
    """Download model weights from Roboflow and place them in output_dir."""

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed.")
        print("       Run: pip install roboflow")
        sys.exit(1)

    print(f"Connecting to Roboflow workspace: {ROBOFLOW_WORKSPACE}")
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)

    print(f"Downloading version {version} in '{ROBOFLOW_FORMAT}' format...")
    print(f"Target directory: {output_dir.resolve()}")

    # Roboflow downloads into a subfolder named <project>-<version>/
    # We download into the parent of output_dir so the structure lands cleanly.
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset = project.version(version).download(
        model_format=ROBOFLOW_FORMAT,
        location=str(output_dir.parent),
        overwrite=True,
    )

    # Roboflow creates a folder like ./models/barbells-detector-1/
    # Rename it to the clean output_dir name if needed.
    roboflow_dir = output_dir.parent / f"{ROBOFLOW_PROJECT}-{version}"
    if roboflow_dir.exists() and roboflow_dir != output_dir:
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        roboflow_dir.rename(output_dir)
        print(f"Renamed {roboflow_dir.name}/ → {output_dir.name}/")

    # Verify weights file exists
    weights = output_dir / "weights" / "best.pt"
    if weights.exists():
        size_mb = weights.stat().st_size / (1024 * 1024)
        print(f"\n✓ Model ready: {weights.resolve()}  ({size_mb:.1f} MB)")
        print("\nRun the tracker with:")
        print(f"    python barbell_tracker.py --video your_clip.mp4")
    else:
        print(f"\nWARNING: Expected weights at {weights} — not found.")
        print("Check the downloaded directory structure:")
        for p in output_dir.rglob("*.pt"):
            print(f"  {p}")
        print("Use --model to point barbell_tracker.py at the correct .pt file.")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    download_model(args.api_key, args.version, output_dir)


if __name__ == "__main__":
    main()
