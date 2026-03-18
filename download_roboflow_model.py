#!/usr/bin/env python3
"""
download_roboflow_model.py
--------------------------
Downloads the barbells-detector dataset from Roboflow and trains a YOLOv8
model locally. Run this once before using barbell_tracker.py.

Usage:
    python download_roboflow_model.py --api-key YOUR_ROBOFLOW_API_KEY

The trained model will be saved to:
    ./models/barbells-detector/weights/best.pt

Roboflow project details:
    Workspace : yolo-project-c2bfs
    Project   : barbells-detector
    Version   : 1
    Classes   : Barbell (bar body), End (plate end)
"""

import argparse
import sys
from pathlib import Path


ROBOFLOW_WORKSPACE = "yolo-project-c2bfs"
ROBOFLOW_PROJECT   = "barbells-detector"
ROBOFLOW_VERSION   = 1

DATASET_DIR   = Path("./models/barbells-detector-dataset")
WEIGHTS_PATH  = Path("./models/barbells-detector/weights/best.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download barbells-detector dataset from Roboflow and train locally."
    )
    parser.add_argument(
        "--api-key", required=True,
        help="Your Roboflow API key (roboflow.com → Settings → Roboflow API)"
    )
    parser.add_argument(
        "--version", type=int, default=ROBOFLOW_VERSION,
        help=f"Dataset version to download (default: {ROBOFLOW_VERSION})"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download if dataset already exists"
    )
    return parser.parse_args()


def download_dataset(api_key: str, version: int) -> Path:
    """Download dataset from Roboflow. Returns path to data.yaml."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    print(f"Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)

    print(f"Downloading dataset v{version} (yolov8 format)...")
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)

    dataset = project.version(version).download(
        model_format="yolov8",
        location=str(DATASET_DIR),
        overwrite=True,
    )

    # Find data.yaml
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        # Roboflow sometimes puts it in a versioned subfolder
        candidates = list(DATASET_DIR.rglob("data.yaml"))
        if candidates:
            yaml_path = candidates[0]
        else:
            print(f"ERROR: data.yaml not found in {DATASET_DIR}")
            sys.exit(1)

    print(f"Dataset ready: {yaml_path}\n")
    return yaml_path


def train(yaml_path: Path, epochs: int):
    """Train YOLOv8n on the downloaded dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    output_dir = Path("./models/barbells-detector")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training YOLOv8n for {epochs} epochs...")
    print(f"Output: {output_dir}/weights/best.pt\n")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )

    if WEIGHTS_PATH.exists():
        size_mb = WEIGHTS_PATH.stat().st_size / (1024 * 1024)
        print(f"\n✓ Model ready: {WEIGHTS_PATH.resolve()}  ({size_mb:.1f} MB)")
        print("\nRun the tracker with:")
        print("    python barbell_tracker.py --video your_clip.mp4")
    else:
        print(f"\nWARNING: Expected weights at {WEIGHTS_PATH} — check training output above.")


def main():
    args = parse_args()

    if WEIGHTS_PATH.exists() and not args.skip_download:
        print(f"Weights already exist at {WEIGHTS_PATH}")
        print("Delete them or pass --skip-download to retrain.")
        sys.exit(0)

    yaml_path = DATASET_DIR / "data.yaml"
    if args.skip_download and yaml_path.exists():
        print(f"Skipping download, using existing dataset at {yaml_path}")
    else:
        yaml_path = download_dataset(args.api_key, args.version)

    train(yaml_path, args.epochs)


if __name__ == "__main__":
    main()
