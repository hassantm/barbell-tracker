# Training Data Guide

## Goal

Retrain the `barbells-detector` model on your own footage to get reliable `End` (plate edge) detections. This unlocks:
- Real mm/px auto-calibration → true m/s velocity values
- Remove `--no-calibrate` from all workflows

## Step 1 — Capture Footage

Record gym sessions at **60fps, side-on, fixed camera**. Cover:
- Multiple loads (60–100% 1RM)
- Different vertical bar positions (bottom, mid, lockout)
- Consistent background and lighting

## Step 2 — Extract Frames

Extract ~150 diverse frames at ~0.5 fps:

```bash
ffmpeg -i clip.MOV -vf "fps=0.5" frames/frame_%04d.jpg
```

Aim for frames where:
- The near plate edge is clearly visible
- Bar is at varied heights across the set
- At least 30 frames show the bar at the bottom and lockout each

## Step 3 — Label in Roboflow

1. Log into Roboflow → open the `barbells-detector` project
2. Upload the extracted frames as a new batch
3. Label two classes:
   - `Barbell` — bounding box around the bar body (existing reliable class)
   - `End` — **tight box around the near plate edge only** (the unreliable class — focus effort here)
4. Target ≥ 120 `End` annotations across 150 frames
5. Generate a new dataset version (v2)

## Step 4 — Train

```bash
python download_roboflow_model.py --api-key YOUR_KEY --version 2 --epochs 150
```

To optionally upgrade from `yolov8n` to `yolov8s` (better accuracy, ~200ms/frame on Pi 5), edit `download_roboflow_model.py` line ~107 before training:

```python
# Change:
model = YOLO("yolov8n.pt")
# To:
model = YOLO("yolov8s.pt")
```

## Step 5 — Validate Calibration

After training, run without `--no-calibrate`:

```bash
python barbell_tracker.py --video input/test_clip.MOV --headless
```

Check `output/summary.json`:
```json
{
  "calibration": {
    "method": "auto",          ← should be "auto", not "manual_fallback"
    "mm_per_pixel": 1.4,       ← typical value for side-on 45lb plate at ~2m distance
    "frames_used": 10
  }
}
```

If `method` is still `manual_fallback`, the `End` class detections are insufficient — review annotations and retrain.

## Step 6 — Update Class Name Constants (Optional)

The current class names in the model are counterintuitive (`CLASS_PLATE = "Barbell"`, `CLASS_BAR = "End"`). When retraining, consider renaming in Roboflow to `BarBody` / `PlateEdge` for clarity, then update `barbell_tracker.py` lines 62–63:

```python
CLASS_PLATE = "PlateEdge"
CLASS_BAR   = "BarBody"
```
