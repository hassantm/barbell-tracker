# Barbell Tracker

Track bar path and velocity from squat, bench, or deadlift footage.

## Setup

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+ and a working OpenCV install with display support  
> (i.e. not headless — the interactive calibration needs a window).

---

## Quick Start

```bash
python barbell_tracker.py --video my_session.mp4
```

This will:
1. Open the first frame for **interactive scale calibration** — click the two
   outermost edges of one plate, then press `Q`
2. Run YOLO + ByteTrack across all frames (preview window opens)
3. Save to `./output/`:
   - `tracked.mp4` — annotated video with path overlay
   - `analysis.png` — bar path, velocity, and MCV per rep
   - `summary.json` — rep-by-rep metrics

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Path to input video |
| `--output` | `./output` | Output directory |
| `--model` | `yolov8n.pt` | YOLO model (downloads automatically on first run) |
| `--plate-diameter` | `450` | Plate diameter in mm (450mm = 20kg/45lb) |
| `--conf` | `0.3` | Detection confidence threshold |
| `--class-ids` | auto | YOLO class IDs to track (space-separated) |
| `--no-calibrate` | off | Skip calibration; uses 1.0 mm/px placeholder |
| `--smooth-window` | `9` | Savitzky-Golay smoothing window (odd int) |
| `--list-classes` | — | Print all class names for the chosen model and exit |

---

## Choosing a Model

### Option A — General model (works out of the box)
The default `yolov8n.pt` detects 80 COCO classes. A barbell won't be  
labelled as "barbell", but the **plates** may be detected as `sports ball`  
(class 32) or the bar/rack as other objects.

Run this to see what gets detected in your video:
```bash
python barbell_tracker.py --video clip.mp4 --list-classes
```

Then filter to the most useful class:
```bash
python barbell_tracker.py --video clip.mp4 --class-ids 32
```

### Option B — Barbell-specific model (recommended)
Roboflow Universe has pretrained barbell detection models. Download one:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace().project("barbell-detection")
project.version(1).download("yolov8")
```
Then point the tracker at it:
```bash
python barbell_tracker.py --video clip.mp4 --model barbell-detection/weights/best.pt
```

### Option C — Fine-tune on your own footage (best accuracy)
1. Record 5–10 sets in your gym
2. Label ~100 frames using [Roboflow](https://roboflow.com) or [Label Studio](https://labelstud.io)
3. Fine-tune YOLOv8n (takes ~15 min on a laptop GPU):
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="dataset.yaml", epochs=50, imgsz=640)
```

---

## Filming Tips for Best Results

- **Camera position**: fixed, side-on to the bar. Perpendicular to the bar's  
  direction of travel.
- **Frame rate**: 60fps+ for velocity accuracy. Most phones can do this.
- **Plate visibility**: keep at least one full plate in frame throughout the set.
- **Lighting**: avoid harsh shadows across the plates — these break colour-based  
  detection and can confuse YOLO.
- **Background**: a plain wall behind the rack helps enormously.

---

## Key Metrics

| Metric | Meaning |
|--------|---------|
| **Bar Path** | Ideally a tight vertical line. Lateral drift = technique issue |
| **Mean Concentric Velocity (MCV)** | Average speed on the way up. Drops as fatigue/load increases |
| **Peak Concentric Velocity** | Max speed during the concentric phase |

MCV benchmarks (approximate, competition lifters):
- > 1.0 m/s → < ~60% 1RM  
- 0.5–0.8 m/s → ~75–85% 1RM  
- < 0.35 m/s → near-maximal effort

---

## Project Roadmap Ideas

- [ ] Automatic plate-size detection (removes need for manual calibration)  
- [ ] Rep quality score (path straightness + velocity consistency)  
- [ ] Session-level fatigue curve (MCV across sets)  
- [ ] Export to CSV for external analysis  
- [ ] Simple web UI (Gradio or Streamlit)  
