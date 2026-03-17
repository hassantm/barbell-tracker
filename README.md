# Barbell Tracker

Track bar path, velocity, and tilt from squat, bench, or deadlift footage.

Uses a Roboflow barbell-specific model (`barbells-detector`) that detects two classes:
- **plate** — bounding box used for position tracking and auto-calibration
- **bar** — centroid used to compute bar tilt angle

---

## Setup

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+ and a working OpenCV install with display support.

---

## Quick Start

### Step 1 — Get a Roboflow API key

1. Create a free account at [roboflow.com](https://roboflow.com)
2. Go to **Settings → API** and copy your API key

### Step 2 — Download the model (once)

```bash
python download_roboflow_model.py --api-key YOUR_API_KEY
```

This downloads the `barbells-detector` model to `./models/barbells-detector/`.  
You only need to do this once.

### Step 3 — Track a video

```bash
python barbell_tracker.py --video my_session.mp4
```

This will:
1. Auto-calibrate scale from the plate bounding-box width (first 10 frames with a plate detection)
2. Run YOLO + ByteTrack across all frames (preview window opens)
3. Save to `./output/`:
   - `tracked.mp4` — annotated video with path overlay and live tilt readout
   - `analysis.png` — 4-panel plot: bar path, velocity, MCV per rep, bar tilt
   - `summary.json` — rep-by-rep metrics including tilt

---

## Auto-Calibration

No manual clicking required.  The tracker measures the plate bounding-box width
in pixels across the first 10 frames where a plate is detected, then computes:

```
mm_per_pixel = plate_diameter_mm / plate_bbox_width_px
```

The 10-frame average is used for the final scale.  This is printed to the console:

```
Auto-calibration: avg plate width = 312.4 px  →  1.441 mm/pixel  (from 10 frames)
```

Use `--no-calibrate` to bypass and use 1.0 mm/px instead (pixel-space output only).  
Use `--plate-diameter` if you're using non-standard plates (default: 450mm = 20kg/45lb).

---

## Bar Tilt

When both a `plate` and `bar` detection are present in the same frame, the
tracker computes the angle between their centroids (degrees from horizontal).

- **0°** = bar perfectly horizontal
- **Positive** = bar tilted upward toward the bar end
- **Negative** = bar tilted downward toward the bar end

Tilt is plotted in panel 4 of `analysis.png` and included in `summary.json`
as `mean_bar_tilt_degrees` (mean during the ascent/concentric phase only).

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Path to input video |
| `--output` | `./output` | Output directory |
| `--model` | `./models/barbells-detector/weights/best.pt` | YOLO model .pt file |
| `--plate-diameter` | `450` | Plate diameter in mm (450mm = 20kg/45lb) |
| `--conf` | `0.3` | Detection confidence threshold |
| `--no-calibrate` | off | Skip auto-calibration; use 1.0 mm/px placeholder |
| `--api-key` | — | Roboflow API key — downloads model if not already present |
| `--smooth-window` | `9` | Savitzky-Golay smoothing window (odd int) |
| `--list-classes` | — | Print all class names for the chosen model and exit |

---

## Outputs

### `tracked.mp4`
Annotated video with:
- Bounding boxes and labels for `plate` and `bar`
- Coloured bar path trail (plate centroid)
- Live tilt angle and calibration status overlaid

### `analysis.png`
4-panel dark-theme plot:

| Panel | Content |
|-------|---------|
| Left (full height) | Bar path — plate centroid in mm space. Ideally a tight vertical line |
| Top right | Velocity over time |
| Middle right | Mean Concentric Velocity per rep (bar chart + peak overlay) |
| Bottom right | Bar tilt over time (degrees from horizontal) |

### `summary.json`
```json
{
  "calibration": {
    "mm_per_pixel": 1.441,
    "method": "auto",
    "frames_used": 10
  },
  "reps": [
    {
      "rep": 1,
      "mean_concentric_velocity_ms": 0.612,
      "peak_concentric_velocity_ms": 0.894,
      "mean_bar_tilt_degrees": 2.3,
      "frame_count": 87
    }
  ]
}
```

---

## Key Metrics

| Metric | Meaning |
|--------|---------|
| **Bar Path** | Ideally a tight vertical line. Lateral drift = technique issue |
| **Mean Concentric Velocity (MCV)** | Average speed on the way up. Drops as fatigue/load increases |
| **Peak Concentric Velocity** | Max speed during the concentric phase |
| **Mean Bar Tilt (ascent)** | Average tilt during the concentric phase. Persistent tilt = uneven loading or setup |

MCV benchmarks (approximate, competition lifters):
- > 1.0 m/s → < ~60% 1RM
- 0.5–0.8 m/s → ~75–85% 1RM
- < 0.35 m/s → near-maximal effort

---

## Filming Tips for Best Results

- **Camera position**: fixed, side-on to the bar. Perpendicular to the bar's
  direction of travel.
- **Frame rate**: 60fps+ for velocity accuracy. Most phones can do this.
- **Plate visibility**: keep at least one full plate in frame throughout the set.
  Auto-calibration needs the plate visible and un-occluded in early frames.
- **Bar visibility**: keep the bar end in frame for tilt detection. If the bar
  is cropped, tilt data will be missing for those frames.
- **Lighting**: avoid harsh shadows across the plates — these break detection
  and can confuse YOLO.
- **Background**: a plain wall behind the rack helps enormously.

---

## Advanced: Using a Different Model

If you have your own fine-tuned weights, point the tracker at them:

```bash
python barbell_tracker.py --video clip.mp4 --model path/to/your/best.pt
```

The tracker expects class names `plate` and `bar` in the model. Check with:

```bash
python barbell_tracker.py --model path/to/best.pt --list-classes
```

---

## Project Roadmap Ideas

- [ ] Session-level fatigue curve (MCV across sets)
- [ ] Rep quality score (path straightness + velocity consistency)
- [ ] Export to CSV for external analysis
- [ ] Simple web UI (Gradio or Streamlit)
- [ ] Automatic left/right plate tracking for side-by-side tilt comparison
