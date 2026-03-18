# Barbell Tracker — Project Context

## What This Does

Tracks barbell bar path, velocity, and rep metrics from lifting video (side-on footage).
Uses YOLOv8 + ByteTrack to detect the bar, segments reps, and outputs:
- Annotated video with bar path overlay
- 4-panel analysis plot (bar path, velocity, MCV per rep, bar tilt)
- `summary.json` with per-rep metrics

## Current Model

Roboflow `barbells-detector` dataset (92 images). Two classes:
- `Barbell` — bar body centroid, used as primary tracking target (detects reliably)
- `End` — plate end (unreliable with this model, not used for production tracking)

Running `--no-calibrate` because plate detection isn't reliable enough for auto-calibration. Outputs are in pixel units, not real m/s.

## Known Issues / Active Work

### 1. Ascent duration (partially solved, still needs work)

**Problem:** Ascent duration calculation was originally measuring from the bottom of the rep all the way to the start of the next rep, including rest time. This gave values of 3-7s instead of realistic 0.2-1.5s.

**Approach:** `find_concentric_end()` in `barbell_tracker.py` tries to detect where the bar settles after lockout. Current strategy:
1. Hard cap of 4s (no real concentric is longer)
2. Post-peak settle detection: find first point after peak velocity where bar stays below 3px/frame for 5 consecutive frames
3. Fallback to the 4s cap if settle not found

**Current issue:** On a 30fps heavy single (~1.36s actual ascent), the settle threshold (3px/frame) may trigger at the sticking point (where bar genuinely slows pre-lockout), or the cap kicks in giving 0.71s instead of ~1.36s. 60fps footage will help — more frames through the lift means smoother signal and better settle detection.

**What to try next:**
- Adaptive settle threshold based on mean velocity of the segment (not fixed 3px/frame)
- Look at the velocity profile shape rather than a fixed threshold
- Test with 60fps footage (user hasn't captured this yet)

### 2. Headless mode (not yet implemented)

Add `--headless` flag to suppress all `cv2.imshow()` calls so the tracker can run on the Pi (no display). Currently Mac-only because of the OpenCV preview window.

Goal: video recorded → uploaded to Pi → numbers auto-appended to training log.

### 3. Model retraining (not yet started)

The current model was trained on Roboflow's public dataset (92 images, mixed footage). To get:
- Reliable plate detection (needed for real mm/px calibration and true m/s velocity)
- Better detection in the user's specific setup (fixed camera, consistent background)

Plan: label ~100-150 frames of own footage in Roboflow, retrain, get proper plate detection.
User has a Roboflow account. Static background + consistent camera angle should train very well.

## Stack

- Python + ultralytics (YOLOv8) + supervision + OpenCV + scipy + matplotlib
- `venv` in repo root
- Model at `./models/barbells-detector/weights/best.pt`

## CLI Flags

```
--video              Path to input video (required)
--output             Output directory (default: ./output)
--model              YOLO model path
--conf               Detection confidence threshold (default: 0.3)
--no-calibrate       Skip auto-calibration (use when plate detection unreliable)
--smooth-window      Savitzky-Golay smoothing window in frames (default: 9, odd int)
--min-rep-amplitude  Min vertical travel in px to count as rep direction change (default: 60)
--api-key            Roboflow API key (downloads model if not present)
--list-classes       Print model class IDs and exit
```

## Useful Context

- User shoots side-on, fixed camera, consistent background — good conditions for retraining
- 30fps currently; 60fps planned for future sessions (will improve settle detection)
- Descent (~1s) is intentionally controlled; ascent is more explosive
- Velocity loss across a set (MCV increasing from ~0.17 to ~0.34s ascent time) is real and meaningful — VBT signal
- Repo: `hassantm/barbell-tracker` on GitHub
