#!/usr/bin/env python3
"""
barbell_tracker.py
------------------
Track barbell bar path, velocity, and tilt from lifting video.
Uses a Roboflow barbell-specific model that detects two classes:
  - 'End'      → plate end of barbell, used for position tracking and auto-calibration
  - 'Barbell'  → bar body centroid, used to compute bar tilt angle

Usage:
    python barbell_tracker.py --video path/to/video.mp4

Options:
    --video            Path to input video (required)
    --output           Output directory (default: ./output)
    --model            YOLO model path (default: ./models/barbells-detector/weights/best.pt)
    --plate-diameter   Plate diameter in mm (default: 450 for 20kg/45lb plate)
    --conf             Detection confidence threshold (default: 0.3)
    --no-calibrate     Skip auto-calibration; use 1.0 mm/px placeholder
    --api-key          Roboflow API key — if provided, model is downloaded before tracking
    --smooth-window    Savitzky-Golay smoothing window in frames (default: 9)
    --list-classes     Print all class IDs/names for the chosen model and exit

Auto-calibration:
    In the first 10 frames where a 'plate' detection occurs, the bounding-box
    width is measured.  mm_per_pixel = plate_diameter_mm / plate_bbox_width_px.
    The 10-frame average is used for the final scale.  Use --no-calibrate to
    bypass and use 1.0 mm/px instead.

Outputs (in --output directory):
    tracked.mp4     Annotated video with bar path overlay
    analysis.png    4-panel plot: bar path, velocity, MCV per rep, bar tilt
    summary.json    Rep-by-rep metrics including tilt
"""

import argparse
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Roboflow project defaults
# ---------------------------------------------------------------------------

ROBOFLOW_WORKSPACE = "yolo-project-c2bfs"
ROBOFLOW_PROJECT   = "barbells-detector"
ROBOFLOW_VERSION   = 1
ROBOFLOW_FORMAT    = "yolov8"

DEFAULT_MODEL_PATH = "./models/barbells-detector/weights/best.pt"

# Class names from the barbells-detector model
# 'Barbell' = bar body — primary tracking target (detected reliably in side-on footage)
# 'End' = plate end — used for tilt angle when detected (lower confidence in practice)
CLASS_PLATE = "Barbell"
CLASS_BAR   = "End"

# Number of frames to average for auto-calibration
CALIBRATION_FRAMES = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    mm_per_pixel: float
    method: str        # 'auto', 'manual_fallback', or 'no_calibrate'
    frames_used: int = 0


@dataclass
class RepData:
    rep_number: int
    frame_indices: list = field(default_factory=list)
    positions_px: list  = field(default_factory=list)   # plate centroid (x, y)
    bar_positions_px: list = field(default_factory=list) # bar centroid (x, y) or None
    tilt_angles_deg: list  = field(default_factory=list) # per-frame tilt (or None)
    timestamps_s: list     = field(default_factory=list)
    phase: list            = field(default_factory=list)  # 'descent' or 'ascent'


# ---------------------------------------------------------------------------
# Auto-calibration from plate bounding-box width
# ---------------------------------------------------------------------------

def auto_calibrate_from_plate(
    plate_bbox_widths: list,
    plate_diameter_mm: float,
) -> CalibrationResult:
    """
    Average the plate bounding-box widths collected over the first N frames
    and convert to mm/pixel.
    """
    if not plate_bbox_widths:
        print("WARNING: No plate detections found for calibration. "
              "Using 1.0 mm/px fallback.")
        return CalibrationResult(
            mm_per_pixel=1.0,
            method="manual_fallback",
            frames_used=0,
        )

    avg_width_px = float(np.mean(plate_bbox_widths))
    mpp = plate_diameter_mm / avg_width_px
    print(f"Auto-calibration: avg plate width = {avg_width_px:.1f} px  →  "
          f"{mpp:.4f} mm/pixel  (from {len(plate_bbox_widths)} frames)")
    return CalibrationResult(
        mm_per_pixel=mpp,
        method="auto",
        frames_used=len(plate_bbox_widths),
    )


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

def ensure_model(api_key: str, model_path: str):
    """Download model from Roboflow if --api-key was provided."""
    output_dir = Path("./models/barbells-detector")
    weights    = Path(model_path)

    if weights.exists():
        print(f"Model already present at {weights} — skipping download.")
        return

    print("Downloading model from Roboflow...")
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    project.version(ROBOFLOW_VERSION).download(
        model_format=ROBOFLOW_FORMAT,
        location=str(output_dir.parent),
        overwrite=True,
    )

    # Rename versioned folder to clean name
    roboflow_dir = output_dir.parent / f"{ROBOFLOW_PROJECT}-{ROBOFLOW_VERSION}"
    if roboflow_dir.exists() and roboflow_dir != output_dir:
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
        roboflow_dir.rename(output_dir)

    if weights.exists():
        print(f"Model saved: {weights.resolve()}")
    else:
        print(f"WARNING: Model download completed but weights not found at {weights}.")
        print("Use --model to specify the correct path.")


# ---------------------------------------------------------------------------
# Detection & tracking
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Load YOLO model via ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    print(f"Loading model: {model_name}")
    return YOLO(model_name)


def list_classes(model_name: str):
    model = load_model(model_name)
    print("\nClasses available in this model:")
    for idx, name in model.names.items():
        print(f"  {idx:3d}: {name}")
    sys.exit(0)


def compute_tilt_angle(plate_centroid: tuple, bar_centroid: tuple) -> float:
    """
    Compute bar tilt in degrees from horizontal.
    Uses the vector from plate centroid to bar centroid.
    Returns 0 when bar is perfectly horizontal.
    Positive = bar tilted upward toward bar end, negative = downward.
    """
    dx = bar_centroid[0] - plate_centroid[0]
    dy = bar_centroid[1] - plate_centroid[1]  # image y is inverted
    # atan2 gives angle relative to horizontal; invert dy for natural convention
    angle_rad = np.arctan2(-dy, dx)
    return float(np.degrees(angle_rad))


def track_video(
    video_path: str,
    model,
    conf: float,
    no_calibrate: bool,
    plate_diameter_mm: float,
    fps: float,
    smooth_window: int,
    min_rep_amplitude_px: int,
    output_dir: Path,
):
    """
    Main tracking loop.

    Returns (reps, calibration, annotated_video_path) or None on failure.

    Tracking strategy:
    - Primary position signal: plate centroid (most stable)
    - Bar centroid recorded when detected
    - Tilt angle computed per frame when both plate and bar are detected
    - Auto-calibration averaged over first CALIBRATION_FRAMES plate detections
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    out_path = output_dir / "tracked.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Raw per-frame data
    # Each entry: (frame_idx, plate_cx, plate_cy, bar_cx_or_None, bar_cy_or_None, tilt_or_None)
    raw_data = []

    # Auto-calibration accumulators
    plate_bbox_widths = []   # collect from first CALIBRATION_FRAMES plate detections

    try:
        import supervision as sv
    except ImportError:
        print("ERROR: supervision not installed. Run: pip install supervision")
        sys.exit(1)

    # Annotators for bounding boxes and labels
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_ann     = sv.LabelAnnotator(text_scale=0.5)

    # Build class-name → index map from the model
    class_name_to_id = {v: k for k, v in model.names.items()}
    plate_class_id   = class_name_to_id.get(CLASS_PLATE)
    bar_class_id     = class_name_to_id.get(CLASS_BAR)

    if plate_class_id is None:
        print(f"WARNING: '{CLASS_PLATE}' class not found in model. "
              f"Available: {list(model.names.values())}")
        # Fall back: use bar class as primary position signal
        if bar_class_id is not None:
            print(f"  → Falling back to '{CLASS_BAR}' as primary tracking target. "
                  f"Auto-calibration unavailable; use --no-calibrate.")
            plate_class_id = bar_class_id
            bar_class_id = None
    if bar_class_id is None:
        print(f"NOTE: '{CLASS_BAR}' class not found — tilt calculation skipped.")

    # Path points for trail drawing (plate centroids)
    path_points = []

    print("Tracking... (press Q in the preview window to stop early)\n")

    cv2.namedWindow("Barbell Tracker", cv2.WINDOW_NORMAL)

    tracker_args = dict(
        source=video_path,
        conf=conf,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    calibration = None  # will be set after collecting enough plate frames

    for result in model.track(**tracker_args):
        frame     = result.orig_img.copy()
        frame_idx += 1

        # -- Extract detections via supervision --
        detections = sv.Detections.from_ultralytics(result)

        plate_cx, plate_cy = None, None
        bar_cx,   bar_cy   = None, None
        tilt                = None

        if len(detections) > 0:
            # --- Plate: find highest-confidence detection of class 'plate' ---
            if plate_class_id is not None:
                plate_mask = detections.class_id == plate_class_id
                if plate_mask.any():
                    plate_dets = detections[plate_mask]
                    best       = np.argmax(plate_dets.confidence)
                    box        = plate_dets.xyxy[best]
                    plate_cx   = int((box[0] + box[2]) / 2)
                    plate_cy   = int((box[1] + box[3]) / 2)
                    bbox_w     = float(box[2] - box[0])

                    # Collect bbox widths for auto-calibration
                    if len(plate_bbox_widths) < CALIBRATION_FRAMES:
                        plate_bbox_widths.append(bbox_w)
                        if len(plate_bbox_widths) == CALIBRATION_FRAMES and calibration is None:
                            calibration = auto_calibrate_from_plate(
                                plate_bbox_widths, plate_diameter_mm
                            )

            # --- Bar: find highest-confidence detection of class 'bar' ---
            if bar_class_id is not None:
                bar_mask = detections.class_id == bar_class_id
                if bar_mask.any():
                    bar_dets = detections[bar_mask]
                    best     = np.argmax(bar_dets.confidence)
                    box      = bar_dets.xyxy[best]
                    bar_cx   = int((box[0] + box[2]) / 2)
                    bar_cy   = int((box[1] + box[3]) / 2)

            # --- Tilt: compute when both plate and bar are present ---
            if plate_cx is not None and bar_cx is not None:
                tilt = compute_tilt_angle(
                    (plate_cx, plate_cy), (bar_cx, bar_cy)
                )

        # Record frame data (only when we have a plate position)
        if plate_cx is not None:
            raw_data.append((frame_idx, plate_cx, plate_cy,
                             bar_cx, bar_cy, tilt))
            path_points.append((plate_cx, plate_cy))

        # -- Draw plate path trail (coloured by time) --
        if len(path_points) > 1:
            for i in range(1, len(path_points)):
                alpha  = i / len(path_points)
                colour = (int(255 * alpha), int(80 * alpha), int(200 * (1 - alpha)))
                cv2.line(frame, path_points[i - 1], path_points[i],
                         colour, 2, cv2.LINE_AA)

        # -- Annotate all detections with boxes and labels --
        if len(detections) > 0:
            labels = []
            for conf_val, cls_id in zip(detections.confidence, detections.class_id):
                name = model.names.get(cls_id, str(cls_id))
                labels.append(f"{name} {conf_val:.2f}")
            frame = box_annotator.annotate(frame, detections)
            frame = label_ann.annotate(frame, detections, labels)

        # -- Draw bar tilt if available --
        if tilt is not None and plate_cx is not None and bar_cx is not None:
            cv2.putText(frame, f"Tilt: {tilt:+.1f}°",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 50), 2)

        # -- Calibration status overlay --
        if calibration is None:
            n = len(plate_bbox_widths)
            cv2.putText(frame, f"Calibrating... {n}/{CALIBRATION_FRAMES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2)
        else:
            cv2.putText(frame, f"Cal: {calibration.mm_per_pixel:.3f} mm/px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

        # -- Frame counter --
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        writer.write(frame)
        cv2.imshow("Barbell Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped early by user.")
            break

    writer.release()
    cv2.destroyAllWindows()
    print(f"\nAnnotated video saved: {out_path}")

    # --- Finalise calibration ---
    if no_calibrate:
        calibration = CalibrationResult(
            mm_per_pixel=1.0, method="no_calibrate", frames_used=0
        )
    elif calibration is None:
        # Fewer than CALIBRATION_FRAMES plate detections — use whatever we have
        calibration = auto_calibrate_from_plate(plate_bbox_widths, plate_diameter_mm)

    if not raw_data:
        print("WARNING: No plate detections recorded. "
              "Check your model and that plates are visible in the video.")
        return None

    # --- Segment reps and compute metrics ---
    reps = segment_and_analyse(raw_data, calibration, fps, smooth_window,
                               min_rep_amplitude_px=min_rep_amplitude_px)
    return reps, calibration, out_path


# ---------------------------------------------------------------------------
# Rep segmentation & analysis
# ---------------------------------------------------------------------------

def find_concentric_end(
    ys_smooth: np.ndarray,
    start_idx: int,
    end_idx: int,
    fps: float,
    settle_frames: int = 5,
    settle_px_per_frame: float = 3.0,
) -> int:
    """
    Find the true end of the concentric (ascent) phase.

    Rather than locating the top of the lift geometrically (unreliable when
    the bar moves around at lockout), we detect when the bar *settles* —
    defined as the first point where per-frame speed stays below
    `settle_px_per_frame` for `settle_frames` consecutive frames.

    `settle_frames` default of 5 = ~167ms at 30fps, ~83ms at 60fps.
    `settle_px_per_frame` of 3.0px is well below meaningful bar movement
    but above detection noise.

    Falls back to end_idx if no settle point is found.
    """
    segment = ys_smooth[start_idx : end_idx + 1]
    if len(segment) < settle_frames + 2:
        return end_idx

    abs_vel = np.abs(np.diff(segment))
    slow    = abs_vel < settle_px_per_frame

    # Slide a window looking for settle_frames consecutive slow frames
    for i in range(len(slow) - settle_frames + 1):
        if slow[i : i + settle_frames].all():
            return start_idx + i  # first frame of the settled window

    return end_idx


def segment_and_analyse(
    raw_data: list,
    calibration: CalibrationResult,
    fps: float,
    smooth_window: int,
    min_rep_amplitude_px: int = 60,
) -> list:
    """
    Split the plate y-position signal into reps by finding direction reversals.
    Computes velocity from plate centroids and tilt stats from tilt_angles.
    Returns a list of RepData.
    """
    if len(raw_data) < 10:
        print("Too few detections for rep segmentation.")
        return []

    # Unpack raw_data columns
    frame_indices = [d[0] for d in raw_data]
    xs        = np.array([d[1] for d in raw_data], dtype=float)
    ys        = np.array([d[2] for d in raw_data], dtype=float)
    bar_xs    = [d[3] for d in raw_data]   # may be None
    bar_ys    = [d[4] for d in raw_data]   # may be None
    tilts_raw = [d[5] for d in raw_data]   # may be None

    # Smooth plate y-positions for rep detection
    win = smooth_window
    win = min(win, len(ys) - (0 if len(ys) % 2 == 1 else 1))
    win = win if win % 2 == 1 else win - 1
    win = max(win, 3)
    ys_smooth = savgol_filter(ys, window_length=win, polyorder=2)

    # Find direction changes (sign changes in first derivative of smoothed y)
    dy           = np.diff(ys_smooth)
    sign_changes = np.where(np.diff(np.sign(dy)))[0] + 1

    # Filter out tiny wiggles: require minimum vertical travel
    # Default 60px is roughly 10-15% of frame height — filters jitter while keeping real reps
    MIN_AMPLITUDE_PX = min_rep_amplitude_px
    filtered = []
    for idx in sign_changes:
        if not filtered:
            filtered.append(idx)
        else:
            amplitude = abs(ys_smooth[idx] - ys_smooth[filtered[-1]])
            if amplitude > MIN_AMPLITUDE_PX:
                filtered.append(idx)
    sign_changes = filtered

    print(f"Detected {len(sign_changes)} direction changes → "
          f"approximately {len(sign_changes) // 2} reps")

    # Build rep segments between consecutive reversals
    breakpoints = [0] + list(sign_changes) + [len(ys) - 1]
    reps        = []
    rep_num     = 0

    for i in range(0, len(breakpoints) - 2, 2):
        start = breakpoints[i]
        mid   = breakpoints[i + 1] if (i + 1) < len(breakpoints) else len(ys) - 1
        raw_end = breakpoints[i + 2] if (i + 2) < len(breakpoints) else len(ys) - 1

        # Trim the ascent to where the bar reaches its highest point
        # (minimum y in image coords) rather than the next reversal,
        # which would include the rest period between reps.
        end = find_concentric_end(ys_smooth, mid, raw_end, fps)

        rep_num += 1
        rd = RepData(rep_number=rep_num)

        for seg_start, seg_end, phase_label in [
            (start, mid, "descent"),
            (mid,   end, "ascent"),
        ]:
            for j in range(seg_start, seg_end + 1):
                rd.frame_indices.append(frame_indices[j])
                rd.positions_px.append((xs[j], ys_smooth[j]))
                rd.bar_positions_px.append(
                    (bar_xs[j], bar_ys[j]) if bar_xs[j] is not None else None
                )
                rd.tilt_angles_deg.append(tilts_raw[j])
                rd.timestamps_s.append(frame_indices[j] / fps)
                rd.phase.append(phase_label)

        # --- Velocity (from plate centroid) ---
        pos_arr = np.array(rd.positions_px)
        t_arr   = np.array(rd.timestamps_s)

        if len(pos_arr) > win:
            dy_rep = np.diff(pos_arr[:, 1]) * calibration.mm_per_pixel
            dt_rep = np.diff(t_arr)
            dt_rep[dt_rep == 0] = 1e-6
            vel    = dy_rep / dt_rep / 1000.0   # mm/s → m/s

            vel_win = min(win, len(vel) - (0 if len(vel) % 2 == 1 else 1))
            vel_win = max(vel_win if vel_win % 2 == 1 else vel_win - 1, 3)
            vel_smooth = savgol_filter(vel, window_length=vel_win, polyorder=2)

            # Mean and peak concentric (ascent) velocity
            ascent_mask = np.array(rd.phase[1:]) == "ascent"
            if ascent_mask.any():
                mcv    = float(np.mean(np.abs(vel_smooth[ascent_mask])))
                peak_v = float(np.max(np.abs(vel_smooth[ascent_mask])))
                rd.__dict__["mean_concentric_velocity"]  = round(mcv, 3)
                rd.__dict__["peak_concentric_velocity"]  = round(peak_v, 3)
            else:
                rd.__dict__["mean_concentric_velocity"]  = None
                rd.__dict__["peak_concentric_velocity"]  = None

            rd.__dict__["velocity_ms"]    = vel_smooth.tolist()
            rd.__dict__["velocity_times"] = t_arr[1:].tolist()
        else:
            rd.__dict__["mean_concentric_velocity"]  = None
            rd.__dict__["peak_concentric_velocity"]  = None
            rd.__dict__["velocity_ms"]               = []
            rd.__dict__["velocity_times"]            = []

        # --- Mean bar tilt during ascent ---
        ascent_tilts = [
            t for t, p in zip(rd.tilt_angles_deg, rd.phase)
            if p == "ascent" and t is not None
        ]
        if ascent_tilts:
            rd.__dict__["mean_bar_tilt_degrees"] = round(float(np.mean(ascent_tilts)), 2)
        else:
            rd.__dict__["mean_bar_tilt_degrees"] = None

        # --- Rep duration by phase ---
        t_arr = np.array(rd.timestamps_s)
        phases = np.array(rd.phase)
        descent_times = t_arr[phases == "descent"]
        ascent_times  = t_arr[phases == "ascent"]
        rd.__dict__["descent_duration_s"] = round(float(descent_times[-1] - descent_times[0]), 2) if len(descent_times) > 1 else None
        rd.__dict__["ascent_duration_s"]  = round(float(ascent_times[-1]  - ascent_times[0]),  2) if len(ascent_times)  > 1 else None
        rd.__dict__["total_duration_s"]   = round(float(t_arr[-1] - t_arr[0]), 2) if len(t_arr) > 1 else None

        reps.append(rd)

    return reps


# ---------------------------------------------------------------------------
# Output: analysis plots (4-panel)
# ---------------------------------------------------------------------------

def save_plots(
    reps: list,
    calibration: CalibrationResult,
    raw_data: list,
    fps: float,
    output_dir: Path,
):
    """
    4-panel dark-theme plot:
      1. Bar path (plate centroid, mm space)
      2. Velocity over time
      3. MCV per rep (bar chart + peak overlay)
      4. Bar tilt over time (all frames with tilt data)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if not reps:
        return

    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)

    colours = plt.cm.plasma(np.linspace(0.2, 0.9, len(reps)))

    # ------------------------------------------------------------------ #
    # Panel 1: Bar path (plate centroid x vs y in mm)
    # ------------------------------------------------------------------ #
    ax_path = fig.add_subplot(gs[:, 0])
    ax_path.set_facecolor("#16213e")
    ax_path.set_title("Bar Path (plate centroid, side view)", color="white", pad=10)
    ax_path.set_xlabel("Horizontal (mm)", color="grey")
    ax_path.set_ylabel("Vertical (mm)", color="grey")
    ax_path.tick_params(colors="grey")
    ax_path.invert_yaxis()   # y increases downward in image coordinates

    for rep, col in zip(reps, colours):
        pos = np.array(rep.positions_px) * calibration.mm_per_pixel
        ax_path.plot(pos[:, 0], pos[:, 1], color=col, linewidth=1.5,
                     label=f"Rep {rep.rep_number}")

    ax_path.legend(fontsize=7, labelcolor="white",
                   facecolor="#1a1a2e", edgecolor="none")

    # ------------------------------------------------------------------ #
    # Panel 2: Velocity over time
    # ------------------------------------------------------------------ #
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_vel.set_facecolor("#16213e")
    ax_vel.set_title("Bar Velocity (plate centroid)", color="white", pad=10)
    ax_vel.set_xlabel("Time (s)", color="grey")
    ax_vel.set_ylabel("Velocity (m/s)", color="grey")
    ax_vel.tick_params(colors="grey")
    ax_vel.axhline(0, color="#444", linewidth=0.8)

    for rep, col in zip(reps, colours):
        vel   = rep.__dict__.get("velocity_ms")
        times = rep.__dict__.get("velocity_times")
        if vel and times:
            ax_vel.plot(times, vel, color=col, linewidth=1.2,
                        label=f"Rep {rep.rep_number}")

    ax_vel.legend(fontsize=7, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="none")

    # ------------------------------------------------------------------ #
    # Panel 3: Mean concentric velocity per rep
    # ------------------------------------------------------------------ #
    ax_mcv = fig.add_subplot(gs[1, 1])
    ax_mcv.set_facecolor("#16213e")
    ax_mcv.set_title("Mean Concentric Velocity per Rep", color="white", pad=10)
    ax_mcv.set_xlabel("Rep", color="grey")
    ax_mcv.set_ylabel("MCV (m/s)", color="grey")
    ax_mcv.tick_params(colors="grey")

    mcv_vals  = [r.__dict__.get("mean_concentric_velocity") for r in reps]
    peak_vals = [r.__dict__.get("peak_concentric_velocity") for r in reps]
    rep_nums  = [r.rep_number for r in reps]

    valid = [(n, m, p) for n, m, p in zip(rep_nums, mcv_vals, peak_vals)
             if m is not None]
    if valid:
        ns, ms, ps = zip(*valid)
        ax_mcv.bar(ns, ms, color=[colours[i - 1] for i in ns],
                   alpha=0.8, label="MCV")
        ax_mcv.plot(ns, ps, "w--o", markersize=4, linewidth=1, label="Peak")
        ax_mcv.legend(fontsize=7, labelcolor="white",
                      facecolor="#1a1a2e", edgecolor="none")

    # ------------------------------------------------------------------ #
    # Panel 4: Bar tilt over time (all frames where tilt is available)
    # ------------------------------------------------------------------ #
    # Collect tilt data across all reps, preserving time ordering
    all_tilt_times  = []
    all_tilt_angles = []

    for rep in reps:
        for t, tilt in zip(rep.timestamps_s, rep.tilt_angles_deg):
            if tilt is not None:
                all_tilt_times.append(t)
                all_tilt_angles.append(tilt)

    # Also build a per-rep colour map for the tilt panel
    rep_tilt_data = {}
    for rep, col in zip(reps, colours):
        ts  = [t for t, a in zip(rep.timestamps_s, rep.tilt_angles_deg) if a is not None]
        ang = [a for a in rep.tilt_angles_deg if a is not None]
        if ts:
            rep_tilt_data[rep.rep_number] = (ts, ang, col)

    # Reuse gs space — add a new sub-axis below panel 3 using a nested GridSpec trick
    # Simpler: create the 4th axis as a subplot at position gs[1, 1] with shared x=time
    # We already used gs[1,1] for MCV.  Switch to a 2×2 grid where top row = path+velocity,
    # bottom row = MCV+tilt.  Re-do using gridspec positions:
    #   gs[0,0] = path (rowspan 2) — keep as is
    #   gs[0,1] = velocity
    #   gs[1,1] = MCV    → BUT we need 4 panels — restructure to 3-col or use nested
    #
    # The simplest fix: re-create the figure with a 2×3 layout where column 0 spans rows.
    # That changes figure dimensions, so instead we squeeze panels 2-4 into the right column
    # as a vertical stack of 3.  Rebuild with a new GridSpec.

    # Close the draft figure and rebuild with better layout
    plt.close(fig)

    fig   = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    gs    = gridspec.GridSpec(3, 2, figure=fig,
                              height_ratios=[1, 1, 1],
                              hspace=0.6, wspace=0.35)

    # Left column: bar path spans all 3 rows
    ax_path = fig.add_subplot(gs[:, 0])
    ax_vel  = fig.add_subplot(gs[0, 1])
    ax_mcv  = fig.add_subplot(gs[1, 1])
    ax_tilt = fig.add_subplot(gs[2, 1])

    # ---- Redraw panel 1: Bar path ----
    ax_path.set_facecolor("#16213e")
    ax_path.set_title("Bar Path (plate centroid, side view)", color="white", pad=10)
    ax_path.set_xlabel("Horizontal (mm)", color="grey")
    ax_path.set_ylabel("Vertical (mm)", color="grey")
    ax_path.tick_params(colors="grey")
    ax_path.invert_yaxis()

    for rep, col in zip(reps, colours):
        pos = np.array(rep.positions_px) * calibration.mm_per_pixel
        ax_path.plot(pos[:, 0], pos[:, 1], color=col, linewidth=1.5,
                     label=f"Rep {rep.rep_number}")
    ax_path.legend(fontsize=7, labelcolor="white",
                   facecolor="#1a1a2e", edgecolor="none")

    # ---- Redraw panel 2: Velocity ----
    ax_vel.set_facecolor("#16213e")
    ax_vel.set_title("Bar Velocity (plate centroid)", color="white", pad=8)
    ax_vel.set_xlabel("Time (s)", color="grey")
    ax_vel.set_ylabel("Velocity (m/s)", color="grey")
    ax_vel.tick_params(colors="grey")
    ax_vel.axhline(0, color="#444", linewidth=0.8)

    for rep, col in zip(reps, colours):
        vel   = rep.__dict__.get("velocity_ms")
        times = rep.__dict__.get("velocity_times")
        if vel and times:
            ax_vel.plot(times, vel, color=col, linewidth=1.2,
                        label=f"Rep {rep.rep_number}")
    ax_vel.legend(fontsize=7, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="none")

    # ---- Redraw panel 3: MCV ----
    ax_mcv.set_facecolor("#16213e")
    ax_mcv.set_title("Mean Concentric Velocity per Rep", color="white", pad=8)
    ax_mcv.set_xlabel("Rep", color="grey")
    ax_mcv.set_ylabel("MCV (m/s)", color="grey")
    ax_mcv.tick_params(colors="grey")

    if valid:
        ns, ms, ps = zip(*valid)
        ax_mcv.bar(ns, ms, color=[colours[i - 1] for i in ns], alpha=0.8, label="MCV")
        ax_mcv.plot(ns, ps, "w--o", markersize=4, linewidth=1, label="Peak")
        ax_mcv.legend(fontsize=7, labelcolor="white",
                      facecolor="#1a1a2e", edgecolor="none")

    # ---- Panel 4: Bar tilt ----
    ax_tilt.set_facecolor("#16213e")
    ax_tilt.set_title("Bar Tilt Over Time", color="white", pad=8)
    ax_tilt.set_xlabel("Time (s)", color="grey")
    ax_tilt.set_ylabel("Tilt (°)", color="grey")
    ax_tilt.tick_params(colors="grey")
    ax_tilt.axhline(0, color="#555", linewidth=0.8, linestyle="--")

    if rep_tilt_data:
        for rep_n, (ts, ang, col) in rep_tilt_data.items():
            ax_tilt.plot(ts, ang, color=col, linewidth=1.2, label=f"Rep {rep_n}")
        ax_tilt.legend(fontsize=7, labelcolor="white",
                       facecolor="#1a1a2e", edgecolor="none")
    else:
        ax_tilt.text(
            0.5, 0.5, "No tilt data\n(bar class not detected)",
            transform=ax_tilt.transAxes,
            ha="center", va="center", color="grey", fontsize=10,
        )

    for ax in [ax_path, ax_vel, ax_mcv, ax_tilt]:
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    plt.suptitle("Barbell Session Analysis", color="white", fontsize=14, y=1.01)

    plot_path = output_dir / "analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Analysis plot saved: {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# Save JSON summary
# ---------------------------------------------------------------------------

def save_summary(
    reps: list,
    calibration: CalibrationResult,
    output_dir: Path,
):
    summary = {
        "calibration": {
            "mm_per_pixel":  calibration.mm_per_pixel,
            "method":        calibration.method,
            "frames_used":   calibration.frames_used,
        },
        "reps": [],
    }

    for rep in reps:
        summary["reps"].append({
            "rep":                          rep.rep_number,
            "mean_concentric_velocity_ms":  rep.__dict__.get("mean_concentric_velocity"),
            "peak_concentric_velocity_ms":  rep.__dict__.get("peak_concentric_velocity"),
            "mean_bar_tilt_degrees":        rep.__dict__.get("mean_bar_tilt_degrees"),
            "descent_duration_s":           rep.__dict__.get("descent_duration_s"),
            "ascent_duration_s":            rep.__dict__.get("ascent_duration_s"),
            "total_duration_s":             rep.__dict__.get("total_duration_s"),
            "frame_count":                  len(rep.frame_indices),
        })

    out = output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON saved: {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Track barbell bar path, velocity, and tilt from lifting video."
    )
    parser.add_argument("--video",
                        required=False,
                        help="Path to input video")
    parser.add_argument("--output",
                        default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--model",
                        default=DEFAULT_MODEL_PATH,
                        help=f"YOLO model .pt file (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--plate-diameter",
                        type=float, default=450.0,
                        help="Plate diameter in mm (default: 450 for 20kg/45lb)")
    parser.add_argument("--conf",
                        type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--no-calibrate",
                        action="store_true",
                        help="Skip auto-calibration; use 1.0 mm/px placeholder")
    parser.add_argument("--api-key",
                        default=None,
                        help="Roboflow API key — downloads model if not already present")
    parser.add_argument("--smooth-window",
                        type=int, default=9,
                        help="Savitzky-Golay smoothing window in frames (odd int, default: 9)")
    parser.add_argument("--min-rep-amplitude",
                        type=int, default=60,
                        help="Minimum vertical travel in pixels to count as a rep direction change (default: 60)")
    parser.add_argument("--list-classes",
                        action="store_true",
                        help="Print all class IDs/names for the chosen model and exit")

    # Legacy flag kept for backwards compatibility (was used with old class filtering)
    parser.add_argument("--class-ids",
                        nargs="+", type=int, default=None,
                        help=argparse.SUPPRESS)   # hidden; ignored with the new model

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Optional: auto-download model from Roboflow ---
    if args.api_key:
        ensure_model(args.api_key, args.model)

    # --- Load model ---
    model = load_model(args.model)

    if args.list_classes:
        list_classes(args.model)   # exits

    if not args.video:
        print("ERROR: --video is required. Use --help for usage.")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Read FPS from video ---
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # --- Track ---
    result = track_video(
        video_path       = args.video,
        model            = model,
        conf             = args.conf,
        no_calibrate     = args.no_calibrate,
        plate_diameter_mm= args.plate_diameter,
        fps              = fps,
        smooth_window         = args.smooth_window,
        min_rep_amplitude_px  = args.min_rep_amplitude,
        output_dir            = output_dir,
    )

    if result is None:
        print("No detections. Exiting.")
        sys.exit(0)

    reps, calibration, video_out = result

    if not reps:
        print("No reps detected. Exiting.")
        sys.exit(0)

    # --- Console rep summary ---
    print("\n=== Rep Summary ===")
    print(f"{'Rep':>4}  {'Down (s)':>9}  {'Up (s)':>7}  {'MCV (m/s)':>10}  "
          f"{'Peak (m/s)':>11}  {'Ready?':>7}")
    print("-" * 58)
    for rep in reps:
        mcv     = rep.__dict__.get("mean_concentric_velocity")
        peak    = rep.__dict__.get("peak_concentric_velocity")
        down    = rep.__dict__.get("descent_duration_s")
        up      = rep.__dict__.get("ascent_duration_s")
        # "ready to go up" heuristic: concentric under 2s
        ready   = "✓ YES" if (up is not None and up < 2.0) else ("  no" if up is not None else "  n/a")
        print(
            f"{rep.rep_number:>4}  "
            f"{str(round(down, 2)) if down is not None else 'n/a':>9}  "
            f"{str(round(up, 2)) if up is not None else 'n/a':>7}  "
            f"{str(round(mcv, 3)) if mcv is not None else 'n/a':>10}  "
            f"{str(round(peak, 3)) if peak is not None else 'n/a':>11}  "
            f"{ready:>7}"
        )

    # Collect raw_data for tilt plotting (rebuild from reps)
    # (track_video returns reps; raw_data needed for plot is embedded in reps)
    raw_data_for_plot = []  # unused in save_plots — plots draw from reps directly

    # --- Plots & JSON ---
    save_plots(reps, calibration, raw_data_for_plot, fps, output_dir)
    save_summary(reps, calibration, output_dir)

    print("\nDone! Outputs in:", output_dir.resolve())


if __name__ == "__main__":
    main()
