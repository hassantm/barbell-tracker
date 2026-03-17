#!/usr/bin/env python3
"""
barbell_tracker.py
------------------
Track barbell bar path and velocity from lifting video.

Usage:
    python barbell_tracker.py --video path/to/video.mp4

Options:
    --video            Path to input video (required)
    --output           Output directory (default: ./output)
    --model            YOLO model path or name (default: yolov8n.pt)
    --plate-diameter   Plate diameter in mm (default: 450 for 20kg/45lb plate)
    --conf             Detection confidence threshold (default: 0.3)
    --class-ids        YOLO class IDs to track, space-separated (default: auto-detect)
    --no-calibrate     Skip interactive scale calibration (use estimated scale only)
    --smooth-window    Savitzky-Golay smoothing window in frames (default: 9)

Notes on the YOLO model:
    The default yolov8n.pt is a general model — it won't detect barbells by name.
    For best results, use a barbell-specific model from Roboflow:
        from roboflow import Roboflow
        rf = Roboflow(api_key="YOUR_KEY")
        model = rf.workspace().project("barbell-detection").version(1).download("yolov8")
    
    With the general model, use --class-ids to target relevant detected classes.
    Run `python barbell_tracker.py --list-classes` to see what your model detects.
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
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    mm_per_pixel: float
    ref_point_a: tuple
    ref_point_b: tuple
    pixel_distance: float


@dataclass
class RepData:
    rep_number: int
    frame_indices: list = field(default_factory=list)
    positions_px: list = field(default_factory=list)   # (x, y) in pixels
    timestamps_s: list = field(default_factory=list)
    phase: list = field(default_factory=list)          # 'descent' or 'ascent'


# ---------------------------------------------------------------------------
# Interactive calibration
# ---------------------------------------------------------------------------

class Calibrator:
    """Click two points on the plate rim to establish mm/pixel scale."""

    def __init__(self):
        self.points = []
        self.done = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 2:
            self.points.append((x, y))
            print(f"  Point {len(self.points)}: ({x}, {y})")
        if len(self.points) == 2:
            self.done = True

    def run(self, frame: np.ndarray, plate_diameter_mm: float) -> CalibrationResult:
        print("\n--- Scale Calibration ---")
        print(f"Click the two OUTERMOST edges of one plate (diameter = {plate_diameter_mm}mm).")
        print("Press 'r' to reset points, 'q' to accept.\n")

        win = "Calibration — click plate edges, then press Q"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self.mouse_callback)

        display = frame.copy()

        while True:
            vis = display.copy()

            for pt in self.points:
                cv2.circle(vis, pt, 8, (0, 255, 255), -1)

            if len(self.points) == 2:
                cv2.line(vis, self.points[0], self.points[1], (0, 255, 255), 2)
                px_dist = np.linalg.norm(
                    np.array(self.points[0]) - np.array(self.points[1])
                )
                mpp = plate_diameter_mm / px_dist
                cv2.putText(vis, f"{mpp:.3f} mm/px — press Q to accept",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('r'):
                self.points = []
                self.done = False
            elif key == ord('q') and len(self.points) == 2:
                break

        cv2.destroyWindow(win)

        px_dist = np.linalg.norm(
            np.array(self.points[0]) - np.array(self.points[1])
        )
        mpp = plate_diameter_mm / px_dist
        print(f"Scale set: {mpp:.4f} mm/pixel  ({px_dist:.1f} px = {plate_diameter_mm}mm)\n")

        return CalibrationResult(
            mm_per_pixel=mpp,
            ref_point_a=self.points[0],
            ref_point_b=self.points[1],
            pixel_distance=px_dist,
        )


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


def track_video(
    video_path: str,
    model,
    conf: float,
    class_filter: Optional[list],
    calibration: CalibrationResult,
    fps: float,
    smooth_window: int,
    output_dir: Path,
):
    """Main tracking loop. Returns list of RepData."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    out_path = output_dir / "tracked.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    raw_positions  = []   # (frame_idx, cx, cy)
    path_points    = []   # drawn on each frame

    frame_idx = 0
    import supervision as sv

    # Supervision annotators
    box_annotator  = sv.BoxAnnotator(thickness=2)
    label_ann      = sv.LabelAnnotator(text_scale=0.5)
    trace_ann      = sv.TraceAnnotator(thickness=2, trace_length=120, color=sv.Color.RED)

    print("Tracking... (press Q in the preview window to stop early)\n")

    # We run ultralytics tracking frame-by-frame so we can annotate ourselves
    tracker_args = dict(
        source=video_path,
        conf=conf,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )
    if class_filter:
        tracker_args["classes"] = class_filter

    cap.release()  # ultralytics will re-open

    cv2.namedWindow("Barbell Tracker", cv2.WINDOW_NORMAL)

    for result in model.track(**tracker_args):
        frame = result.orig_img.copy()
        frame_idx += 1

        # -- Draw calibration reference line --
        if calibration:
            cv2.line(frame, calibration.ref_point_a, calibration.ref_point_b,
                     (255, 220, 0), 1, cv2.LINE_AA)

        # -- Extract detections via supervision --
        detections = sv.Detections.from_ultralytics(result)

        if len(detections) > 0:
            # Pick the highest-confidence detection in each frame
            # (single-object tracking — the barbell)
            best_idx = np.argmax(detections.confidence)
            best_box = detections.xyxy[best_idx]
            cx = int((best_box[0] + best_box[2]) / 2)
            cy = int((best_box[1] + best_box[3]) / 2)
            raw_positions.append((frame_idx, cx, cy))
            path_points.append((cx, cy))

        # -- Draw bar path trail --
        if len(path_points) > 1:
            for i in range(1, len(path_points)):
                alpha = i / len(path_points)
                colour = (int(255 * alpha), int(80 * alpha), int(200 * (1 - alpha)))
                cv2.line(frame, path_points[i - 1], path_points[i], colour, 2, cv2.LINE_AA)

        # -- Annotate boxes and labels --
        if len(detections) > 0:
            labels = []
            for conf_val, cls_id in zip(detections.confidence, detections.class_id):
                labels.append(f"{model.names[cls_id]} {conf_val:.2f}")
            frame = box_annotator.annotate(frame, detections)
            frame = label_ann.annotate(frame, detections, labels)

        # -- Frame counter --
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        writer.write(frame)

        cv2.imshow("Barbell Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped early by user.")
            break

    writer.release()
    cv2.destroyAllWindows()
    print(f"\nAnnotated video saved: {out_path}")

    if not raw_positions:
        print("WARNING: No detections recorded. Check your model and --class-ids setting.")
        return []

    # -- Segment reps and compute velocity --
    reps = segment_and_analyse(raw_positions, calibration, fps, smooth_window)
    return reps, out_path


# ---------------------------------------------------------------------------
# Rep segmentation
# ---------------------------------------------------------------------------

def segment_and_analyse(
    raw_positions: list,
    calibration: CalibrationResult,
    fps: float,
    smooth_window: int,
) -> list:
    """
    Split the y-position signal into reps by finding direction reversals.
    Returns a list of RepData with velocity computed per phase.
    """
    if len(raw_positions) < 10:
        print("Too few detections for rep segmentation.")
        return []

    frame_indices = [p[0] for p in raw_positions]
    xs = np.array([p[1] for p in raw_positions], dtype=float)
    ys = np.array([p[2] for p in raw_positions], dtype=float)

    # Smooth y-positions
    win = min(smooth_window, len(ys) - (1 if len(ys) % 2 == 0 else 0))
    win = win if win % 2 == 1 else win - 1
    win = max(win, 3)
    ys_smooth = savgol_filter(ys, window_length=win, polyorder=2)

    # Find direction changes (sign changes in first derivative)
    dy = np.diff(ys_smooth)
    sign_changes = np.where(np.diff(np.sign(dy)))[0] + 1  # frame indices of reversals

    # Filter out tiny wiggles — require movement of at least 10px
    MIN_AMPLITUDE_PX = 10
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
    reps = []
    rep_num = 0

    for i in range(0, len(breakpoints) - 2, 2):
        # Each rep = one descent + one ascent
        start = breakpoints[i]
        mid   = breakpoints[i + 1] if i + 1 < len(breakpoints) else len(ys) - 1
        end   = breakpoints[i + 2] if i + 2 < len(breakpoints) else len(ys) - 1

        rep_num += 1
        rd = RepData(rep_number=rep_num)

        for seg_start, seg_end, phase_label in [(start, mid, "descent"), (mid, end, "ascent")]:
            for j in range(seg_start, seg_end + 1):
                rd.frame_indices.append(frame_indices[j])
                rd.positions_px.append((xs[j], ys_smooth[j]))
                rd.timestamps_s.append(frame_indices[j] / fps)
                rd.phase.append(phase_label)

        # Compute velocity for this rep
        pos_arr = np.array(rd.positions_px)
        t_arr   = np.array(rd.timestamps_s)

        if len(pos_arr) > win:
            dy_rep = np.diff(pos_arr[:, 1]) * calibration.mm_per_pixel
            dt_rep = np.diff(t_arr)
            dt_rep[dt_rep == 0] = 1e-6
            vel = dy_rep / dt_rep / 1000.0  # mm/s → m/s

            # Smooth velocity
            vel_win = min(win, len(vel) - (0 if len(vel) % 2 == 1 else 1))
            vel_win = max(vel_win if vel_win % 2 == 1 else vel_win - 1, 3)
            vel_smooth = savgol_filter(vel, window_length=vel_win, polyorder=2)

            # Mean concentric (ascent) velocity
            ascent_mask = np.array(rd.phase[1:]) == "ascent"
            if ascent_mask.any():
                mcv = float(np.mean(np.abs(vel_smooth[ascent_mask])))
                peak_v = float(np.max(np.abs(vel_smooth[ascent_mask])))
                rd.__dict__["mean_concentric_velocity"] = round(mcv, 3)
                rd.__dict__["peak_concentric_velocity"] = round(peak_v, 3)
            else:
                rd.__dict__["mean_concentric_velocity"] = None
                rd.__dict__["peak_concentric_velocity"] = None

            rd.__dict__["velocity_ms"]    = vel_smooth.tolist()
            rd.__dict__["velocity_times"] = t_arr[1:].tolist()

        reps.append(rd)

    return reps


# ---------------------------------------------------------------------------
# Output: plots
# ---------------------------------------------------------------------------

def save_plots(reps: list, calibration: CalibrationResult, output_dir: Path):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if not reps:
        return

    fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    colours = plt.cm.plasma(np.linspace(0.2, 0.9, len(reps)))

    # --- Bar path (x vs y in mm) ---
    ax_path = fig.add_subplot(gs[:, 0])
    ax_path.set_facecolor("#16213e")
    ax_path.set_title("Bar Path (side view)", color="white", pad=10)
    ax_path.set_xlabel("Horizontal (mm)", color="grey")
    ax_path.set_ylabel("Vertical (mm)", color="grey")
    ax_path.tick_params(colors="grey")
    ax_path.invert_yaxis()  # y increases downward in image coords

    for rep, col in zip(reps, colours):
        pos = np.array(rep.positions_px) * calibration.mm_per_pixel
        ax_path.plot(pos[:, 0], pos[:, 1], color=col, linewidth=1.5,
                     label=f"Rep {rep.rep_number}")

    ax_path.legend(fontsize=7, labelcolor="white",
                   facecolor="#1a1a2e", edgecolor="none")

    # --- Velocity over time ---
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_vel.set_facecolor("#16213e")
    ax_vel.set_title("Bar Velocity", color="white", pad=10)
    ax_vel.set_xlabel("Time (s)", color="grey")
    ax_vel.set_ylabel("Velocity (m/s)", color="grey")
    ax_vel.tick_params(colors="grey")
    ax_vel.axhline(0, color="#444", linewidth=0.8)

    for rep, col in zip(reps, colours):
        vel = rep.__dict__.get("velocity_ms")
        times = rep.__dict__.get("velocity_times")
        if vel and times:
            ax_vel.plot(times, vel, color=col, linewidth=1.2,
                        label=f"Rep {rep.rep_number}")

    ax_vel.legend(fontsize=7, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="none")

    # --- Mean concentric velocity per rep ---
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

    for ax in [ax_path, ax_vel, ax_mcv]:
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    plt.suptitle("Barbell Session Analysis", color="white", fontsize=14, y=1.01)

    plot_path = output_dir / "analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Analysis plot saved: {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# Save JSON summary
# ---------------------------------------------------------------------------

def save_summary(reps: list, calibration: CalibrationResult, output_dir: Path):
    summary = {
        "calibration": {
            "mm_per_pixel": calibration.mm_per_pixel,
        },
        "reps": []
    }
    for rep in reps:
        summary["reps"].append({
            "rep": rep.rep_number,
            "mean_concentric_velocity_ms": rep.__dict__.get("mean_concentric_velocity"),
            "peak_concentric_velocity_ms": rep.__dict__.get("peak_concentric_velocity"),
            "frame_count": len(rep.frame_indices),
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
        description="Track barbell bar path and velocity from lifting video."
    )
    parser.add_argument("--video",    required=False, help="Path to input video")
    parser.add_argument("--output",   default="./output", help="Output directory")
    parser.add_argument("--model",    default="yolov8n.pt",
                        help="YOLO model file (default: yolov8n.pt)")
    parser.add_argument("--plate-diameter", type=float, default=450.0,
                        help="Plate diameter in mm (default: 450 for 20kg/45lb)")
    parser.add_argument("--conf",     type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--class-ids", nargs="+", type=int, default=None,
                        help="YOLO class IDs to track (space-separated)")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip interactive calibration; use 1.0 mm/px as placeholder")
    parser.add_argument("--smooth-window", type=int, default=9,
                        help="Savitzky-Golay smoothing window in frames (odd int, default: 9)")
    parser.add_argument("--list-classes", action="store_true",
                        help="Print all class IDs/names for the chosen model and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model(args.model)

    if args.list_classes:
        list_classes(args.model)

    if not args.video:
        print("ERROR: --video is required. Use --help for usage.")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Open first frame for calibration ---
    cap = cv2.VideoCapture(args.video)
    ret, first_frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if not ret:
        print(f"ERROR: Cannot read first frame from {args.video}")
        sys.exit(1)

    # --- Calibration ---
    if args.no_calibrate:
        print("Skipping calibration — using 1.0 mm/px placeholder.")
        calibration = CalibrationResult(
            mm_per_pixel=1.0,
            ref_point_a=(0, 0),
            ref_point_b=(0, 0),
            pixel_distance=args.plate_diameter,
        )
    else:
        calibrator = Calibrator()
        calibration = calibrator.run(first_frame, args.plate_diameter)

    # --- Track ---
    result = track_video(
        video_path    = args.video,
        model         = model,
        conf          = args.conf,
        class_filter  = args.class_ids,
        calibration   = calibration,
        fps           = fps,
        smooth_window = args.smooth_window,
        output_dir    = output_dir,
    )

    if not result:
        print("No reps detected. Exiting.")
        sys.exit(0)

    reps, video_out = result

    # --- Print rep summary to console ---
    print("\n=== Rep Summary ===")
    print(f"{'Rep':>4}  {'Frames':>7}  {'MCV (m/s)':>10}  {'Peak (m/s)':>11}")
    print("-" * 38)
    for rep in reps:
        mcv  = rep.__dict__.get("mean_concentric_velocity")
        peak = rep.__dict__.get("peak_concentric_velocity")
        print(f"{rep.rep_number:>4}  {len(rep.frame_indices):>7}  "
              f"{mcv if mcv is not None else 'n/a':>10}  "
              f"{peak if peak is not None else 'n/a':>11}")

    # --- Plots & JSON ---
    save_plots(reps, calibration, output_dir)
    save_summary(reps, calibration, output_dir)

    print("\nDone! Outputs in:", output_dir.resolve())


if __name__ == "__main__":
    main()
