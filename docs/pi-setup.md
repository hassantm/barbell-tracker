# Raspberry Pi Setup

## Goal

Video recorded at the gym → uploaded to Pi → numbers auto-appended to training log.

## Install

```bash
# Headless OpenCV (no display dependencies)
pip install opencv-python-headless ultralytics supervision scipy matplotlib

# Do NOT install opencv-python (requires display libraries)
```

## Run

```bash
python barbell_tracker.py \
  --video /path/to/clip.MOV \
  --no-calibrate \
  --headless
```

- `--headless` suppresses all preview windows.
- `--no-calibrate` skips plate-size calibration (use until model retraining gives reliable `End` detections).
- Outputs go to `./output/` by default: `tracked.mp4`, `analysis.png`, `summary.json`.

## Auto-run on Upload (inotifywait)

Install `inotify-tools`, then create a watcher script:

```bash
#!/usr/bin/env bash
# watch-and-track.sh
INBOX=/home/pi/barbell-inbox
OUTPUT=/home/pi/barbell-output

inotifywait -m -e close_write "$INBOX" --format '%f' | while read FNAME; do
    EXT="${FNAME##*.}"
    if [[ "$EXT" =~ ^(mp4|mov|MOV|MP4)$ ]]; then
        echo "New video: $FNAME"
        python /home/pi/barbell-tracker/barbell_tracker.py \
            --video "$INBOX/$FNAME" \
            --output "$OUTPUT/${FNAME%.*}" \
            --no-calibrate \
            --headless
        echo "Done: $OUTPUT/${FNAME%.*}/summary.json"
    fi
done
```

Make executable and run at boot via a systemd unit or `@reboot` cron entry:

```
@reboot /home/pi/watch-and-track.sh >> /home/pi/barbell-tracker.log 2>&1
```

## Tuning on Pi

If ascent durations look wrong, adjust the settle threshold:

```bash
python barbell_tracker.py --video clip.MOV --headless --settle-fraction 0.10
```

Lower `--settle-fraction` = tighter settle threshold = longer measured ascents.
Default is 0.15; range 0.10–0.20 is sensible.

## Pi 5 Notes

- Pi 5 can run `yolov8s` (~200ms/frame) once the model is retrained.
- Current `yolov8n` model runs faster but is less accurate.
- For 30fps video, real-time tracking is not needed — batch processing is fine.
