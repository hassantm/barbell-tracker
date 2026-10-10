# Improvement Log

## 2026-03-19 — Ascent Duration Fix

### Problem
`find_concentric_end()` used a hardcoded 3.0 px/frame settle threshold. On a heavy single the bar genuinely slows at the sticking point (2–4 px/frame), triggering a false "settled" signal mid-lift. A 30fps heavy squat single with ~1.36s actual ascent was measured as 0.71s.

### Root Cause
- Fixed `settle_px_per_frame=3.0` was too high relative to the sticking-point velocity on slow, heavy reps.
- No displacement check: the settle window only tested speed, not whether the bar had actually stopped moving.

### Changes (`barbell_tracker.py`)
1. **Adaptive threshold** — `adaptive_settle = max(mean_vel * settle_fraction, 1.5)` replaces the hardcoded 3.0. At a typical heavy single (mean ~8 px/frame, `settle_fraction=0.15`), threshold = max(1.2, 1.5) = 1.5 px/frame — well below typical sticking-point velocity.
2. **Displacement confirmation** — a candidate settle window is only accepted if the net bar displacement over that window is < 8 px. A sticking point has net upward displacement; true lockout has near-zero net movement.
3. **`--settle-fraction` CLI flag** — default 0.15; tune without code changes. Lower = tighter threshold.
4. **`verbose` param** — `find_concentric_end(..., verbose=True)` prints mean_vel, adaptive_settle, and detected frame offset for debugging.

### Before / After (target)
| Clip | Actual ascent | Before | After (target) |
|------|--------------|--------|----------------|
| 20260316-squats-top-sgl.MOV | ~1.36s | 0.71s | 1.21–1.51s |

*Run validation after capturing 60fps footage to confirm improvement.*

---

## 2026-03-19 — Headless Mode

### Problem
`cv2.imshow()` calls made the tracker Mac-only. Running on Raspberry Pi (no display) was impossible.

### Changes (`barbell_tracker.py`)
- `--headless` flag suppresses all `cv2.imshow()`, `cv2.namedWindow()`, `cv2.destroyAllWindows()` calls.
- Headless mode prints frame progress every 50 frames instead (`Frame X/Y (Z%)`).
- `opencv-python-headless` recommended for Pi installs (see `docs/pi-setup.md`).
