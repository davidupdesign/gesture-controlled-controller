# Gesture-Controlled Presentation Controller — Project Spec

## Goal

Control an **existing, already-open presentation** (PowerPoint, Keynote, Google Slides in a browser, PDF viewer) using hand gestures captured from the webcam. The tool is a background controller that injects OS-level keystrokes into whatever window has focus. It is **not** a slideshow viewer.

Target platform for v1: macOS (Apple Silicon, M4 Max). Keep the keystroke layer abstracted so Windows/Linux can be added later.

---

## Stack

| Component | Role |
|---|---|
| **Python 3.11 or 3.12** | Runtime. Avoid 3.13 — MediaPipe wheels are unreliable there. |
| **OpenCV (`opencv-python`)** | Webcam capture loop, BGR→RGB conversion, debug preview window, exit key handling. |
| **MediaPipe (`mediapipe`)** | Hand detection. Returns 21 normalized 3D landmarks per hand + handedness + confidence. Use the **Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker`), not the legacy `mp.solutions.hands` API. |
| **pyautogui** | OS-level keystroke injection (`press('right')`, `press('left')`, etc.). This is the piece that makes the tool work with third-party apps. |
| **Custom logic** | Gesture classification. MediaPipe gives coordinates, not gestures. This layer is the entire quality of the product. |

Do **not** use `cvzone`. It wraps MediaPipe with convenience helpers but hides raw landmark data, pins outdated MediaPipe versions, and causes install conflicts on Apple Silicon.

---

## Pipeline

```
OpenCV VideoCapture → frame (BGR)
  → cv2.cvtColor BGR→RGB
  → MediaPipe HandLandmarker → 21 landmarks (normalized 0–1)
  → landmark buffer (ring buffer, last N frames)
  → gesture classifier (pose + motion)
  → arming / cooldown state machine
  → action mapper → pyautogui keystroke
```

---

## Gesture design

Two detection strategies. Implement both; the classifier combines them.

**Static pose** — which fingers are extended, derived from landmark y-positions relative to their PIP joints. Cheap, stable, but unnatural to hold and prone to firing on incidental pointing during speech.

**Motion / swipe** — buffer wrist x-position over the last N frames, compute displacement and velocity. Fire when displacement exceeds a normalized threshold within a time window. Natural to perform, but noisier.

### Proposed mapping (v1)

| Gesture | Action | Key |
|---|---|---|
| Open palm swipe right | Next slide | `right` |
| Open palm swipe left | Previous slide | `left` |
| Closed fist held ~1s | Toggle blank/black screen | `b` |
| Open palm held ~1s | Arm/disarm the controller | — (internal) |

Keep the mapping in a config dict so gestures can be remapped without touching detection code.

---

## The hard problems

These are the parts that determine whether this is usable or a demo. Solve them explicitly.

**False positives during natural speech.** A presenter gestures constantly while talking. Naive swipe detection will fire repeatedly. Mitigations, layered:
- *Arming*: controller is disarmed by default. A distinct arming gesture (open palm held above a y-threshold for ~1s) enables recognition for a short window, then auto-disarms. This is the single most important design decision in the project.
- *Height gate*: only recognize gestures when the wrist is above a configurable y-line near the top of the frame. Natural talking gestures happen lower.
- *Pose gate*: require a specific hand shape (open palm, all five fingers extended) throughout the swipe. Filters out most conversational motion.
- *Confidence threshold*: discard frames below a detection confidence floor.

**Debouncing.** One physical swipe spans 10–20 frames. Without a cooldown, one swipe fires 15 slide advances. Implement a state machine with an explicit `IDLE → DETECTING → FIRED → COOLDOWN` cycle and a cooldown of ~800ms–1.2s.

**Latency.** Must feel instant, target sub-200ms from gesture completion to keypress. Budget: capture + inference + buffer window. Downscale frames before inference if needed. Run inference on every frame; do not skip frames, it corrupts velocity calculation.

**Focus.** `pyautogui` sends keys to whatever window is focused. If the debug preview window has focus, the script keys itself. Either run headless in production, or make the preview a non-activating always-on-top overlay.

**Lighting.** Webcam hand tracking degrades badly in dim or heavily backlit rooms — exactly the conditions of a presentation venue. Test this early. Log detection confidence so degradation is diagnosable.

---

## macOS environment requirements

Both permissions are granted to the *host application* running the script (Terminal, iTerm, or Cursor), not to Python itself.

- **System Settings → Privacy & Security → Accessibility** — required for `pyautogui`. Without it the script runs silently, reports no error, and no keystrokes land. This is the most common source of "it doesn't work."
- **System Settings → Privacy & Security → Camera** — required for OpenCV capture.

If you switch from running in Terminal to running in Cursor, you must grant both again for the new host app.

---

## Suggested build order

1. Webcam capture loop + preview window + clean exit. Verify frame rate.
2. MediaPipe HandLandmarker integrated. Draw the 21 landmarks on the preview. Confirm tracking is stable at your typical presenting distance (2–4m from camera, which is far — test it, it may force a camera repositioning or a resolution bump).
3. Static pose classifier (`fingers_up()` → 5-element boolean array). Display the result on the preview.
4. Landmark ring buffer + velocity computation. Display live velocity on the preview.
5. Swipe classifier with thresholds exposed as constants at the top of the file.
6. Arming state machine + cooldown.
7. `pyautogui` action layer, abstracted behind an `ActionDispatcher` class so the keystroke backend is swappable.
8. Config file for thresholds, gesture mapping, and camera index.
9. Headless mode flag.

Keep everything in one file until step 6, then split into `capture.py`, `gestures.py`, `actions.py`, `config.py`.

---

## Non-goals for v1

- Building a slideshow viewer. The tool drives existing presentation software.
- Slide annotation, drawing, laser pointer overlay.
- Multi-hand gestures.
- Windows/Linux support (but keep the action layer abstracted for it).
- Packaging/distribution.
