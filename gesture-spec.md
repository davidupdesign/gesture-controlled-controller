# Gesture-Controlled Presentation Controller — Project Spec (v2)

## Goal

Control an **existing, already-open presentation** (PowerPoint, Keynote, Google Slides in a browser, PDF viewer) using hand gestures captured from the webcam. The tool is a background controller that injects OS-level keystrokes into whatever window has focus. It is **not** a slideshow viewer.

Target platform for v1: macOS (Apple Silicon, M4 Max). Keep the keystroke layer abstracted so Windows/Linux can be added later.

---

## Measure this before writing any code

**What is the actual camera-to-presenter distance and lighting in the real use case?**

Everything downstream depends on this number and it is currently a guess. A presenter standing at a lectern with a laptop is often 0.8–1.5m from the camera. A presenter roaming a stage is 3–4m. These are different projects:

| Measured distance | Consequence |
|---|---|
| < 1.5m | Full-frame inference works. Build as specced. |
| 1.5–3m | ROI cropping is mandatory (see *Detection at distance*). Thresholds must be hand-size normalized. |
| > 3m | Laptop webcam is not viable. Requires an external camera with a longer focal length, repositioned closer, or a different input modality entirely. |

Go stand where you will actually present, and measure. Record the answer here before proceeding:

```
MEASURED_DISTANCE_M   = 1.5
VENUE_LIGHTING        = ____  (still untested - see Lighting under The hard problems)
CAMERA                = built-in (MacBook Pro Camera)
ROI_CROP_REQUIRED     = False  # settled by Spike B, 2026-08-23
```

---

## Step 0 — Spikes (gate before building)

Two binary risks decide whether this project is possible. Both are throwaway scripts. Do them first; neither takes an hour.

**Spike A — Keystroke injection (~10 min).**
```python
import pyautogui, time
pyautogui.PAUSE = 0
time.sleep(3)          # focus Keynote/Slides during this window
pyautogui.press('right')
```
Proves macOS Accessibility permission is correctly granted to the host app end-to-end. *Kill criterion:* the keystroke does not land even after granting Accessibility to the terminal/editor you launched from and fully restarting it.

**Spike B — Tracking at real distance (~30 min).**
Run MediaPipe HandLandmarker on the webcam and draw the 21 landmarks, standing at `MEASURED_DISTANCE_M` under `VENUE_LIGHTING`. Watch stability and logged confidence for 60 seconds of natural movement.

*Outcome:* sets `ROI_CROP_REQUIRED`. If full-frame inference is unstable, retry with a center-region crop upscaled before inference. If it fails even cropped, the camera or the geometry has to change — that is a project-reshaping result and it is much cheaper to learn now.

Both spikes are disposable. Do not try to grow them into the real thing.

---

## Stack

| Component | Role |
|---|---|
| **Python 3.11 or 3.12** | Runtime. Avoid 3.13 — MediaPipe wheels are unreliable there. |
| **OpenCV (`opencv-python`)** | Webcam capture, colour conversion, ROI crop, status HUD rendering. |
| **MediaPipe (`mediapipe`)** | Hand detection. 21 normalized 3D landmarks per hand + handedness + confidence. Use the **Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker`), not the legacy `mp.solutions.hands` API. |
| **`hand_landmarker.task`** | The Tasks API requires an explicitly downloaded model bundle and a path to it. Vendor it into the repo or fetch it in a setup script — do not rely on an implicit download. |
| **`pyautogui`** | OS-level keystroke injection. Swappable — see *Action layer*. |
| **Custom logic** | Gesture classification. MediaPipe gives coordinates, not gestures. This layer is the entire quality of the product. |

Do **not** use `cvzone`. It wraps MediaPipe with convenience helpers but hides raw landmark data, pins outdated MediaPipe versions, and causes install conflicts on Apple Silicon.

### Running mode

Use **`VIDEO` mode** (`detect_for_video(image, timestamp_ms)`) with monotonically increasing timestamps.

Do **not** use `LIVE_STREAM` mode. It is asynchronous with a result callback and silently drops frames under load — the exact failure the velocity calculation is sensitive to. `VIDEO` mode is synchronous and processes every frame you hand it.

### Environment (verified 2026-08-22, darwin arm64 / cp312)

| Pin | Why |
|---|---|
| `mediapipe==0.10.35` | **`mediapipe` 1.0.x is unusable here.** `HandLandmarker` creation aborts (SIGABRT) in `DrishtiMetalHelper` with `Check failed: service_ Service is unavailable`, for every `running_mode` × `delegate` combination. 0.10.35 is the newest working version in the same wheel lineage. Retest before raising the pin. |
| `cv2` from `opencv-contrib-python` only | `opencv-python` and `opencv-contrib-python` both own the `cv2` namespace. Installing both, then removing either, leaves an importable but empty `cv2` husk. `mediapipe` already depends on `opencv-contrib-python`, which is a superset — never add `opencv-python` alongside it. |
| Python 3.12.13 via `uv` | System Python is 3.9.6; `mediapipe` needs 3.11/3.12. |

Reproduce with `uv venv --python 3.12 && uv pip install -e .`; `requirements.lock` holds the exact resolved set.

---

## Coordinate and timing conventions

Fix these once, at the top of the codebase, and never reason about them again. Each of these is a multi-hour bug if left implicit.

**Mirroring.** Run inference on the **unflipped** frame. Flip only for display. MediaPipe's handedness labels assume an unmirrored image, and flipping before inference silently inverts every direction in the system.

**Direction semantics.** In an unflipped frame, the presenter moving their hand to *their own right* produces **decreasing image x**. Define one constant and derive everything from it:
```python
SWIPE_INVERT = False   # False: hand sweeps to presenter's right -> NEXT slide
```
Default is presenter's-right → next, matching the arrow-key mental model. Some people instinctively swipe leftward to advance, phone-style. Flip the flag, do not rewrite the classifier.

**Aspect correction.** Normalized x and y are relative to the image, which is not square. Multiply normalized x by `frame_width / frame_height` before computing any magnitude or angle that mixes the two axes.

**Scale normalization.** Landmarks are normalized to the *image*, not the hand — so a fixed displacement threshold means a flick at 0.5m and a full arm sweep at 3m. Normalize every distance by per-frame **palm length**:
```python
palm = dist(landmark[0], landmark[9])   # wrist -> middle-finger MCP
```
Smooth it over a few frames. **All thresholds in this spec are expressed in palm-lengths, never in normalized image units.** This is what makes the tuning transfer between distances.

**Time, not frames.** Store `(timestamp_s, landmarks, confidence)` tuples in the ring buffer and compute derivatives as Δx/Δt in seconds. Never use per-frame deltas. Frame drops will happen regardless of intent; time-based derivatives degrade gracefully where frame-indexed ones produce garbage.

---

## Pipeline

```
capture thread: VideoCapture -> latest-frame slot (depth 1, overwrite)
main loop:
  latest frame (BGR)
  -> optional ROI crop + upscale        [if ROI_CROP_REQUIRED]
  -> cv2.cvtColor BGR->RGB
  -> HandLandmarker.detect_for_video    -> 21 landmarks + confidence
  -> confidence gate
  -> ring buffer append (timestamp, landmarks)   [gap check]
  -> pose classifier (open palm?)
  -> arming state machine (height gate + hysteresis)
  -> swipe classifier (velocity + direction consistency)
  -> cooldown gate
  -> ActionDispatcher -> keystroke
  -> status HUD update
```

**Measured 2026-08-23, and the earlier claim here was wrong.** This spec previously asserted that AVFoundation buffers 2–3 frames, that `CAP_PROP_BUFFERSIZE` is ignored, and that a capture thread was the largest latency saving after `pyautogui.PAUSE`. None of that holds on this machine.

| Load | Threaded | Direct |
|---|---|---|
| 5.1ms work (realistic) | 26.8 fps, 35% of a core | 27.5 fps, 32% of a core |
| 50ms work (overloaded) | 18.0 fps | 13.7 fps |
| 80ms work (overloaded) | 11.4 fps | 9.1 fps |

At the real workload the two are equivalent — the thread is worth nothing measurable. It pays off only when the consumer is slower than the camera, where it recovers 25–31% by taking `read()`'s blocking time off the critical path. The buffer is also shallow, not deep: direct-mode `read()` blocks 17–20ms rather than returning instantly from a backlog.

**Keep the capture thread anyway, but as insurance rather than optimization.** Under overload it degrades the right way — it drops stale frames instead of queueing them — which matters on unknown venue hardware or a thermally throttled machine. It is not load-bearing for latency, so do not spend effort tuning it.

**The main loop must wait for a new frame, not poll.** Polling `read()` at 5.1ms work spins at 179 fps and re-processes the same frame 85% of the time: 104% of a core to produce identical results. Blocking on a condition variable until the frame index advances gives 26.8 fps at 35% — same output, a third of the CPU.

---

## Detection at distance

**Settled by Spike B at 1.5m, full-frame, 1280×720:**

| Mode | Detection | `PALM px` median | Inference median |
|---|---|---|---|
| **Full-frame** | **91.2%** | 39.3 | 5.1ms |
| ROI 0.50 crop | 89.4% | 37.6 | 5.1ms |

Both figures include the walk to and from the camera, during which no hand was in view, so each understates steady-state tracking. Confidence while tracking was 0.98 median.

**ROI cropping is not required, and at this distance it is counterproductive.** It bought no detection and cost 1.8 points. The mechanism matters for anyone tempted to revisit it: cropping helps only when the hand is too *small* for the detector, and at 1.5m a 39px palm is already sufficient. What the crop does do is discard field of view — and a swipe is a large lateral hand movement, so the ROI box throws away precisely the volume the gesture needs. For this gesture set, cropping trades away the thing that matters.

Keep this section because the reasoning still applies if the geometry changes. MediaPipe's palm detector resizes its input to a small fixed square (roughly 192×192) regardless of capture resolution, so **raising capture resolution does not help a small hand** — a 60px hand in a 1280px frame is ~9px at the detector. Cropping and upscaling is the only lever, and it is worth reaching for only past roughly 2m, where the palm drops under ~25px. If it is ever re-enabled, the crop must be wide enough to contain a full swipe.

Note also that once the landmarker has acquired a hand it tracks from the previous frame's ROI and holds on far better than it acquires. **Cold-start acquisition is the hard case** — a system that looks fine after you have waved close to the camera may never acquire from a standing start.

This two-stage design also shows up in the timings: inference costs **11.5ms median while searching** for a hand versus **5.1ms while tracking** one, because the palm detector runs on every frame until something is found. The expensive path is therefore the idle path, which is the harmless direction — you are not mid-gesture while the system is searching. Budget against 5.1ms for gesture latency and 11.5ms for steady-state CPU load.

---

## Gesture design

Two detection strategies, combined by the classifier.

**Static pose** — which fingers are extended. Used as a *gate*, not as a trigger. Cheap and stable, but unnatural to hold, and firing actions directly on poses is prone to false positives during incidental pointing.

**Motion / swipe** — wrist displacement and velocity over the ring buffer. Natural to perform, noisier. This is what fires actions.

### Pose classification (rotation-invariant)

Do **not** compare fingertip y-positions against PIP joints. That breaks the moment the hand rolls or tilts, and it is simply wrong for the thumb, which extends laterally with a handedness-dependent sign.

Use radial distance from the wrist instead:
```python
extended(finger) = dist(tip, wrist) > dist(pip, wrist) * 1.15
```
This is rotation-invariant and handedness-agnostic.

**Define `OPEN_PALM` as all four non-thumb fingers extended.** Ignore the thumb entirely. Thumb state is the least reliable landmark group and it buys nothing here — sidestepping it removes a whole class of flakiness.

### Arming — implicit and continuous

**This is the single most important design decision in the project**, and v1 of this spec got the mechanism wrong: it used *open palm held 1s* as a toggle, while also using *open palm* as the swipe pose. Those are the same signal. Pausing with your hand up after a swipe — entirely natural — would silently disarm the controller, and there is no way to tell which state you are in.

Arming is therefore **implicit, continuous, and non-toggling**:

- Wrist above `ARM_Y_LINE` **and** `OPEN_PALM`, sustained for `ARM_ENTER_MS` → **ARMED**
- Wrist below the line, hand lost, or pose broken, sustained for `ARM_EXIT_MS` → **DISARMED**

No hold gesture. No timer. No toggle. The height gate *is* the arming mechanism. Dropping your arm is the panic stop, which is exactly the reflex you want under stress.

Asymmetric hysteresis: enter fast (~250ms) so the control feels responsive; exit slow (~1200ms) so a momentary tracking dropout mid-swipe does not disarm you.

### Status feedback (required, not optional)

A headless, disarmed-by-default controller with no indicator is unusable — the presenter cannot know whether they are armed. Ship a **status HUD**: a small, always-on-top, **non-activating** window showing armed/disarmed state, current detection confidence, and the last action fired.

It lives on the laptop screen, which faces the presenter, not the audience. Non-activating is a hard requirement: if the HUD can take focus, `pyautogui` will type into it instead of the presentation.

### Mapping (v1)

| Gesture | Action | Key |
|---|---|---|
| Open-palm swipe toward presenter's right | Next slide | `right` |
| Open-palm swipe toward presenter's left | Previous slide | `left` |

That is the whole v1 surface. **Blank-screen-on-fist is cut** — a relaxed hand at rest reads as a partial fist, and blanking the deck mid-sentence is the one false positive that is embarrassing rather than merely annoying. Revisit once the false-fire rate on next/prev is measured and acceptable.

Keep the mapping in a config dict so gestures can be remapped without touching detection code.

---

## State machine

Three states plus a cooldown. The v1 spec's `DETECTING` state is unnecessary — the ring buffer *is* the detection state.

```
DISARMED --[open palm above line, ARM_ENTER_MS]--> ARMED
ARMED    --[below line / lost / pose broken, ARM_EXIT_MS]--> DISARMED
ARMED    --[swipe confirmed]--> FIRED --[COOLDOWN_MS]--> ARMED
```

Clear the ring buffer on every transition into `ARMED` and into `COOLDOWN`. A buffer carried across a state change is a phantom-fire waiting to happen.

**Buffer gap invalidation.** If the interval between consecutive buffered samples exceeds `BUFFER_GAP_MS`, discard the buffer. Without this, a hand that occludes or leaves frame mid-swipe and returns produces an enormous spurious velocity and fires immediately. This is the most likely phantom-fire bug in the whole system.

---

## Thresholds (starting values — tune against fixtures)

These are informed starting points, not derived constants. Every one of them is expected to move during the tuning pass. Expose them in `config.py`.

```python
# Arming
ARM_Y_LINE            = 0.45    # normalized; wrist above this (upper 45% of frame)
ARM_ENTER_MS          = 250
ARM_EXIT_MS           = 1200

# Swipe (all distances in palm-lengths)
SWIPE_MIN_VELOCITY    = 8.0     # palm-lengths / second
SWIPE_MIN_DISPLACE    = 2.5     # palm-lengths
SWIPE_CONSISTENT_N    = 4       # consecutive same-direction frames to confirm
SWIPE_MAX_Y_DRIFT     = 1.5     # palm-lengths; rejects diagonal flails

# Gating
MIN_CONFIDENCE        = 0.6
MIN_PALM_PX           = 20      # reject impossible detections; Spike B logged a 6.3px "palm"
BUFFER_GAP_MS         = 150
RING_BUFFER_S         = 0.5

# Debounce
COOLDOWN_MS           = 900
```

**Fire on velocity-threshold crossing, not on completed displacement.** Confirm with `SWIPE_CONSISTENT_N` same-direction frames (~130ms at 30fps) rather than waiting for the full gesture to finish. This is the difference between "instant" and "laggy" and costs nothing in false-positive rate, because the pose and height gates are doing the real filtering.

---

## Latency budget

Target: **sub-200ms** from the swipe crossing the velocity threshold to the keystroke landing. Measured at that point, not at gesture completion — a classifier that waits for the motion to end can never feel instant.

| Stage | Budget @ 30fps |
|---|---|
| Camera-side latency (sensor → `read()` returns) | **unmeasured** |
| Pipeline-side staleness (measured) | 0.1ms med / 1.7ms max |
| Colour convert + ROI crop + resize | ~2ms |
| HandLandmarker inference — **tracking** (measured) | 5.1ms med / 7.5ms p95 |
| HandLandmarker inference — **acquiring** (measured) | 11.5ms med / 13.7ms p95 |
| Classifier + state machine | <1ms |
| Direction-consistency confirmation (4 frames) | ~130ms |
| Keystroke dispatch (`pynput`, measured) | <0.1ms |
| **Total (pipeline only)** | **~138ms + camera latency** |

Two lines dominate and both are configuration, not algorithms: the capture thread (worth 60–100ms) and `pyautogui.PAUSE` (worth 100ms). Get those right before optimizing anything else.

**One term remains unmeasured and it may dominate.** Everything above is measured from the moment `read()` returns. The time from photons hitting the sensor to `read()` returning — driver, USB, and camera pipeline — is invisible from inside the process and is typically 20–50ms on a webcam. Measuring it needs an external reference: point the camera at a millisecond clock, capture a frame, and compare the displayed time against the frame's arrival time. Worth doing before claiming the sub-200ms target is met.

**Instrument this.** Log the timestamp of the frame that triggered the fire and the timestamp after dispatch. A latency budget you cannot measure is a wish.

---

## Action layer

Abstract behind an `ActionDispatcher` class so the keystroke backend is swappable — this is also what makes Windows/Linux support additive later.

**Use `pynput`, not `pyautogui`.** Measured on this machine (50 iterations, macOS arm64, dispatching a bare modifier key):

| Backend | median | p95 |
|---|---|---|
| `pyautogui`, default `PAUSE = 0.1` | 127.8ms | 130.3ms |
| `pyautogui`, `PAUSE = 0` | 24.5ms | 25.3ms |
| raw Quartz `CGEventCreateKeyboardEvent` | 0.04ms | 0.14ms |
| **`pynput`** | **0.03ms** | **0.04ms** |

`pyautogui`'s remaining ~24ms after zeroing `PAUSE` is its own per-call overhead, and it is not tunable away. That is 5× the original budget for this stage and ~12% of the end-to-end target, spent on nothing. `pynput` and raw Quartz are both effectively free.

`pynput` is the default because it is cross-platform, so the `ActionDispatcher` abstraction actually pays off when Windows/Linux arrive. Raw Quartz is an equally fast macOS-only fallback if `pynput` ever becomes a problem — keycode 124 is Right, 123 is Left.

Note the cold-start penalty: the *first* dispatch in a fresh process is slower than steady state (~70ms for `pyautogui`, ~0.1ms for `pynput`). Send one throwaway keystroke to a harmless key at startup so the first real gesture isn't the one that pays it.

If `pyautogui` is used anyway for some reason, both of these are mandatory:
```python
pyautogui.PAUSE    = 0       # default 0.1s sleeps after EVERY call
pyautogui.FAILSAFE = False   # default aborts on cursor in a screen corner
```

---

## Tuning harness

**You cannot tune a gesture classifier by standing up and waving at your laptop.** Build the offline loop early — it is a couple of hours of work and it is the difference between a demo and a product.

**`record.py`** — writes JSONL, one object per frame: `{t, landmarks[21][3], confidence}`. No classification, just capture.

**Fixture set** — record once, reuse forever:
- ≥10 deliberate next-swipes
- ≥10 deliberate previous-swipes
- ≥10 minutes of you talking naturally with your hands, gesturing freely, *with no intent to control anything*

That last one is the important corpus and the one people skip. It is the entire false-positive test.

**`evaluate.py`** — replays fixtures through the classifier and reports recall, precision, and false fires per minute. Threshold changes become a number, not a vibe.

---

## Definition of done (v1)

Ship criteria, measured against the fixture set and a live run:

- **Recall** ≥ 95% on deliberate swipes
- **False fires** < 1 per 10 minutes of natural speech with hands
- **Latency** < 200ms, p95, instrumented
- **Cold start**: acquires tracking from a standing start at `MEASURED_DISTANCE_M` within 2 seconds
- Survives a full 20-minute talk without a spurious slide change

The false-fire number is the one that matters. A controller that misses a swipe is a minor annoyance you recover from by swiping again; a controller that advances the deck while you are making a point is one you switch off and never trust again.

---

## The hard problems

**False positives during natural speech.** A presenter gestures constantly while talking. Mitigations, layered — each one alone is insufficient:
- *Arming* (implicit height gate + pose, above) — the primary defence
- *Pose gate*: `OPEN_PALM` required throughout the swipe, not just at its start
- *Velocity floor*: conversational motion is slower than a deliberate swipe
- *Direction consistency*: `SWIPE_CONSISTENT_N` same-sign frames rejects oscillation
- *Y-drift rejection*: `SWIPE_MAX_Y_DRIFT` rejects diagonal flails
- *Confidence floor*: discard frames below `MIN_CONFIDENCE`

**Debouncing.** One physical swipe spans 10–20 frames. Without a cooldown, one swipe fires 15 slide advances. `COOLDOWN_MS` plus buffer-clearing on state transition handles this.

**Focus.** `pyautogui` sends keys to whatever window is focused. If the HUD has focus, the script keys itself. The HUD must be non-activating and always-on-top; headless mode must be available as a flag regardless.

**Lighting.** Webcam hand tracking degrades badly in dim or heavily backlit rooms — exactly the conditions of a presentation venue. Test under `VENUE_LIGHTING` in Spike B, not at your desk at noon. Log detection confidence continuously so degradation is diagnosable after the fact rather than mysterious during.

**Tracking loss.** Handled by `BUFFER_GAP_MS` invalidation and asymmetric arming hysteresis. Assume it happens constantly; design so it costs a missed swipe, never a phantom one.

---

## macOS environment requirements

Both permissions are granted to the *host application* running the script (Terminal, iTerm, or Cursor), **not** to Python itself.

- **System Settings → Privacy & Security → Accessibility** — required for `pyautogui`. Without it the script runs silently, reports no error, and no keystrokes land. This is the most common source of "it doesn't work." Spike A exists to catch it on day one.
- **System Settings → Privacy & Security → Camera** — required for OpenCV capture.

If you switch from running in Terminal to running in Cursor, you must grant both again for the new host app, and fully restart it.

---

## Build order

1. **Spike A** — keystroke injection into Keynote. *Gate.*
2. **Spike B** — landmark stability at real distance and lighting. *Gate.* Sets `ROI_CROP_REQUIRED`.
3. Capture thread with latest-frame slot + display window + clean exit. Verify frame rate and measure end-to-end frame staleness.
4. HandLandmarker in `VIDEO` mode, landmarks drawn on preview, ROI crop applied if required. Log confidence.
5. Pose classifier (`fingers_up()` → 5-element boolean, radial method) + palm-length normalization. Display both on preview.
6. Timestamped ring buffer + velocity in palm-lengths/sec, with gap invalidation. Display live velocity.
7. **`record.py` + fixture capture.** Record the full fixture set now, before there is anything to tune. *Do not skip this step.*
8. Swipe classifier against thresholds from `config.py`.
9. **`evaluate.py`** + first tuning pass against fixtures.
10. Arming state machine + cooldown.
11. `ActionDispatcher` + `pyautogui` backend, with latency instrumentation.
12. Status HUD (non-activating, always-on-top) + headless flag.
13. Second tuning pass against *Definition of done*. Re-record fixtures if the geometry changed.

Keep everything in one file through step 6, then split into `capture.py`, `gestures.py`, `actions.py`, `config.py`, `hud.py`.

Steps 7 and 9 are the ones under schedule pressure that will feel skippable. They are not — without them, steps 8 and 13 are guesswork, and step 13 is where the project either becomes trustworthy or does not.

---

## Non-goals for v1

- Building a slideshow viewer. The tool drives existing presentation software.
- Slide annotation, drawing, laser pointer overlay.
- Multi-hand gestures.
- Blank/black screen toggle (cut from v1 — revisit after false-fire rate is measured).
- Windows/Linux support (but keep the action layer abstracted for it).
- Packaging/distribution.

---

## Changes from v1 of this spec

| Change | Reason |
|---|---|
| Added Step 0 spikes as gates | Both project-killing risks were buried at steps 2 and 7 |
| Added distance measurement as a prerequisite | Every threshold and the camera choice depend on an unmeasured number |
| Arming changed from a 1s-hold toggle to implicit height-gate | Toggle used the same signal as the swipe pose; holding still after a swipe would disarm you |
| Added status HUD as a requirement | Headless + disarmed-by-default + no indicator is unusable |
| All thresholds expressed in palm-lengths | Image-normalized thresholds only work at the distance they were tuned at |
| Buffer stores timestamps; velocity is Δx/Δt | "Never skip frames" is unenforceable; time-based derivatives make drops harmless |
| Added buffer gap invalidation | Tracking loss mid-swipe otherwise produces a phantom fire |
| Pose classification changed to radial distance from wrist | y-vs-PIP breaks on hand roll and is wrong for the thumb |
| `OPEN_PALM` ignores the thumb | Least reliable landmark group, no benefit |
| Specified `VIDEO` mode explicitly | `LIVE_STREAM` silently drops frames |
| Capture thread demoted to insurance; main loop waits rather than polls | Measured: thread is worth ~0 at realistic load (26.8 vs 27.5 fps) and 25–31% only under overload; the AVFoundation deep-buffering claim was false. Polling burns 104% of a core for identical output |
| Backend switched to `pynput`; `pyautogui.PAUSE = 0` | Measured: `pyautogui` costs ~24ms/keystroke even with `PAUSE = 0`, vs 0.03ms for `pynput` |
| Documented mirroring and direction conventions | Classic multi-hour bug |
| Fire on velocity crossing, not completed displacement | Displacement-window confirmation cannot feel instant |
| Removed the `DETECTING` state | The ring buffer already is that state |
| Cut fist → blank screen | Relaxed hand reads as partial fist; highest-cost false positive |
| Added tuning harness and fixture set (steps 7, 9) | A gesture classifier cannot be tuned interactively |
| Added *Definition of done* with measurable criteria | v1 had no acceptance criteria at all |
| `ROI_CROP_REQUIRED = False`; added `MIN_PALM_PX` | Spike B: ROI cropped 1.8 points *off* detection by clipping swipe volume; a 6.3px palm was detected, which is impossible |
| Noted resolution bumps don't help at distance | Palm detector downsamples to a fixed small square; ROI cropping is the actual fix |
