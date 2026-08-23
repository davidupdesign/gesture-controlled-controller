"""Gesture-controlled presentation controller.

Build steps 3-5: capture, hand tracking, and pose classification. Per the spec, steps 3-6 live in this one file; it splits
into capture/gestures/actions/config/hud once the ring buffer lands.

    python controller.py                      # preview window
    python controller.py --headless --seconds 15
    python controller.py --headless --seconds 15 --no-thread   # A/B the thread

Why the capture thread exists
-----------------------------
OpenCV's AVFoundation backend buffers frames, and CAP_PROP_BUFFERSIZE is
frequently ignored on macOS. A main loop calling read() directly therefore
consumes frames that were captured tens of milliseconds ago - latency the rest
of the pipeline can never win back. FrameGrabber reads continuously on a worker
thread and keeps only the newest frame, so the main loop always gets the
freshest one and slow consumers drop old frames instead of queueing them.

The spec budgets 0-33ms for this stage. `--headless --seconds N` measures it.
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import threading
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = "models/hand_landmarker.task"

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
# Matches Spike B's measured HandLandmarker cost, so the loop is loaded the way
# the real pipeline will load it. Staleness measured against an idle loop lies.
DEFAULT_WORK_MS = 5.1

# Gates. A detection below either of these is discarded before it can reach the
# ring buffer. MIN_PALM_PX exists because Spike B logged a 6.3px "palm", which
# is physically impossible - confidence alone does not filter junk.
MIN_CONFIDENCE = 0.6
MIN_PALM_PX = 20.0

# Standard MediaPipe 21-landmark topology.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm
]
WRIST, MIDDLE_MCP = 0, 9

# (tip, reference joint) per finger. Extension is judged by radial distance from
# the wrist, not by y-position against the PIP: y-comparison breaks the moment
# the hand rolls or tilts, and is simply wrong for the thumb, which extends
# laterally with a handedness-dependent sign.
FINGER_JOINTS = {
    "thumb":  (4, 2),      # tip vs MCP - the thumb has no comparable PIP
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}
FINGER_NAMES = tuple(FINGER_JOINTS)
EXTENDED_RATIO = 1.15      # tip must be this much further from the wrist than the joint


@dataclass(frozen=True)
class Frame:
    """A captured frame plus the instant it was captured, for age measurement."""
    image: "cv2.typing.MatLike"
    captured_at: float          # time.perf_counter() when read() returned
    index: int


class FrameGrabber:
    """Single-slot latest-frame reader. Start it, read() the newest frame, stop it.

    Only ever holds one frame: a slow consumer drops stale frames rather than
    falling behind a queue. read() returns None until the first frame arrives,
    and returns the same Frame again if the consumer outruns the camera - hence
    Frame.index, which lets callers tell a repeat from a fresh capture.
    """

    def __init__(self, camera: int = CAMERA_INDEX, width: int = FRAME_W, height: int = FRAME_H):
        self._camera, self._width, self._height = camera, width, height
        self._cap: cv2.VideoCapture | None = None
        self._latest: Frame | None = None
        self._new_frame = threading.Condition()   # signals a genuinely new frame
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._count = 0
        self.read_ms: list[float] = []      # how long each read() blocked

    def start(self) -> "FrameGrabber":
        self._cap = cv2.VideoCapture(self._camera)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"cannot open camera {self._camera}. Grant Camera permission to "
                f"the host app, or try a different index."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # Requested for completeness; AVFoundation commonly ignores it, which is
        # the whole reason this class exists.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            ok, image = self._cap.read()
            t1 = time.perf_counter()
            if not ok:
                continue
            self.read_ms.append((t1 - t0) * 1000)
            with self._new_frame:
                self._count += 1
                self._latest = Frame(image, t1, self._count)
                self._new_frame.notify_all()

    def read(self) -> Frame | None:
        """Newest frame, or None before the first arrives. Never blocks."""
        with self._new_frame:
            return self._latest

    def read_new(self, after: int, timeout: float = 1.0) -> Frame | None:
        """Block until a frame newer than `after` arrives, or timeout.

        Without this the main loop spins: at 5ms of work against a 33ms frame
        interval it re-processes the same frame ~85% of the time, burning CPU
        to produce identical results. Waiting on the condition costs nothing
        and drops the loop to the camera's actual rate.
        """
        deadline = time.perf_counter() + timeout
        with self._new_frame:
            while self._latest is None or self._latest.index <= after:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._new_frame.wait(remaining)
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "FrameGrabber":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class DirectGrabber:
    """Unthreaded baseline: read() straight from the camera on the caller's thread.

    Exists only to quantify what the capture thread is worth. Same interface.
    """

    def __init__(self, camera: int = CAMERA_INDEX, width: int = FRAME_W, height: int = FRAME_H):
        self._camera, self._width, self._height = camera, width, height
        self._cap: cv2.VideoCapture | None = None
        self._count = 0
        self.read_ms: list[float] = []

    def start(self) -> "DirectGrabber":
        self._cap = cv2.VideoCapture(self._camera)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {self._camera}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self

    def read_new(self, after: int, timeout: float = 1.0) -> Frame | None:
        # Direct read() already blocks until the camera yields a frame.
        return self.read()

    def read(self) -> Frame | None:
        t0 = time.perf_counter()
        ok, image = self._cap.read()
        t1 = time.perf_counter()
        if not ok:
            return None
        self.read_ms.append((t1 - t0) * 1000)
        self._count += 1
        return Frame(image, t1, self._count)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "DirectGrabber":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


@dataclass(frozen=True)
class HandObservation:
    """One frame of accepted hand data.

    `landmarks` are MediaPipe's normalized 0-1 coordinates in the UNFLIPPED
    image, which is the only frame of reference the rest of the pipeline uses.
    Mirroring happens at draw time and nowhere else.

    `palm_px` is the wrist-to-middle-MCP distance in source pixels. It is the
    scale reference every later threshold divides by, so that a swipe means the
    same physical gesture regardless of distance from the camera.
    """
    landmarks: list[tuple[float, float, float]]
    palm_px: float
    confidence: float
    captured_at: float
    frame_index: int
    aspect: float = 1.0      # width/height; normalized x is squashed without it

    def dist(self, a: int, b: int) -> float:
        """Aspect-corrected distance between two landmarks, in normalized-y units.

        MediaPipe normalizes x and y independently against a non-square image,
        so raw normalized x and y are not the same unit. Any magnitude mixing
        the two axes must correct for that first or it is silently wrong.
        """
        ax, ay, _ = self.landmarks[a]
        bx, by, _ = self.landmarks[b]
        return math.hypot((ax - bx) * self.aspect, ay - by)

    @property
    def palm(self) -> float:
        """Wrist-to-middle-MCP length. The scale every threshold divides by.

        Expressing distances in palm-lengths is what makes a threshold mean the
        same physical gesture at 0.4m and at 1.5m. Spike B measured the palm at
        163.8px close up and 39.3px at 1.5m - a 4x swing that any fixed
        image-normalized threshold would get wrong at one end or the other.
        """
        return self.dist(WRIST, MIDDLE_MCP)


class HandTracker:
    """MediaPipe HandLandmarker in VIDEO mode, with the spec's gates applied.

    VIDEO mode (not LIVE_STREAM) because it is synchronous and processes every
    frame handed to it; LIVE_STREAM is callback-driven and silently drops frames
    under load, which is exactly what corrupts velocity estimation.
    """

    def __init__(self, model_path: str = MODEL_PATH,
                 min_confidence: float = MIN_CONFIDENCE,
                 min_palm_px: float = MIN_PALM_PX):
        self._min_confidence = min_confidence
        self._min_palm_px = min_palm_px
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._last_ts_ms = -1
        self.infer_ms: list[float] = []
        self.rejected = {"confidence": 0, "palm_px": 0}

    def process(self, frame: Frame) -> HandObservation | None:
        """Run inference on one frame. Returns None if no hand passed the gates."""
        # Inference runs on the UNFLIPPED image - MediaPipe's handedness labels
        # assume an unmirrored frame, and flipping first inverts every direction
        # downstream.
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)

        # detect_for_video requires strictly increasing timestamps or it raises.
        ts_ms = max(int(frame.captured_at * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        t0 = time.perf_counter()
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )
        self.infer_ms.append((time.perf_counter() - t0) * 1000)

        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]
        # NOTE: MediaPipe Tasks exposes the handedness classification score, not
        # a per-frame detection confidence. It is a proxy, and a good one in
        # practice (0.98 median in Spike B), but do not read it as "how sure the
        # model is that this is a hand".
        confidence = result.handedness[0][0].score if result.handedness else 0.0
        if confidence < self._min_confidence:
            self.rejected["confidence"] += 1
            return None

        h, w = frame.image.shape[:2]
        palm_px = float(np.hypot(
            (lm[MIDDLE_MCP].x - lm[WRIST].x) * w,
            (lm[MIDDLE_MCP].y - lm[WRIST].y) * h,
        ))
        if palm_px < self._min_palm_px:
            self.rejected["palm_px"] += 1
            return None

        return HandObservation(
            landmarks=[(p.x, p.y, p.z) for p in lm],
            palm_px=palm_px,
            confidence=confidence,
            captured_at=frame.captured_at,
            frame_index=frame.index,
            aspect=w / h,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def fingers_up(obs: HandObservation) -> tuple[bool, ...]:
    """Which fingers are extended, as (thumb, index, middle, ring, pinky).

    Rotation-invariant and handedness-agnostic: a finger is extended when its
    tip is measurably further from the wrist than its reference joint, which
    holds however the hand is oriented.
    """
    return tuple(
        obs.dist(tip, WRIST) > obs.dist(joint, WRIST) * EXTENDED_RATIO
        for tip, joint in FINGER_JOINTS.values()
    )


def is_open_palm(fingers: tuple[bool, ...]) -> bool:
    """All four non-thumb fingers extended.

    The thumb is deliberately ignored. It is the least reliable landmark group,
    and including it buys nothing here - it only adds a way for a genuine open
    palm to be rejected.
    """
    return all(fingers[1:])


def format_fingers(fingers: tuple[bool, ...]) -> str:
    """Compact readout: 'TIMRP' all up, '.IMRP' thumb down."""
    return "".join(n[0].upper() if up else "." for n, up in zip(FINGER_NAMES, fingers))


def draw_landmarks(view, obs: HandObservation, width: int, height: int,
                   open_palm: bool = False) -> None:
    """Draw the hand skeleton onto the MIRRORED preview.

    `view` is already flipped for display, so x is mirrored here - the sole
    place in the codebase where that happens.
    """
    colour = (0, 255, 120) if open_palm else (140, 140, 140)
    pts = [(width - int(x * width), int(y * height)) for x, y, _ in obs.landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(view, pts[a], pts[b], colour, 2)
    for pt in pts:
        cv2.circle(view, pt, 3, (0, 200, 255), -1)


def draw_hud(image, lines) -> None:
    """Outlined text, legible against any background."""
    for i, line in enumerate(lines):
        pos = (12, 30 + i * 26)
        cv2.putText(image, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    ap.add_argument("--headless", action="store_true", help="no preview window")
    ap.add_argument("--seconds", type=float, default=0, help="auto-exit after N seconds")
    ap.add_argument("--no-thread", action="store_true", help="unthreaded baseline")
    ap.add_argument("--spin", action="store_true",
                    help="poll read() instead of waiting for a new frame")
    ap.add_argument("--no-track", action="store_true",
                    help="capture only; substitute simulated work for inference")
    ap.add_argument("--work-ms", type=float, default=DEFAULT_WORK_MS,
                    help="simulated per-frame cost when --no-track is set")
    args = ap.parse_args()

    Grabber = DirectGrabber if args.no_thread else FrameGrabber
    mode = "direct (unthreaded)" if args.no_thread else "threaded"

    staleness: list[float] = []
    palms: list[float] = []
    confs: list[float] = []
    open_palm_frames = 0
    gaps: list[float] = []            # continuous tracking dropouts, seconds
    frames = repeats = hits = 0
    last_index = -1
    miss_run_start: float | None = None
    t_start = time.perf_counter()
    cpu_start = time.process_time()

    print(f"capture: {mode}  tracking: {'off' if args.no_track else 'on'}  "
          f"{'headless' if args.headless else 'preview (q to quit)'}")

    tracker = None if args.no_track else HandTracker()
    try:
        with Grabber(args.camera) as grabber:
            while True:
                frame = grabber.read() if args.spin else grabber.read_new(last_index)
                if frame is None:
                    if args.seconds and time.perf_counter() - t_start > args.seconds:
                        break
                    continue

                now = time.perf_counter()
                if frame.index == last_index:
                    repeats += 1
                else:
                    staleness.append((now - frame.captured_at) * 1000)
                    last_index = frame.index
                frames += 1

                obs = None
                fingers: tuple[bool, ...] = ()
                open_palm = False
                if tracker is not None:
                    obs = tracker.process(frame)
                    if obs is not None:
                        hits += 1
                        palms.append(obs.palm_px)
                        confs.append(obs.confidence)
                        fingers = fingers_up(obs)
                        open_palm = is_open_palm(fingers)
                        open_palm_frames += open_palm
                        if miss_run_start is not None:
                            gaps.append(now - miss_run_start)
                            miss_run_start = None
                    elif miss_run_start is None:
                        miss_run_start = now
                elif args.work_ms:
                    spin_until = time.perf_counter() + args.work_ms / 1000
                    while time.perf_counter() < spin_until:
                        pass

                if not args.headless:
                    view = cv2.flip(frame.image, 1)
                    h, w = view.shape[:2]
                    if obs is not None:
                        draw_landmarks(view, obs, w, h, open_palm)
                    elapsed = now - t_start
                    draw_hud(view, [
                        f"loop: {frames / max(elapsed, 1e-6):5.1f} fps   "
                        f"track: {hits / max(frames, 1) * 100:5.1f}%",
                        f"fingers: {format_fingers(fingers)}" if obs else "fingers: -----",
                        f"OPEN PALM" if open_palm else "-",
                        f"palm: {obs.palm_px:5.1f} px  conf: {obs.confidence:.2f}"
                        if obs else "palm:    -- px",
                    ])
                    cv2.imshow("controller - step 5 pose", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.seconds and now - t_start > args.seconds:
                    break
    finally:
        if tracker is not None:
            tracker.close()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    cpu = time.process_time() - cpu_start
    stale = sorted(staleness)
    print("\n" + "=" * 62)
    print(f"mode           {mode}, tracking {'off' if args.no_track else 'on'}")
    print(f"loop           {frames / max(elapsed, 1e-6):.1f} fps over {elapsed:.1f}s "
          f"({frames} frames, {repeats} repeats)")
    print(f"cpu            {cpu:.1f}s of {elapsed:.1f}s wall = "
          f"{cpu / max(elapsed, 1e-6) * 100:.0f}% of one core")
    if stale:
        print(f"staleness      median={st.median(stale):.1f}ms  "
              f"p95={percentile(stale, .95):.1f}ms  max={stale[-1]:.1f}ms")
    if tracker is not None:
        infer = sorted(tracker.infer_ms)
        print(f"tracking       {hits / max(frames, 1) * 100:.1f}% of {frames} frames")
        if infer:
            print(f"inference      median={st.median(infer):.1f}ms  "
                  f"p95={percentile(infer, .95):.1f}ms")
        if palms:
            print(f"palm px        median={st.median(palms):.1f}  "
                  f"min={min(palms):.1f}  max={max(palms):.1f}")
        if confs:
            print(f"confidence     median={st.median(confs):.2f}  min={min(confs):.2f}")
        print(f"open palm      {open_palm_frames} frames "
              f"({open_palm_frames / max(hits, 1) * 100:.1f}% of tracked)")
        print(f"rejected       confidence={tracker.rejected['confidence']}  "
              f"palm_px={tracker.rejected['palm_px']}")
        if gaps:
            ms = sorted(g * 1000 for g in gaps)
            print(f"dropouts       count={len(ms)}  longest={ms[-1]:.0f}ms  "
                  f"median={st.median(ms):.0f}ms  over_150ms={sum(1 for g in ms if g > 150)}")
            print("               -> informs BUFFER_GAP_MS and ARM_EXIT_MS")
        else:
            print("dropouts       none")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
