"""Hand tracking, pose classification, and motion analysis."""

from __future__ import annotations

import collections
import math
import statistics as st
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from capture import Frame
from config import (
    BUFFER_GAP_MS,
    SWIPE_CONSISTENT_N,
    SWIPE_INVERT,
    SWIPE_MAX_Y_DRIFT,
    SWIPE_MIN_DISPLACE,
    SWIPE_MIN_VELOCITY,
    EXTENDED_RATIO,
    MIN_CONFIDENCE,
    MIN_PALM_PX,
    MODEL_PATH,
    PALM_SMOOTH_N,
    RING_BUFFER_S,
    VELOCITY_LOOKBACK_S,
)

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
# laterally with a handedness-dependent sign. Measured across a 360-degree
# sweep: radial 100% correct, naive y-comparison 32%.
FINGER_JOINTS = {
    "thumb":  (4, 2),      # tip vs MCP - the thumb has no comparable PIP
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}
FINGER_NAMES = tuple(FINGER_JOINTS)


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


@dataclass(frozen=True)
class Sample:
    """One buffered instant of wrist motion, in aspect-corrected normalized units."""
    t: float
    x: float
    y: float
    palm: float


class MotionBuffer:
    """Time-indexed ring buffer of wrist positions, yielding palm-normalized velocity.

    Two decisions here are load-bearing:

    Derivatives are taken against wall-clock time, never against frame counts.
    Frames will be dropped no matter how carefully the loop is written, and a
    frame-indexed derivative silently reports a fast gesture when it was really
    a slow one across a gap.

    A gap longer than `gap_ms` clears the buffer outright. Without that, a hand
    that occludes mid-swipe and returns produces an enormous spurious velocity
    across the discontinuity and fires immediately. This is the single most
    likely phantom-fire bug in the system.
    """

    def __init__(self, window_s: float = RING_BUFFER_S, gap_ms: float = BUFFER_GAP_MS,
                 palm_smooth_n: int = PALM_SMOOTH_N):
        self._window_s = window_s
        self._gap_s = gap_ms / 1000.0
        self._samples: collections.deque[Sample] = collections.deque()
        self._palms: collections.deque[float] = collections.deque(maxlen=palm_smooth_n)
        self.gap_invalidations = 0

    def __len__(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()
        self._palms.clear()

    def append(self, obs: HandObservation) -> bool:
        """Add an observation. Returns False if it invalidated the buffer.

        The palm scale is a running median rather than the instantaneous value:
        a single bad frame's palm estimate would otherwise rescale every
        velocity derived from it.
        """
        gapped = bool(self._samples) and (obs.captured_at - self._samples[-1].t) > self._gap_s
        if gapped:
            self.clear()
            self.gap_invalidations += 1

        self._palms.append(obs.palm)
        palm = st.median(self._palms)

        wx, wy, _ = obs.landmarks[WRIST]
        self._samples.append(Sample(obs.captured_at, wx * obs.aspect, wy, palm))

        cutoff = obs.captured_at - self._window_s
        while self._samples and self._samples[0].t < cutoff:
            self._samples.popleft()
        return not gapped

    @property
    def latest(self) -> Sample | None:
        return self._samples[-1] if self._samples else None

    def _span(self, lookback_s: float) -> tuple[Sample, Sample] | None:
        """Oldest sample within `lookback_s` of the newest, paired with the newest."""
        if len(self._samples) < 2:
            return None
        newest = self._samples[-1]
        oldest = self._samples[0]
        for sample in self._samples:
            if newest.t - sample.t <= lookback_s:
                oldest = sample
                break
        if oldest is newest or newest.t <= oldest.t:
            return None
        return oldest, newest

    def velocity(self, lookback_s: float = VELOCITY_LOOKBACK_S) -> tuple[float, float] | None:
        """(vx, vy) in palm-lengths per second, or None if there is not enough data."""
        span = self._span(lookback_s)
        if span is None:
            return None
        oldest, newest = span
        dt = newest.t - oldest.t
        palm = newest.palm
        if palm <= 0:
            return None
        return ((newest.x - oldest.x) / palm / dt, (newest.y - oldest.y) / palm / dt)

    def displacement(self) -> tuple[float, float] | None:
        """(dx, dy) in palm-lengths across the whole buffer."""
        if len(self._samples) < 2:
            return None
        oldest, newest = self._samples[0], self._samples[-1]
        palm = newest.palm
        if palm <= 0:
            return None
        return ((newest.x - oldest.x) / palm, (newest.y - oldest.y) / palm)


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


@dataclass(frozen=True)
class Swipe:
    """A confirmed swipe, with the evidence that produced it."""
    action: str              # "next" | "prev"
    direction: int           # +1 increasing image x, -1 decreasing
    peak_velocity: float     # palm-lengths/sec
    displacement: float      # palm-lengths travelled during the run
    frames: int
    t: float


class SwipeClassifier:
    """Confirms a swipe from sustained, consistent motion.

    Fires on a run of consecutive same-direction frames above a velocity floor,
    rather than on a completed displacement. Waiting for the gesture to finish
    cannot feel instant; confirming over ~4 frames costs ~130ms at 30fps.

    The consistency requirement is load-bearing, not a refinement. Live data
    contains single-frame velocity spikes of 109 and 220 palm-lengths/sec -
    landmark glitches, since the run producing the 220 had no buffer gap at all.
    A bare velocity threshold fires on every one of those. Nothing here should
    ever be reduced to "is the hand moving fast".

    It is also not sufficient on its own. Ambient conversational motion was
    measured at p99 = 4.5 palm/s while deliberate swipes sat around p95 = 3.9 -
    the populations overlap, so velocity cannot establish intent. The caller
    must gate on pose and arming; this class assumes that has already happened.
    """

    def __init__(self, min_velocity: float = SWIPE_MIN_VELOCITY,
                 consistent_n: int = SWIPE_CONSISTENT_N,
                 max_y_drift: float = SWIPE_MAX_Y_DRIFT,
                 min_displacement: float = SWIPE_MIN_DISPLACE,
                 invert: bool = SWIPE_INVERT):
        self.min_velocity = min_velocity
        self.consistent_n = consistent_n
        self.max_y_drift = max_y_drift
        self.min_displacement = min_displacement
        self.invert = invert
        self.reset()

    def reset(self) -> None:
        self._direction: int | None = None
        self._frames = 0
        self._peak = 0.0
        self._start: Sample | None = None
        # Set once a run has fired. The motion is still qualifying - a physical
        # swipe spans 10-20 frames - so the run must be held down until it stops
        # qualifying, or it re-confirms every consistent_n frames and fires
        # repeatedly across a single gesture.
        self._spent = False

    @property
    def run_frames(self) -> int:
        """Consecutive qualifying frames so far - useful for a progress readout."""
        return self._frames

    def _action(self, direction: int) -> str:
        # In an unflipped frame, the presenter moving their hand to their own
        # right produces DECREASING image x.
        presenter_right = direction < 0
        return "next" if presenter_right != self.invert else "prev"

    def update(self, buffer: MotionBuffer, open_palm: bool) -> Swipe | None:
        """Feed one frame. Returns a Swipe on the frame the run is confirmed.

        Exactly one Swipe per physical gesture. Firing marks the run spent
        rather than clearing it: the hand is still travelling, so resetting
        would start a fresh run that re-confirms a few frames later and fires
        again - a 20-frame sweep produced three fires before this was caught.
        The run is released only when the motion stops qualifying, through the
        pose breaking, the velocity dropping, or the direction reversing.
        """
        if not open_palm:
            self.reset()
            return None

        velocity = buffer.velocity()
        latest = buffer.latest
        if velocity is None or latest is None or latest.palm <= 0:
            self.reset()
            return None

        vx, _ = velocity
        if abs(vx) < self.min_velocity:
            self.reset()
            return None

        direction = 1 if vx > 0 else -1
        if direction != self._direction or self._start is None:
            self._direction = direction
            self._start = latest
            self._frames = 0
            self._peak = 0.0
            self._spent = False

        self._frames += 1
        self._peak = max(self._peak, abs(vx))
        if self._spent:
            return None

        dx = (latest.x - self._start.x) / latest.palm
        dy = (latest.y - self._start.y) / latest.palm

        # Reject diagonal flails: a swipe is lateral, and vertical travel this
        # large means the arm is doing something else.
        if abs(dy) > self.max_y_drift:
            self.reset()
            return None

        if self._frames >= self.consistent_n and abs(dx) >= self.min_displacement:
            swipe = Swipe(
                action=self._action(direction),
                direction=direction,
                peak_velocity=self._peak,
                displacement=abs(dx),
                frames=self._frames,
                t=latest.t,
            )
            self._spent = True
            return swipe
        return None
