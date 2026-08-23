"""Camera capture.

The threaded grabber is insurance, not optimization. Measured at the real
workload it is worth nothing (26.8 fps at 35% of a core versus 27.5 at 32% for
a direct read); it earns 25-31% only when the consumer is slower than the
camera. What it does buy is the right failure mode under load - it drops stale
frames instead of queueing them - which matters on unknown venue hardware.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2

from config import CAMERA_INDEX, FRAME_H, FRAME_W


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
