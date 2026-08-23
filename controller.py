"""Gesture-controlled presentation controller.

Build step 3: capture. Per the spec, steps 3-6 live in this one file; it splits
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
import statistics as st
import threading
import time
from dataclasses import dataclass

import cv2

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
# Matches Spike B's measured HandLandmarker cost, so the loop is loaded the way
# the real pipeline will load it. Staleness measured against an idle loop lies.
DEFAULT_WORK_MS = 5.1


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
    ap.add_argument("--work-ms", type=float, default=DEFAULT_WORK_MS,
                    help="simulated per-frame inference cost")
    ap.add_argument("--spin", action="store_true",
                    help="poll read() instead of waiting for a new frame")
    args = ap.parse_args()

    Grabber = DirectGrabber if args.no_thread else FrameGrabber
    mode = "direct (unthreaded)" if args.no_thread else "threaded"

    staleness, loop_ms = [], []
    frames = repeats = 0
    last_index = -1
    t_start = time.perf_counter()
    t_prev = t_start
    cpu_start = time.process_time()

    print(f"capture: {mode}  work={args.work_ms:.1f}ms  "
          f"{'headless' if args.headless else 'preview (q to quit)'}")

    with Grabber(args.camera) as grabber:
        while True:
            frame = grabber.read() if args.spin else grabber.read_new(last_index)
            if frame is None:
                if args.seconds and time.perf_counter() - t_start > args.seconds:
                    break
                continue

            now = time.perf_counter()
            if frame.index == last_index:
                repeats += 1           # consumer outran the camera
            else:
                staleness.append((now - frame.captured_at) * 1000)
                last_index = frame.index
            frames += 1

            # Stand in for HandLandmarker so the loop is realistically loaded.
            if args.work_ms:
                spin_until = time.perf_counter() + args.work_ms / 1000
                while time.perf_counter() < spin_until:
                    pass

            loop_ms.append((now - t_prev) * 1000)
            t_prev = now

            if not args.headless:
                view = cv2.flip(frame.image, 1)
                elapsed = now - t_start
                draw_hud(view, [
                    f"mode: {mode}",
                    f"loop: {frames / max(elapsed, 1e-6):5.1f} fps",
                    f"stale: {staleness[-1]:5.1f} ms" if staleness else "stale:    -- ms",
                    f"repeats: {repeats}",
                ])
                cv2.imshow("controller - step 3 capture", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.seconds and now - t_start > args.seconds:
                break

    if not args.headless:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    cpu = time.process_time() - cpu_start
    reads = sorted(grabber.read_ms)
    stale = sorted(staleness)
    print("\n" + "=" * 60)
    print(f"mode           {mode}")
    print(f"loop           {frames / max(elapsed, 1e-6):.1f} fps over {elapsed:.1f}s "
          f"({frames} frames, {repeats} repeats)")
    print(f"cpu            {cpu:.1f}s of {elapsed:.1f}s wall = {cpu/max(elapsed,1e-6)*100:.0f}% of one core")
    if stale:
        print(f"staleness      median={st.median(stale):.1f}ms  p95={percentile(stale, .95):.1f}ms  "
              f"max={stale[-1]:.1f}ms")
    if reads:
        print(f"read() blocked median={st.median(reads):.1f}ms  p95={percentile(reads, .95):.1f}ms")
    print("=" * 60)
    print("staleness is the spec's 0-33ms capture line: how old a frame already")
    print("is when the pipeline receives it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
