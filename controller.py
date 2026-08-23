"""Gesture-controlled presentation controller.

Build steps 3-8: capture, hand tracking, pose, motion buffer, swipe classifier.
Still missing the arming state machine (step 10) and the action layer (step 11),
so nothing here presses a key yet - detections are reported to the preview only.

    python controller.py                       # preview
    python controller.py --headless --seconds 20
    python controller.py --no-track            # capture benchmark only
"""

from __future__ import annotations

import argparse
import statistics as st
import time

import cv2

from capture import DirectGrabber, FrameGrabber
from config import BUFFER_GAP_MS, CAMERA_INDEX, DEFAULT_WORK_MS
from gestures import (
    HandTracker,
    MotionBuffer,
    SwipeClassifier,
    fingers_up,
    is_open_palm,
)
from hud import draw_hud, draw_landmarks, format_fingers, percentile


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
    gaps: list[float] = []
    vx_all: list[float] = []
    vx_open: list[float] = []
    swipes: list = []
    frames = repeats = hits = open_palm_frames = 0
    last_index = -1
    miss_run_start: float | None = None
    last_swipe_at = 0.0

    buffer = MotionBuffer()
    classifier = SwipeClassifier()
    tracker = None if args.no_track else HandTracker()

    t_start = time.perf_counter()
    cpu_start = time.process_time()
    print(f"capture: {mode}  tracking: {'off' if args.no_track else 'on'}  "
          f"{'headless' if args.headless else 'preview (q to quit)'}")

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
                velocity = None
                swipe = None

                if tracker is not None:
                    obs = tracker.process(frame)
                    if obs is not None:
                        hits += 1
                        palms.append(obs.palm_px)
                        confs.append(obs.confidence)
                        fingers = fingers_up(obs)
                        open_palm = is_open_palm(fingers)
                        open_palm_frames += open_palm
                        buffer.append(obs)
                        velocity = buffer.velocity()
                        if velocity is not None:
                            vx_all.append(abs(velocity[0]))
                            if open_palm:
                                vx_open.append(abs(velocity[0]))
                        swipe = classifier.update(buffer, open_palm)
                        if swipe is not None:
                            swipes.append(swipe)
                            last_swipe_at = now
                            print(f"  [{now - t_start:6.2f}s] SWIPE {swipe.action.upper():4} "
                                  f"peak={swipe.peak_velocity:5.1f} palm/s  "
                                  f"disp={swipe.displacement:4.1f}  "
                                  f"frames={swipe.frames}")
                        if miss_run_start is not None:
                            gaps.append(now - miss_run_start)
                            miss_run_start = None
                    elif miss_run_start is None:
                        miss_run_start = now
                        classifier.reset()
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
                    recent = swipes[-1] if swipes and now - last_swipe_at < 1.2 else None
                    draw_hud(view, [
                        f"loop: {frames / max(elapsed, 1e-6):5.1f} fps   "
                        f"track: {hits / max(frames, 1) * 100:5.1f}%",
                        f"fingers: {format_fingers(fingers)}" if obs else "fingers: -----",
                        "OPEN PALM" if open_palm else "-",
                        f"vx: {velocity[0]:+7.1f} palm/s   run: {classifier.run_frames}"
                        if velocity else "vx:      -- palm/s",
                        f">>> {recent.action.upper()} <<<" if recent else "",
                    ])
                    cv2.imshow("controller - step 8 swipe", view)
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
        print(f"buffer         {buffer.gap_invalidations} gap invalidations "
              f"(> {BUFFER_GAP_MS:.0f}ms)")
        for label, series in (("|vx| all", vx_all), ("|vx| open palm", vx_open)):
            if not series:
                continue
            v = sorted(series)
            print(f"{label:14} p50={st.median(v):5.1f}  p90={percentile(v, .90):5.1f}  "
                  f"p95={percentile(v, .95):5.1f}  p99={percentile(v, .99):5.1f}  "
                  f"max={v[-1]:5.1f}  palm/s")
        nxt = sum(s.action == "next" for s in swipes)
        print(f"SWIPES         {len(swipes)}  ({nxt} next, {len(swipes) - nxt} prev)"
              f"  = {len(swipes) / max(elapsed / 60, 1e-9):.1f} per minute")
        if swipes:
            peaks = sorted(s.peak_velocity for s in swipes)
            print(f"               peak velocity median={st.median(peaks):.1f} palm/s  "
                  f"min={peaks[0]:.1f}  max={peaks[-1]:.1f}")
        print(f"rejected       confidence={tracker.rejected['confidence']}  "
              f"palm_px={tracker.rejected['palm_px']}")
        if gaps:
            ms = sorted(g * 1000 for g in gaps)
            print(f"dropouts       count={len(ms)}  longest={ms[-1]:.0f}ms  "
                  f"median={st.median(ms):.0f}ms  over_150ms={sum(1 for g in ms if g > 150)}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
