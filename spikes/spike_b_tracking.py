"""Spike B - does hand tracking actually work at real presenting distance?

Throwaway. Answers one question: at MEASURED_DISTANCE_M under venue lighting,
does MediaPipe hold a stable hand track - and is ROI cropping required to get
there? Sets ROI_CROP_REQUIRED in the spec.

    python spikes/spike_b_tracking.py                 # full-frame inference
    python spikes/spike_b_tracking.py --roi           # start in ROI-crop mode
    python spikes/spike_b_tracking.py --camera 1      # pick a different camera

Live keys:  r = toggle ROI crop   [ / ] = shrink/grow ROI   q = quit + summary

The number that matters is PALM px - the wrist-to-middle-MCP distance in source
pixels. MediaPipe's palm detector downsamples its input to a small fixed square,
so raw capture resolution does not help; a hand that is small in the frame is
small at the detector. Cropping to an ROI and upscaling is what helps, because
it makes the hand large *relative to the detector's input*.
"""

import argparse
import statistics as st
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = "models/hand_landmarker.task"

# Standard MediaPipe 21-landmark topology.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm
]


def crop_roi(frame, frac):
    """Centered crop covering `frac` of width, biased to the upper frame where
    gestures happen. Returns (cropped, (x0, y0, w, h)) for coordinate mapping."""
    h, w = frame.shape[:2]
    cw, ch = int(w * frac), int(h * frac)
    x0 = (w - cw) // 2
    y0 = int((h - ch) * 0.35)          # bias up: hands are raised, not centered
    return frame[y0:y0 + ch, x0:x0 + cw], (x0, y0, cw, ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--roi", action="store_true", help="start in ROI-crop mode")
    ap.add_argument("--roi-frac", type=float, default=0.5)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"FAIL: cannot open camera {args.camera}. Grant Camera permission "
              f"to Terminal, or try --camera 1.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    use_roi, roi_frac = args.roi, args.roi_frac
    frames = hits = 0
    palms, infer_ms, confs = [], [], []
    mode_stats = {}          # mode label -> [frames, hits]
    t_start = time.perf_counter()

    print("Stand at your real presenting distance. r=ROI  [ ]=size  q=quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        # Inference runs on the UNFLIPPED frame; we flip only for display.
        src = frame
        offset, scale = (0, 0), 1.0
        if use_roi:
            src, (x0, y0, cw, ch) = crop_roi(frame, roi_frac)
            src = cv2.resize(src, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            offset, scale = (x0, y0), cw / args.width

        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int((time.perf_counter() - t_start) * 1000),
        )
        infer_ms.append((time.perf_counter() - t0) * 1000)

        mode = f"ROI {roi_frac:.2f}" if use_roi else "full-frame"
        mode_stats.setdefault(mode, [0, 0])
        mode_stats[mode][0] += 1

        h, w = frame.shape[:2]
        pts, palm_px = None, None
        if res.hand_landmarks:
            hits += 1
            mode_stats[mode][1] += 1
            lm = res.hand_landmarks[0]
            # Map normalized landmarks back to FULL-FRAME pixel coordinates.
            pts = [(int(p.x * args.width * scale + offset[0]),
                    int(p.y * args.height * scale + offset[1])) for p in lm]
            palm_px = float(np.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]))
            palms.append(palm_px)
            if res.handedness:
                confs.append(res.handedness[0][0].score)

        view = cv2.flip(frame, 1)
        if pts:
            fpts = [(w - x, y) for x, y in pts]      # mirror to match the view
            for a, b in CONNECTIONS:
                cv2.line(view, fpts[a], fpts[b], (0, 255, 120), 2)
            for x, y in fpts:
                cv2.circle(view, (x, y), 3, (0, 200, 255), -1)
        if use_roi:
            _, (x0, y0, cw, ch) = crop_roi(frame, roi_frac)
            cv2.rectangle(view, (w - x0 - cw, y0), (w - x0, y0 + ch), (255, 180, 0), 2)

        rate = hits / frames * 100
        fps = frames / (time.perf_counter() - t_start)
        hud = [
            f"mode: {mode}",
            f"detect: {rate:5.1f}%   fps: {fps:4.1f}",
            f"PALM px: {palm_px:5.1f}" if palm_px else "PALM px:   --",
            f"infer: {st.median(infer_ms):4.1f}ms",
        ]
        for i, line in enumerate(hud):
            cv2.putText(view, line, (12, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, line, (12, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Spike B - tracking at distance", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            use_roi = not use_roi
        if k == ord("["):
            roi_frac = max(0.2, roi_frac - 0.05)
        if k == ord("]"):
            roi_frac = min(1.0, roi_frac + 0.05)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print("\n" + "=" * 58)
    print(f"frames={frames}  overall detection={hits/max(frames,1)*100:.1f}%")
    if infer_ms:
        print(f"inference  median={st.median(infer_ms):.1f}ms  p95={sorted(infer_ms)[int(len(infer_ms)*.95)-1]:.1f}ms")
    if palms:
        print(f"PALM px    median={st.median(palms):.1f}  min={min(palms):.1f}  max={max(palms):.1f}")
    if confs:
        print(f"confidence median={st.median(confs):.2f}  min={min(confs):.2f}")
    print("\nper mode:")
    for mode, (f, hgot) in mode_stats.items():
        print(f"  {mode:14} {hgot/max(f,1)*100:5.1f}% over {f} frames")
    print("=" * 58)
    print("Guide: detection >90% and PALM px >45 = comfortable. If full-frame is")
    print("weak but ROI is strong, ROI_CROP_REQUIRED = True.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
