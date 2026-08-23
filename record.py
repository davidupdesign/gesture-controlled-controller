"""Record labeled landmark streams for offline classifier tuning.

A gesture classifier cannot be tuned by standing up and waving at a laptop.
Every threshold set from a live run so far has had to be revised, because each
run has different lighting, distance, and intent, and none of them can be
replayed. This records the raw landmark stream once so thresholds can be fitted
and re-fitted against fixed data.

    # 10 prompted swipes in each direction
    python record.py --label next --reps 10
    python record.py --label prev --reps 10

    # the false-positive corpus: talk with your hands, intend nothing
    python record.py --label talking --seconds 300

Output is one JSONL file per clip in clips/. The first line is a header holding
the label and, for prompted clips, the ground-truth windows during which a
gesture was requested. Everything after is one frame per line.

The talking corpus is the one that gets skipped and the one that matters: it is
the entire false-positive test, and false positives are what make a controller
untrustworthy. A missed swipe is an annoyance you recover from by swiping
again; a slide that advances mid-sentence is one you switch off.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from capture import FrameGrabber
from gestures import HandTracker
from hud import draw_hud

CLIPS_DIR = Path("clips")

# Prompted-clip cadence, seconds.
PREP_S, GO_S, REST_S = 2.0, 1.5, 1.5


def draw_banner(view, text, colour=(255, 255, 255), scale=2.0) -> None:
    """Large centred text - the prompt has to be readable from presenting distance."""
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 3)
    h, w = view.shape[:2]
    org = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(view, text, org, font, scale, (0, 0, 0), 9, cv2.LINE_AA)
    cv2.putText(view, text, org, font, scale, colour, 3, cv2.LINE_AA)


def prompt_schedule(reps: int, label: str) -> list[dict]:
    """Ground-truth windows: when a gesture was requested, and which one.

    Recorded as intervals rather than instants because the exact moment inside
    the GO window is the performer's choice. Step 9 scores a fire as correct if
    it lands within the window (plus a little slack for reaction time).
    """
    schedule, t = [], 0.0
    for i in range(reps):
        t += PREP_S
        schedule.append({"index": i, "label": label, "start": t, "end": t + GO_S})
        t += GO_S + REST_S
    return schedule


def phase_at(t: float, schedule: list[dict]) -> tuple[str, dict | None]:
    for window in schedule:
        if window["start"] - PREP_S <= t < window["start"]:
            return "prep", window
        if window["start"] <= t < window["end"]:
            return "go", window
        if window["end"] <= t < window["end"] + REST_S:
            return "rest", window
    return ("done", None) if schedule and t >= schedule[-1]["end"] + REST_S else ("prep", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="next | prev | talking | any name you want to score against")
    ap.add_argument("--reps", type=int, default=0, help="prompted repetitions")
    ap.add_argument("--seconds", type=float, default=0, help="free-running duration")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", type=Path, default=CLIPS_DIR)
    args = ap.parse_args()

    if not args.reps and not args.seconds:
        ap.error("give --reps for a prompted clip or --seconds for a free-running one")

    schedule = prompt_schedule(args.reps, args.label) if args.reps else []
    duration = (schedule[-1]["end"] + REST_S) if schedule else args.seconds

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    arrow = {"next": "SWIPE  -->", "prev": "<--  SWIPE"}.get(args.label, "GO")
    print(f"recording '{args.label}' for {duration:.0f}s -> {path}")
    print("q aborts without saving\n")

    frames_written = 0
    aborted = False
    t_start = time.perf_counter()

    with HandTracker() as tracker, FrameGrabber(args.camera) as grabber, \
            path.open("w") as out:
        out.write(json.dumps({
            "type": "header",
            "label": args.label,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reps": args.reps,
            "duration_s": duration,
            "expected": schedule,
        }) + "\n")

        last_index = -1
        while True:
            frame = grabber.read_new(last_index)
            if frame is None:
                if time.perf_counter() - t_start > duration:
                    break
                continue
            last_index = frame.index
            t = frame.captured_at - t_start
            if t > duration:
                break

            obs = tracker.process(frame)
            if obs is not None:
                out.write(json.dumps({
                    "t": round(t, 5),
                    "conf": round(obs.confidence, 4),
                    "palm_px": round(obs.palm_px, 2),
                    "aspect": round(obs.aspect, 6),
                    # Rounded to 5dp: well under a pixel at any sane resolution,
                    # and it roughly halves the file size.
                    "lm": [[round(v, 5) for v in pt] for pt in obs.landmarks],
                }) + "\n")
                frames_written += 1

            view = cv2.flip(frame.image, 1)
            if schedule:
                phase, window = phase_at(t, schedule)
                n = (window["index"] + 1) if window else args.reps
                banner = {"prep": f"READY  {n}/{args.reps}",
                          "go": arrow,
                          "rest": "rest",
                          "done": "done"}[phase]
                colour = {"prep": (120, 200, 255), "go": (80, 255, 120),
                          "rest": (160, 160, 160), "done": (255, 255, 255)}[phase]
                draw_banner(view, banner, colour, 2.4 if phase == "go" else 1.6)
            else:
                draw_banner(view, f"{duration - t:.0f}s", (160, 160, 160), 1.2)

            draw_hud(view, [
                f"label: {args.label}   {t:5.1f}s / {duration:.0f}s",
                f"frames: {frames_written}   tracked: {'yes' if obs else 'NO'}",
            ])
            cv2.imshow("record - q aborts", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                aborted = True
                break

    cv2.destroyAllWindows()

    if aborted:
        path.unlink(missing_ok=True)
        print("\naborted, nothing saved")
        return 1

    elapsed = time.perf_counter() - t_start
    print(f"\nwrote {frames_written} frames over {elapsed:.1f}s -> {path}")
    print(f"tracked {frames_written / max(elapsed * 29.5, 1) * 100:.0f}% of expected frames")
    if schedule:
        print(f"ground truth: {len(schedule)} '{args.label}' windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
