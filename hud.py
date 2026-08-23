"""Preview rendering."""

from __future__ import annotations

import cv2

from gestures import FINGER_NAMES, HAND_CONNECTIONS, HandObservation


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
