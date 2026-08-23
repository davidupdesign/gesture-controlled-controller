"""Synthetic tests for the pose classifier.

The point of these is the rotation sweep. The spec rejected the usual
"fingertip y above PIP y" test in favour of radial distance from the wrist,
specifically so classification survives a rolled or tilted hand. That claim is
cheap to assert here and expensive to discover is false on stage.
"""

import math

import pytest

from controller import (
    FINGER_JOINTS,
    HandObservation,
    fingers_up,
    format_fingers,
    is_open_palm,
)

# Radii along each finger's axis, in arbitrary units from the wrist.
JOINT_RADII = {"mcp": 1.0, "pip": 1.45, "dip": 1.8}
TIP_EXTENDED, TIP_CURLED = 2.2, 1.05
THUMB_RADII = {1: 0.6, 2: 1.0, 3: 1.5}
THUMB_TIP_EXTENDED, THUMB_TIP_CURLED = 2.0, 0.95

# Direction each finger points, fanning out from the wrist (radians).
FINGER_DIRECTIONS = {
    "thumb": math.radians(-60),
    "index": math.radians(-20),
    "middle": math.radians(0),
    "ring": math.radians(20),
    "pinky": math.radians(40),
}
# (mcp, pip, dip, tip) landmark ids per non-thumb finger.
FINGER_CHAIN = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def synthetic_hand(extended, rotation=0.0, aspect=1.0, scale=0.12, centre=(0.5, 0.5)):
    """Build a 21-landmark hand with the named fingers extended or curled.

    Landmarks are laid out in a uniform metric space, rotated by `rotation`,
    then written into normalized image coordinates - dividing x by `aspect` so
    that HandObservation.dist, which multiplies it back, recovers the original
    geometry. That round-trip is itself part of what is under test.
    """
    pts = [(0.0, 0.0)] * 21

    def place(idx, direction, radius):
        angle = direction + rotation
        pts[idx] = (radius * math.cos(angle), radius * math.sin(angle))

    place(0, 0.0, 0.0)                                    # wrist at the origin
    thumb_dir = FINGER_DIRECTIONS["thumb"]
    for idx, radius in THUMB_RADII.items():
        place(idx, thumb_dir, radius)
    place(4, thumb_dir, THUMB_TIP_EXTENDED if extended.get("thumb") else THUMB_TIP_CURLED)

    for name, (mcp, pip, dip, tip) in FINGER_CHAIN.items():
        direction = FINGER_DIRECTIONS[name]
        place(mcp, direction, JOINT_RADII["mcp"])
        place(pip, direction, JOINT_RADII["pip"])
        place(dip, direction, JOINT_RADII["dip"])
        place(tip, direction, TIP_EXTENDED if extended.get(name) else TIP_CURLED)

    cx, cy = centre
    landmarks = [(cx + x * scale / aspect, cy + y * scale, 0.0) for x, y in pts]
    return HandObservation(
        landmarks=landmarks, palm_px=100.0, confidence=0.99,
        captured_at=0.0, frame_index=0, aspect=aspect,
    )


ALL = {n: True for n in FINGER_JOINTS}
NONE = {n: False for n in FINGER_JOINTS}


def test_open_hand_reads_all_fingers_extended():
    assert fingers_up(synthetic_hand(ALL)) == (True,) * 5


def test_fist_reads_no_fingers_extended():
    assert fingers_up(synthetic_hand(NONE)) == (False,) * 5


def test_open_palm_ignores_the_thumb():
    """A tucked thumb must not reject an otherwise open palm."""
    assert is_open_palm(fingers_up(synthetic_hand({**ALL, "thumb": False})))
    assert is_open_palm(fingers_up(synthetic_hand(ALL)))


@pytest.mark.parametrize("finger", ["index", "middle", "ring", "pinky"])
def test_one_curled_finger_breaks_open_palm(finger):
    assert not is_open_palm(fingers_up(synthetic_hand({**ALL, finger: False})))


@pytest.mark.parametrize("degrees", range(0, 360, 15))
def test_classification_is_rotation_invariant(degrees):
    """The central claim: a rolled or tilted hand classifies identically.

    A y-position-against-PIP classifier fails most of this sweep - an upside
    down open hand reads as a fist.
    """
    rotated = synthetic_hand(ALL, rotation=math.radians(degrees))
    assert fingers_up(rotated) == (True,) * 5, f"open hand misread at {degrees} deg"
    assert is_open_palm(fingers_up(rotated))

    fist = synthetic_hand(NONE, rotation=math.radians(degrees))
    assert fingers_up(fist) == (False,) * 5, f"fist misread at {degrees} deg"


@pytest.mark.parametrize("aspect", [1.0, 4 / 3, 16 / 9, 1280 / 720])
def test_classification_survives_non_square_framing(aspect):
    """Aspect correction must undo the squash of normalized x."""
    assert fingers_up(synthetic_hand(ALL, aspect=aspect)) == (True,) * 5
    assert fingers_up(synthetic_hand(NONE, aspect=aspect)) == (False,) * 5


def test_palm_length_is_positive_and_scales():
    """palm is the unit every later threshold divides by, so it must track size."""
    small = synthetic_hand(ALL, scale=0.06).palm
    large = synthetic_hand(ALL, scale=0.12).palm
    assert small > 0
    assert large == pytest.approx(small * 2, rel=1e-6)


def test_palm_length_is_rotation_invariant():
    lengths = [synthetic_hand(ALL, rotation=math.radians(d)).palm for d in range(0, 360, 30)]
    assert max(lengths) == pytest.approx(min(lengths), rel=1e-9)


def test_format_fingers_readout():
    assert format_fingers((True,) * 5) == "TIMRP"
    assert format_fingers((False, True, True, True, True)) == ".IMRP"
    assert format_fingers((False,) * 5) == "....."
