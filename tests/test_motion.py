"""Tests for the motion buffer.

The distance-invariance test is the one that matters. Every swipe threshold in
the spec is expressed in palm-lengths precisely so the same physical gesture
scores the same whether the hand is 0.4m or 1.5m from the camera - a 4x swing
in apparent size, measured. If normalization is wrong, thresholds tuned at a
desk are meaningless on stage.
"""

import math

import pytest

from config import BUFFER_GAP_MS
from gestures import WRIST, HandObservation, MotionBuffer


def observation(t, wrist_x, wrist_y=0.5, palm_units=0.1, aspect=1.0):
    """A hand at a given wrist position, with `palm_units` of palm length.

    Only the wrist and middle-MCP matter here; the rest is padding. x is
    divided by aspect so HandObservation.dist, which multiplies it back,
    recovers the intended geometry.
    """
    landmarks = [(0.0, 0.0, 0.0)] * 21
    landmarks[WRIST] = (wrist_x / aspect, wrist_y, 0.0)
    landmarks[9] = (wrist_x / aspect, wrist_y - palm_units, 0.0)   # middle MCP
    return HandObservation(
        landmarks=landmarks, palm_px=100.0, confidence=0.99,
        captured_at=t, frame_index=int(t * 1000), aspect=aspect,
    )


def feed(buffer, positions, dt=1 / 30, **kw):
    for i, x in enumerate(positions):
        buffer.append(observation(i * dt, x, **kw))
    return buffer


def test_empty_buffer_has_no_velocity():
    assert MotionBuffer().velocity() is None
    assert MotionBuffer().displacement() is None


def test_single_sample_has_no_velocity():
    buf = MotionBuffer()
    buf.append(observation(0.0, 0.5))
    assert buf.velocity() is None


def test_constant_velocity_is_measured_in_palm_lengths_per_second():
    """0.1 normalized units per frame at 30fps, with a 0.1 palm = 30 palm/s."""
    buf = feed(MotionBuffer(), [i * 0.1 for i in range(6)], palm_units=0.1)
    vx, _ = buf.velocity()
    assert vx == pytest.approx(30.0, rel=0.05)


def test_stationary_hand_has_zero_velocity():
    buf = feed(MotionBuffer(), [0.5] * 8)
    vx, vy = buf.velocity()
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy == pytest.approx(0.0, abs=1e-9)


def test_direction_sign_follows_motion():
    right = feed(MotionBuffer(), [0.3 + i * 0.05 for i in range(6)]).velocity()[0]
    left = feed(MotionBuffer(), [0.7 - i * 0.05 for i in range(6)]).velocity()[0]
    assert right > 0 and left < 0
    assert right == pytest.approx(-left, rel=1e-6)


@pytest.mark.parametrize("palm_units,label", [(0.2, "close to camera"), (0.05, "far away")])
def test_velocity_is_distance_invariant(palm_units, label):
    """The same gesture in palm-lengths scores the same at any apparent size.

    A hand twice as large in frame moves twice as many normalized units for the
    same physical swipe; dividing by palm length must cancel that exactly.
    """
    # Physical motion is 3 palm-lengths per 0.1s => 30 palm/s, whatever the scale.
    step = palm_units * 3 * (1 / 30) / 0.1
    buf = feed(MotionBuffer(), [i * step for i in range(6)], palm_units=palm_units)
    vx, _ = buf.velocity()
    assert vx == pytest.approx(30.0, rel=0.05), f"scale-dependent velocity when {label}"


@pytest.mark.parametrize("aspect", [1.0, 16 / 9, 1280 / 720])
def test_velocity_is_aspect_corrected(aspect):
    buf = feed(MotionBuffer(), [i * 0.1 for i in range(6)], palm_units=0.1, aspect=aspect)
    assert buf.velocity()[0] == pytest.approx(30.0, rel=0.05)


def test_gap_longer_than_threshold_clears_the_buffer():
    """The phantom-fire guard: a hand lost mid-swipe must not cross the gap."""
    buf = feed(MotionBuffer(), [0.1, 0.15, 0.2])
    assert len(buf) == 3

    late = (BUFFER_GAP_MS / 1000.0) * 2
    accepted = buf.append(observation(2 / 30 + late, 0.9))   # far away, long after
    assert accepted is False
    assert buf.gap_invalidations == 1
    assert len(buf) == 1                      # only the post-gap sample survives
    assert buf.velocity() is None             # and it cannot yet imply motion


def test_gap_shorter_than_threshold_is_tolerated():
    """A single dropped frame - the 34ms median measured live - must survive."""
    buf = feed(MotionBuffer(), [0.1, 0.15])
    accepted = buf.append(observation(1 / 30 + 0.034, 0.2))
    assert accepted is True
    assert buf.gap_invalidations == 0
    assert len(buf) == 3


def test_buffer_discards_samples_older_than_its_window():
    buf = MotionBuffer(window_s=0.2)
    feed(buf, [i * 0.01 for i in range(30)])          # 1 second at 30fps
    assert len(buf) <= 7                              # 0.2s holds about 6 frames
    assert buf._samples[-1].t - buf._samples[0].t <= 0.2 + 1e-9


def test_displacement_spans_the_whole_buffer():
    buf = feed(MotionBuffer(), [i * 0.05 for i in range(6)], palm_units=0.1)
    dx, _ = buf.displacement()
    assert dx == pytest.approx(2.5, rel=1e-6)         # 0.25 units / 0.1 palm


def test_palm_scale_is_median_smoothed_against_a_bad_frame():
    """One corrupt palm estimate must not rescale every velocity after it."""
    buf = MotionBuffer()
    for i in range(4):
        buf.append(observation(i / 30, i * 0.1, palm_units=0.1))
    buf.append(observation(4 / 30, 0.4, palm_units=0.4))   # a wildly wrong frame
    buf.append(observation(5 / 30, 0.5, palm_units=0.1))
    vx, _ = buf.velocity()
    assert vx == pytest.approx(30.0, rel=0.15)


def test_clear_resets_everything():
    buf = feed(MotionBuffer(), [0.1, 0.2, 0.3])
    buf.clear()
    assert len(buf) == 0
    assert buf.velocity() is None
