"""Tests for the swipe classifier.

The glitch-rejection cases are the point. Live recordings contain single-frame
velocity spikes of 109 and 220 palm-lengths/sec - landmark errors, not motion,
since the run producing the 220 had no buffer gap at all. A classifier that
fires on those advances slides at random, so the consistency requirement has to
be shown to actually suppress them rather than merely being present.
"""

import pytest

from gestures import MotionBuffer, SwipeClassifier
from tests.test_motion import observation

FPS = 30.0


def run(positions, open_palm=True, classifier=None, buffer=None, dt=1 / FPS, **kw):
    """Feed a position sequence through buffer + classifier; collect any swipes."""
    classifier = classifier or SwipeClassifier()
    buffer = buffer or MotionBuffer()
    fired = []
    for i, x in enumerate(positions):
        buffer.append(observation(i * dt, x, **kw))
        swipe = classifier.update(buffer, open_palm)
        if swipe is not None:
            fired.append(swipe)
    return fired


def sweep(start, step, n=12):
    return [start + i * step for i in range(n)]


# --- the gesture fires -------------------------------------------------------

def test_sustained_leftward_motion_fires_next():
    """Decreasing image x is the presenter's own right, which maps to next."""
    fired = run(sweep(0.8, -0.03))
    assert len(fired) >= 1
    assert fired[0].action == "next"
    assert fired[0].direction == -1


def test_sustained_rightward_motion_fires_prev():
    fired = run(sweep(0.2, 0.03))
    assert len(fired) >= 1
    assert fired[0].action == "prev"


def test_swipe_reports_its_evidence():
    swipe = run(sweep(0.8, -0.03))[0]
    assert swipe.frames >= SwipeClassifier().consistent_n
    assert swipe.peak_velocity >= SwipeClassifier().min_velocity
    assert swipe.displacement >= SwipeClassifier().min_displacement


def test_invert_flips_the_mapping():
    fired = run(sweep(0.8, -0.03), classifier=SwipeClassifier(invert=True))
    assert fired[0].action == "prev"


# --- the gesture does not fire -----------------------------------------------

def test_no_fire_without_open_palm():
    """Pose gating happens upstream; a moving hand that is not a palm is not a swipe."""
    assert run(sweep(0.8, -0.03), open_palm=False) == []


def test_no_fire_below_the_velocity_floor():
    assert run(sweep(0.5, -0.001)) == []


def test_no_fire_before_enough_consistent_frames():
    """Three qualifying frames is not four."""
    assert run(sweep(0.8, -0.03, n=3)) == []


def test_no_fire_on_a_stationary_hand():
    assert run([0.5] * 12) == []


# --- glitch rejection --------------------------------------------------------

def test_single_frame_spike_does_not_fire():
    """The measured failure mode: one bad landmark frame, then recovery.

    The excursion and the return have opposite signs, so the run breaks rather
    than accumulating - which is exactly why direction consistency is required
    and a bare velocity threshold is not enough.
    """
    assert run([0.5, 0.5, 0.5, 0.95, 0.5, 0.5, 0.5, 0.5]) == []


def test_oscillating_hand_does_not_fire():
    """Waving back and forth while talking must never accumulate a run."""
    assert run([0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5]) == []


def test_direction_reversal_resets_the_run():
    classifier = SwipeClassifier()
    buffer = MotionBuffer()
    run(sweep(0.5, -0.03, n=3), classifier=classifier, buffer=buffer)
    assert classifier.run_frames > 0
    # Reversing before confirmation must discard the partial run, not extend it.
    fired = run([0.5, 0.53, 0.56], classifier=classifier, buffer=buffer,
                dt=1 / FPS)
    assert fired == []


def test_diagonal_flail_is_rejected_by_y_drift():
    """Large vertical travel means the arm is doing something other than swiping."""
    classifier = SwipeClassifier(max_y_drift=0.5)
    buffer = MotionBuffer()
    fired = []
    for i in range(12):
        obs = observation(i / FPS, 0.8 - i * 0.03, wrist_y=0.5 + i * 0.03)
        buffer.append(obs)
        swipe = classifier.update(buffer, True)
        if swipe:
            fired.append(swipe)
    assert fired == []


# --- debouncing --------------------------------------------------------------

def test_one_continuous_sweep_fires_exactly_once():
    """A physical swipe spans 10-20 frames; it must not fire on each of them."""
    assert len(run(sweep(0.95, -0.02, n=20))) == 1


def test_a_second_swipe_can_fire_after_the_run_breaks():
    classifier = SwipeClassifier()
    first = run(sweep(0.95, -0.03), classifier=classifier)
    second = run(sweep(0.95, -0.03), classifier=classifier, buffer=MotionBuffer())
    assert len(first) == 1 and len(second) == 1


def test_reset_clears_a_partial_run():
    classifier = SwipeClassifier()
    buffer = MotionBuffer()
    run(sweep(0.8, -0.03, n=3), classifier=classifier, buffer=buffer)
    classifier.reset()
    assert classifier.run_frames == 0


# --- scale invariance --------------------------------------------------------

@pytest.mark.parametrize("palm_units", [0.05, 0.1, 0.2])
def test_firing_is_distance_invariant(palm_units):
    """The same physical swipe must fire whether the hand is near or far."""
    step = -palm_units * 0.3        # 0.3 palm-lengths per frame at any scale
    fired = run(sweep(0.95, step, n=14), palm_units=palm_units)
    assert len(fired) == 1, f"scale-dependent firing at palm={palm_units}"
