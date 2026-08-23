"""Tests for the recorder's ground-truth scheduling.

If the prompt schedule and the recorded expectation windows disagree, every
number step 9 produces is wrong in a way that looks plausible - so this is
worth pinning down before any fixtures are captured against it.
"""

import pytest

from record import GO_S, PREP_S, REST_S, phase_at, prompt_schedule


def test_schedule_has_one_window_per_rep():
    schedule = prompt_schedule(10, "next")
    assert len(schedule) == 10
    assert [w["index"] for w in schedule] == list(range(10))
    assert {w["label"] for w in schedule} == {"next"}


def test_windows_are_go_length_and_do_not_overlap():
    schedule = prompt_schedule(5, "next")
    for window in schedule:
        assert window["end"] - window["start"] == pytest.approx(GO_S)
    for earlier, later in zip(schedule, schedule[1:]):
        assert later["start"] >= earlier["end"] + REST_S


def test_first_window_starts_after_a_prep_interval():
    """The performer needs time to get ready before the first prompt."""
    assert prompt_schedule(3, "next")[0]["start"] == pytest.approx(PREP_S)


def test_empty_schedule_for_zero_reps():
    assert prompt_schedule(0, "talking") == []


def test_phases_run_prep_then_go_then_rest():
    schedule = prompt_schedule(2, "next")
    first = schedule[0]
    assert phase_at(first["start"] - 0.1, schedule)[0] == "prep"
    assert phase_at(first["start"] + 0.1, schedule)[0] == "go"
    assert phase_at(first["end"] + 0.1, schedule)[0] == "rest"


def test_phase_reports_which_window_is_active():
    schedule = prompt_schedule(3, "next")
    _, window = phase_at(schedule[1]["start"] + 0.1, schedule)
    assert window["index"] == 1


def test_done_only_after_the_final_rest():
    schedule = prompt_schedule(2, "next")
    last = schedule[-1]
    assert phase_at(last["end"] + REST_S / 2, schedule)[0] == "rest"
    assert phase_at(last["end"] + REST_S + 0.1, schedule)[0] == "done"


@pytest.mark.parametrize("reps", [1, 5, 10, 25])
def test_total_duration_matches_the_schedule(reps):
    schedule = prompt_schedule(reps, "next")
    expected = reps * (PREP_S + GO_S + REST_S)
    assert schedule[-1]["end"] + REST_S == pytest.approx(expected)


def test_every_instant_before_done_has_a_phase():
    """No gaps: a frame must never fall outside the schedule and go unlabeled."""
    schedule = prompt_schedule(4, "next")
    end = schedule[-1]["end"] + REST_S
    t = 0.0
    while t < end:
        assert phase_at(t, schedule)[0] in {"prep", "go", "rest"}
        t += 0.05
