import queue
import threading

import pytest

from oat_notes import overlay as hud
from oat_notes.overlay import (
    Overlay,
    OverlayState,
    meter_heights,
    phase_label,
    pixel_rounded_rows,
)


def test_recording_phases_show_the_pill():
    for phase in (hud.LISTENING, hud.LATCHED, hud.TRANSCRIBING, hud.FORMATTING):
        assert phase in hud._VISIBLE


def test_idle_hides_the_pill():
    assert hud.IDLE not in hud._VISIBLE
    assert hud.CANCELLED not in hud._VISIBLE


@pytest.mark.parametrize(
    "phase, expected",
    [
        (hud.LISTENING, "REC"),
        (hud.TRANSCRIBING, "THINKING"),
        (hud.INSERTED, "INSERTED"),
        (hud.ERROR, "ERROR"),
        (hud.LOADING, "LOADING"),
        (hud.IDLE, ""),
    ],
)
def test_phase_labels(phase, expected):
    assert phase_label(OverlayState(phase=phase)) == expected


def test_formatting_label_reflects_email_mode():
    assert phase_label(OverlayState(phase=hud.FORMATTING, email=True)) == "EMAIL"
    assert phase_label(OverlayState(phase=hud.FORMATTING, email=False)) == "CLEANING"


def test_explicit_label_wins():
    assert phase_label(OverlayState(phase=hud.LISTENING, label="CUSTOM")) == "CUSTOM"


def test_recording_in_email_mode_says_so():
    assert phase_label(OverlayState(phase=hud.LISTENING, email=True)) == "REC EMAIL"
    assert phase_label(OverlayState(phase=hud.LATCHED, email=True)) == "LOCKED EMAIL"


def test_email_suffix_is_only_for_recording_phases():
    assert phase_label(OverlayState(phase=hud.INSERTED, email=True)) == "INSERTED"


def test_every_label_fits_beside_the_meter():
    phases = (
        hud.LISTENING, hud.LATCHED, hud.TRANSCRIBING, hud.FORMATTING,
        hud.INSERTED, hud.CANCELLED, hud.ERROR, hud.LOADING,
    )
    for phase in phases:
        for email in (False, True):
            label = phase_label(OverlayState(phase=phase, email=email))
            assert len(label) <= hud.LABEL_BUDGET, f"{label!r} overruns the meter"


# -- pixel-rounded body ----------------------------------------------------


def test_corners_step_inward_and_the_middle_is_full_width():
    rows = pixel_rounded_rows(60, 30, step=3, corner_steps=2)
    assert rows[0][0] == 6
    assert rows[1][0] == 3
    assert rows[2][0] == 0
    assert rows[-1][0] == 6
    assert rows[-2][0] == 3


def test_body_is_symmetric_top_to_bottom():
    rows = pixel_rounded_rows(60, 30, step=3, corner_steps=2)
    insets = [left for left, _, _, _ in rows]
    assert insets == insets[::-1]


def test_rows_are_horizontally_centred():
    for left, _, right, _ in pixel_rounded_rows(60, 30, step=3, corner_steps=2):
        assert left == 60 - right


def test_rows_tile_the_full_height():
    rows = pixel_rounded_rows(60, 30, step=3, corner_steps=2)
    assert rows[0][1] == 0
    assert rows[-1][3] == 30
    for previous, current in zip(rows, rows[1:]):
        assert previous[3] == current[1]


def test_a_height_that_is_not_a_multiple_of_the_step_still_fits():
    rows = pixel_rounded_rows(60, 31, step=3, corner_steps=2)
    assert rows[-1][3] == 31


def test_square_corners_when_no_stepping_is_asked_for():
    rows = pixel_rounded_rows(60, 30, step=3, corner_steps=0)
    assert all(left == 0 and right == 60 for left, _, right, _ in rows)


# -- level meter -----------------------------------------------------------


def test_meter_is_flat_at_silence():
    heights = meter_heights(0.0)
    assert len(heights) == hud._METER_BARS
    assert all(height == pytest.approx(0.08) for height in heights)


def test_meter_rises_with_level():
    quiet = meter_heights(0.02)
    loud = meter_heights(0.4)
    assert max(loud) > max(quiet)


def test_meter_stays_within_bounds():
    for level in (-1.0, 0.0, 0.001, 0.5, 1.0, 50.0):
        assert all(0.0 <= height <= 1.0 for height in meter_heights(level))


def test_meter_arches_toward_the_middle():
    heights = meter_heights(0.3)
    middle = len(heights) // 2
    assert heights[middle] > heights[0]


# -- posting ---------------------------------------------------------------


def test_post_is_lossy_rather_than_blocking():
    overlay = Overlay()
    overlay._queue = queue.Queue(maxsize=2)
    for _ in range(50):
        overlay.post(OverlayState(phase=hud.LISTENING))
    assert overlay._queue.qsize() == 2


def test_drain_keeps_only_the_newest_state():
    overlay = Overlay()
    overlay._render = lambda: None
    for level in (0.1, 0.2, 0.9):
        overlay.post(OverlayState(phase=hud.LISTENING, level=level))
    overlay._drain()
    assert overlay._state.level == 0.9


def test_drain_with_nothing_queued_keeps_the_current_state():
    overlay = Overlay()
    overlay._render = lambda: None
    overlay._state = OverlayState(phase=hud.LISTENING, level=0.5)
    overlay._drain()
    assert overlay._state.level == 0.5


def test_post_phase_carries_the_email_flag():
    overlay = Overlay()
    overlay._render = lambda: None
    overlay.post_phase(hud.FORMATTING, {"email": True})
    overlay._drain()
    assert overlay._state.email is True


def test_post_phase_reads_email_from_an_inserted_mode():
    overlay = Overlay()
    overlay._render = lambda: None
    overlay.post_phase(hud.INSERTED, {"mode": "email"})
    overlay._drain()
    assert overlay._state.email is True


def test_levels_are_ignored_when_not_recording():
    overlay = Overlay()
    overlay._state = OverlayState(phase=hud.IDLE)
    overlay.post_level(0.9)
    assert overlay._queue.empty()

    overlay._state = OverlayState(phase=hud.LISTENING)
    overlay.post_level(0.9)
    assert not overlay._queue.empty()


# -- Win32 helpers ---------------------------------------------------------


def test_work_area_is_sane():
    left, top, right, bottom = hud.cursor_work_area()
    assert right > left and bottom > top


def test_focus_guard_tolerates_a_missing_window():
    hud.make_click_through_and_unfocusable(0)


def test_dpi_awareness_is_idempotent():
    hud.set_dpi_awareness()
    hud.set_dpi_awareness()


# -- shutdown fallback -----------------------------------------------------


def test_shutdown_falls_back_when_the_overlay_cannot_start():
    from oat_notes.server import _await_shutdown

    class BrokenOverlay:
        def run(self, should_stop):
            raise RuntimeError("no display")

    class FakeState:
        shutdown = threading.Event()

    state = FakeState()
    threading.Timer(0.05, state.shutdown.set).start()
    _await_shutdown(state, BrokenOverlay(), show_overlay=True)


def test_shutdown_skips_the_overlay_when_disabled():
    from oat_notes.server import _await_shutdown

    class ExplodingOverlay:
        def run(self, should_stop):
            raise AssertionError("overlay should not have been started")

    class FakeState:
        shutdown = threading.Event()

    state = FakeState()
    state.shutdown.set()
    _await_shutdown(state, ExplodingOverlay(), show_overlay=False)
