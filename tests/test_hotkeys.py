import time

import pytest

from oat_notes.hotkeys import DIGIT, Chord, HotkeyListener


class FakeKey:
    def __init__(self, char=None, vk=None, name=""):
        if char is not None:
            self.char = char
        if vk is not None:
            self.vk = vk
        self.name = name


CTRL = FakeKey(name="ctrl_l")
ALT = FakeKey(name="alt_l")
SHIFT = FakeKey(name="shift_l")
WIN = FakeKey(name="cmd_l")


def make_listener(modifiers=("ctrl", "alt")):
    received = []
    listener = HotkeyListener(received.append, modifiers=modifiers)
    return listener, received


def press_chord(listener, key, modifier_keys=(CTRL, ALT)):
    for modifier in modifier_keys:
        listener._handle_press(modifier)
    listener._handle_press(key)
    for modifier in reversed(modifier_keys):
        listener._handle_release(modifier)


def test_chord_fires_switch():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="2"))
    assert received == [1]


def test_bare_digit_does_not_fire():
    listener, received = make_listener()
    listener._handle_press(FakeKey(char="2"))
    assert received == []


def test_digit_after_release_does_not_fire():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="1"))
    listener._handle_press(FakeKey(char="2"))
    assert received == [0]


def test_ctrl_only_does_not_fire():
    listener, received = make_listener()
    listener._handle_press(CTRL)
    listener._handle_press(FakeKey(char="3"))
    assert received == []


def test_numpad_vk_path():
    listener, received = make_listener()
    press_chord(listener, FakeKey(vk=99))
    assert received == [2]


def test_control_character_vk_fallback():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="\x11", vk=ord("1")))
    assert received == [0]


def test_zero_and_letters_ignored():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="0"))
    press_chord(listener, FakeKey(char="a"))
    assert received == []


def test_out_of_roster_digit_still_forwards():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="9"))
    assert received == [8]


def test_custom_ctrl_shift_chord_fires():
    listener, received = make_listener(("ctrl", "shift"))
    press_chord(listener, FakeKey(char="4"), (CTRL, SHIFT))
    assert received == [3]


def test_old_chord_does_not_fire_after_customization():
    listener, received = make_listener(("ctrl", "shift"))
    press_chord(listener, FakeKey(char="4"), (CTRL, ALT))
    assert received == []


def test_extra_modifier_does_not_match():
    listener, received = make_listener(("ctrl", "alt"))
    press_chord(listener, FakeKey(char="2"), (CTRL, ALT, SHIFT))
    assert received == []


def test_windows_modifier_is_supported():
    listener, received = make_listener(("win", "shift"))
    press_chord(listener, FakeKey(char="5"), (WIN, SHIFT))
    assert received == [4]


def test_brackets_page_speaker_banks():
    switched = []
    pages = []
    listener = HotkeyListener(switched.append, on_page=pages.append)
    press_chord(listener, FakeKey(char="["))
    press_chord(listener, FakeKey(char="]"))
    assert pages == [-1, 1]
    assert switched == []


def test_bracket_virtual_key_fallback():
    pages = []
    listener = HotkeyListener(lambda index: None, on_page=pages.append)
    press_chord(listener, FakeKey(vk=219))
    press_chord(listener, FakeKey(vk=221))
    assert pages == [-1, 1]


# -- modifier-only chords (dictation push-to-talk) -------------------------


def make_dictation_listener(modifiers=("ctrl", "win")):
    events = []
    listener = HotkeyListener()
    listener.bind(
        "dictation",
        Chord.of(modifiers),
        on_press=lambda: events.append("press"),
        on_release=lambda held: events.append(("release", held)),
        on_cancel=lambda: events.append("cancel"),
    )
    return listener, events


def phases(events):
    return [event[0] if isinstance(event, tuple) else event for event in events]


def test_modifier_only_chord_arms_and_commits():
    listener, events = make_dictation_listener()
    listener._handle_press(CTRL)
    assert events == []  # partial chord does nothing
    listener._handle_press(WIN)
    listener._handle_release(WIN)
    listener._handle_release(CTRL)
    assert phases(events) == ["press", "release"]


def test_modifier_only_chord_reports_hold_duration():
    listener, events = make_dictation_listener()
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    time.sleep(0.02)
    listener._handle_release(WIN)
    assert events[1][1] >= 0.02


def test_other_key_cancels_armed_chord():
    listener, events = make_dictation_listener()
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_press(FakeKey(char="d"))
    listener._handle_release(WIN)
    listener._handle_release(CTRL)
    assert phases(events) == ["press", "cancel"]


def test_cancel_fires_once_for_repeated_keys():
    listener, events = make_dictation_listener()
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_press(FakeKey(char="d"))
    listener._handle_press(FakeKey(char="d"))
    assert phases(events) == ["press", "cancel"]


def test_chord_rearms_after_a_cancelled_cycle():
    listener, events = make_dictation_listener()
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_press(FakeKey(char="d"))
    listener._handle_release(WIN)
    listener._handle_release(CTRL)
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_release(WIN)
    assert phases(events) == ["press", "cancel", "press", "release"]


def test_extra_modifier_does_not_arm():
    listener, events = make_dictation_listener()
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_press(SHIFT)
    listener._handle_release(SHIFT)
    listener._handle_release(WIN)
    # Ctrl+Win armed, Shift joining is a different chord — but the arm already
    # happened, so releasing must still commit rather than strand the state.
    assert phases(events) == ["press", "release"]


def test_modifier_autorepeat_arms_once():
    listener, events = make_dictation_listener()
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_press(WIN)
    listener._handle_press(CTRL)
    assert phases(events) == ["press"]


def test_dictation_and_speaker_chords_coexist():
    switched = []
    listener = HotkeyListener(switched.append, modifiers=("ctrl", "alt"))
    events = []
    listener.bind(
        "dictation",
        Chord.of(("ctrl", "win")),
        on_press=lambda: events.append("press"),
        on_release=lambda held: events.append("release"),
    )
    press_chord(listener, FakeKey(char="2"))
    assert switched == [1] and events == []

    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_release(WIN)
    listener._handle_release(CTRL)
    assert switched == [1] and events == ["press", "release"]


def test_unbind_stops_delivery():
    listener, events = make_dictation_listener()
    listener.unbind("dictation")
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_release(WIN)
    assert events == []


def test_named_key_chord_matches_by_name():
    fired = []
    listener = HotkeyListener()
    listener.bind("cancel", Chord.of((), "esc"), on_press=lambda: fired.append(True))
    listener._handle_press(FakeKey(name="esc"))
    assert fired == [True]


def test_named_key_chord_ignores_held_modifiers():
    fired = []
    listener = HotkeyListener()
    listener.bind("cancel", Chord.of((), "esc"), on_press=lambda: fired.append(True))
    listener._handle_press(CTRL)
    listener._handle_press(FakeKey(name="esc"))
    assert fired == []


def test_callback_error_does_not_kill_the_hook():
    switched = []
    listener = HotkeyListener()
    listener.bind("boom", Chord.of(("ctrl", "alt"), DIGIT), on_press=_raise)
    listener.bind("good", Chord.of(("ctrl", "shift"), DIGIT), switched.append)
    press_chord(listener, FakeKey(char="1"), (CTRL, ALT))
    press_chord(listener, FakeKey(char="3"), (CTRL, SHIFT))
    assert switched == [2]


def _raise(*_):
    raise RuntimeError("callback exploded")


def test_chord_rejects_unknown_modifier():
    with pytest.raises(ValueError, match="unknown hotkey modifier"):
        Chord.of(("ctrl", "hyper"))


def test_chord_rejects_empty():
    with pytest.raises(ValueError, match="at least one modifier or a key"):
        Chord.of(())
