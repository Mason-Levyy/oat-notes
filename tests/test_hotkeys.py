import time
from types import SimpleNamespace

import pytest

from oat_notes import inject
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
    assert events == []
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


def test_the_injected_key_does_not_cancel_the_chord():
    listener, events = make_dictation_listener()
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_press(FakeKey(vk=0xFC))
    listener._handle_release(WIN)
    assert phases(events) == ["press", "release"]


def test_a_real_key_still_cancels_the_chord():
    listener, events = make_dictation_listener()
    listener._handle_press(CTRL)
    listener._handle_press(WIN)
    listener._handle_press(FakeKey(char="d"))
    assert phases(events) == ["press", "cancel"]


class SuppressedEvent(Exception):
    pass


class WindowsHookHarness:
    """Keep hook filtering separate from pynput's delayed key callbacks."""

    def __init__(self, monkeypatch, modifiers=("ctrl", "win")):
        self.listener, self.events = make_dictation_listener(modifiers)
        self.listener._listener = SimpleNamespace(suppress_event=self.suppress_event)
        self.callbacks = []
        self.queued = []
        self.delivered = []
        monkeypatch.setattr(inject, "_send", self.queued.extend)

    @staticmethod
    def suppress_event():
        raise SuppressedEvent

    def event(self, vk, up=False, tag=0):
        data = SimpleNamespace(vkCode=vk, dwExtraInfo=tag)
        try:
            notify = self.listener._filter_windows_event(0x0101 if up else 0x0100, data)
        except SuppressedEvent:
            return False
        self.delivered.append((vk, up))
        if notify:
            self.callbacks.append((vk, up))
        return True

    def drain(self):
        for event in self.queued:
            self.event(event.ki.wVk, bool(event.ki.dwFlags & 2), event.ki.dwExtraInfo)
        self.queued.clear()
        names = {0xA2: "ctrl_l", 0xA0: "shift_l", 0xA4: "alt_l",
                 inject.VK_LWIN: "cmd_l", inject.VK_RWIN: "cmd_r"}
        for vk, up in self.callbacks:
            key = FakeKey(vk=vk, name=names.get(vk, ""))
            callback = self.listener._handle_release if up else self.listener._handle_press
            callback(key)
        self.callbacks.clear()


@pytest.mark.parametrize("windows_key", [inject.VK_LWIN, inject.VK_RWIN])
@pytest.mark.parametrize("windows_first", [False, True])
@pytest.mark.parametrize("release_windows_first", [False, True])
def test_win_dictation_masks_release_in_every_order(
    monkeypatch, windows_key, windows_first, release_windows_first
):
    harness = WindowsHookHarness(monkeypatch)
    pressed = [windows_key, 0xA2] if windows_first else [0xA2, windows_key]
    released = [windows_key, 0xA2] if release_windows_first else [0xA2, windows_key]
    for vk in pressed:
        assert harness.event(vk)
    for vk in released:
        assert harness.event(vk, up=True) is (vk != windows_key)
    harness.drain()
    assert harness.delivered[-3:] == [
        (inject.VK_NONAME, False), (inject.VK_NONAME, True), (windows_key, True)
    ]
    assert phases(harness.events) == ["press", "release"]
    assert harness.listener._held == set()


def test_bare_windows_key_and_native_shortcuts_are_delivered(monkeypatch):
    harness = WindowsHookHarness(monkeypatch)
    assert harness.event(inject.VK_LWIN)
    assert harness.event(inject.VK_LWIN, up=True)
    harness.drain()
    assert harness.events == []
    assert harness.queued == []
    harness.event(0xA2)
    harness.event(inject.VK_LWIN)
    harness.event(ord("D"))
    assert harness.event(inject.VK_LWIN, up=True)
    harness.event(0xA2, up=True)
    harness.drain()
    assert phases(harness.events) == ["press", "cancel"]
    assert not any(vk == inject.VK_NONAME for vk, _ in harness.delivered)


def test_our_paste_events_do_not_cancel_or_arm_dictation(monkeypatch):
    harness = WindowsHookHarness(monkeypatch)
    harness.event(0xA2)
    harness.event(inject.VK_LWIN)
    harness.drain()
    harness.event(inject.VK_CONTROL, tag=inject.INPUT_TAG)
    harness.event(inject.VK_V, tag=inject.INPUT_TAG)
    harness.event(inject.VK_CONTROL, up=True, tag=inject.INPUT_TAG)
    harness.drain()
    assert phases(harness.events) == ["press"]
    assert harness.listener._physical_modifiers == {0xA2, inject.VK_LWIN}


def test_mask_failure_allows_physical_release_to_avoid_a_stuck_windows_key(monkeypatch):
    harness = WindowsHookHarness(monkeypatch)
    harness.event(0xA2)
    harness.event(inject.VK_LWIN)

    def fail(_vk):
        raise inject.InjectionError("input rejected")

    monkeypatch.setattr(inject, "release_windows_key", fail)
    assert harness.event(inject.VK_LWIN, up=True)
    harness.event(0xA2, up=True)
    harness.drain()
    assert phases(harness.events) == ["press", "release"]
    assert harness.listener._physical_modifiers == set()


def test_email_chord_masks_release_and_resets_between_cycles(monkeypatch):
    harness = WindowsHookHarness(monkeypatch, modifiers=("ctrl", "shift", "win"))
    for _ in range(2):
        for vk in (inject.VK_RWIN, 0xA0, 0xA2):
            harness.event(vk)
        harness.drain()
        for vk in (0xA2, 0xA0):
            harness.event(vk, up=True)
        assert not harness.event(inject.VK_RWIN, up=True)
        harness.drain()
    assert phases(harness.events) == ["press", "release", "press", "release"]
    assert harness.listener._consumed_windows_keys == set()


def test_non_windows_chord_passes_without_masking(monkeypatch):
    harness = WindowsHookHarness(monkeypatch, modifiers=("ctrl", "alt"))
    for vk in (0xA2, 0xA4):
        assert harness.event(vk)
    for vk in (0xA4, 0xA2):
        assert harness.event(vk, up=True)
    harness.drain()
    assert phases(harness.events) == ["press", "release"]
    assert not any(vk == inject.VK_NONAME for vk, _ in harness.delivered)


def make_letter_listener(modifiers=("ctrl", "alt"), letter="z"):
    fired = []
    listener = HotkeyListener()
    listener.bind("replay", Chord.of(modifiers, letter), lambda: fired.append(True))
    return listener, fired


def test_letter_chord_fires_on_the_plain_character():
    listener, fired = make_letter_listener()
    press_chord(listener, FakeKey(char="z"), (CTRL, ALT))
    assert fired == [True]


def test_letter_chord_survives_the_control_character():
    """Ctrl+Alt+Z reports \x1a on Windows, not "z"."""
    listener, fired = make_letter_listener()
    press_chord(listener, FakeKey(char="\x1a", vk=ord("Z")), (CTRL, ALT))
    assert fired == [True]


def test_letter_chord_is_case_insensitive():
    listener, fired = make_letter_listener()
    press_chord(listener, FakeKey(char="Z"), (CTRL, ALT))
    assert fired == [True]


def test_letter_chord_ignores_other_letters():
    listener, fired = make_letter_listener()
    press_chord(listener, FakeKey(char="x"), (CTRL, ALT))
    press_chord(listener, FakeKey(char="a", vk=ord("A")), (CTRL, ALT))
    assert fired == []


def test_a_letter_and_a_digit_chord_share_modifiers_without_colliding():
    switched = []
    replayed = []
    listener = HotkeyListener(switched.append, modifiers=("ctrl", "alt"))
    listener.bind("replay", Chord.of(("ctrl", "alt"), "z"), lambda: replayed.append(True))
    press_chord(listener, FakeKey(char="2"), (CTRL, ALT))
    press_chord(listener, FakeKey(char="z"), (CTRL, ALT))
    assert switched == [1]
    assert replayed == [True]
