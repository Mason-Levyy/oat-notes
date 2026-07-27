import ctypes
import sys

import pytest

from oat_notes import inject

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="Win32 input injection"
)


@windows_only
def test_input_struct_matches_the_win32_layout():
    assert ctypes.sizeof(inject.INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)


@windows_only
def test_empty_injection_is_a_no_op(monkeypatch):
    monkeypatch.setattr(inject, "_send", _fail)
    inject.inject("")


def _fail(*_args, **_kwargs):
    raise AssertionError("should not have sent input")


@windows_only
def test_release_modifiers_forces_up_only_held_keys(monkeypatch):
    sent = []
    monkeypatch.setattr(inject, "held_modifiers", lambda: [inject.VK_CONTROL])
    monkeypatch.setattr(inject, "_send", lambda events: sent.extend(events))
    inject.release_modifiers(timeout=0.02)
    assert [event.ki.wVk for event in sent] == [inject.VK_CONTROL]
    assert all(event.ki.dwFlags & inject._KEYEVENTF_KEYUP for event in sent)


@windows_only
def test_release_modifiers_sends_nothing_when_already_clear(monkeypatch):
    monkeypatch.setattr(inject, "held_modifiers", lambda: [])
    monkeypatch.setattr(inject, "_send", _fail)
    inject.release_modifiers(timeout=0.02)


@windows_only
def test_paste_releases_modifiers_before_sending_ctrl_v(monkeypatch):
    order = []
    monkeypatch.setattr(inject, "read_clipboard_text", lambda: None)
    monkeypatch.setattr(
        inject, "write_clipboard_text", lambda text, private=True: order.append("write")
    )
    monkeypatch.setattr(
        inject, "release_modifiers", lambda *a, **k: order.append("release")
    )
    monkeypatch.setattr(inject, "_send", lambda events: order.append("send"))
    inject.paste_text("hello", restore_clipboard=False)
    assert order == ["write", "release", "send"]


@windows_only
def test_paste_sends_a_ctrl_v_chord(monkeypatch):
    sent = []
    monkeypatch.setattr(inject, "read_clipboard_text", lambda: None)
    monkeypatch.setattr(inject, "write_clipboard_text", lambda *a, **k: None)
    monkeypatch.setattr(inject, "release_modifiers", lambda *a, **k: None)
    monkeypatch.setattr(inject, "_send", lambda events: sent.extend(events))
    inject.paste_text("hello", restore_clipboard=False)
    assert [event.ki.wVk for event in sent] == [
        inject.VK_CONTROL, inject.VK_V, inject.VK_V, inject.VK_CONTROL
    ]
    assert not sent[0].ki.dwFlags & inject._KEYEVENTF_KEYUP
    assert sent[3].ki.dwFlags & inject._KEYEVENTF_KEYUP


@windows_only
def test_clipboard_is_restored_after_pasting(monkeypatch):
    written = []
    monkeypatch.setattr(inject, "read_clipboard_text", lambda: "previous contents")
    monkeypatch.setattr(
        inject,
        "write_clipboard_text",
        lambda text, private=True: written.append((text, private)),
    )
    monkeypatch.setattr(inject, "release_modifiers", lambda *a, **k: None)
    monkeypatch.setattr(inject, "_send", lambda events: None)
    monkeypatch.setattr(inject, "_CLIPBOARD_RESTORE_SECONDS", 0.01)
    inject.paste_text("dictated", restore_clipboard=True)
    _wait_for(lambda: len(written) == 2)
    assert written[0] == ("dictated", True)
    assert written[1] == ("previous contents", False)


@windows_only
def test_non_text_clipboard_is_not_clobbered_back(monkeypatch):
    written = []
    monkeypatch.setattr(inject, "read_clipboard_text", lambda: None)
    monkeypatch.setattr(
        inject, "write_clipboard_text", lambda text, private=True: written.append(text)
    )
    monkeypatch.setattr(inject, "release_modifiers", lambda *a, **k: None)
    monkeypatch.setattr(inject, "_send", lambda events: None)
    monkeypatch.setattr(inject, "_CLIPBOARD_RESTORE_SECONDS", 0.01)
    inject.paste_text("dictated", restore_clipboard=True)
    import time

    time.sleep(0.05)
    assert written == ["dictated"]


@windows_only
def test_inject_restores_focus_first(monkeypatch):
    order = []

    def restore(hwnd):
        order.append(("focus", hwnd))
        return True

    monkeypatch.setattr(inject, "restore_foreground", restore)
    monkeypatch.setattr(inject, "paste_text", lambda text, **k: order.append("paste"))
    inject.inject("hello", hwnd=1234)
    assert order == [("focus", 1234), "paste"]


@windows_only
def test_inject_refuses_when_focus_cannot_be_restored(monkeypatch):
    """Better to insert nothing than to type a dictation into Windows Search."""
    monkeypatch.setattr(inject, "restore_foreground", lambda hwnd: False)
    monkeypatch.setattr(inject, "paste_text", _fail)
    with pytest.raises(inject.InjectionError, match="took focus"):
        inject.inject("hello", hwnd=1234)




@windows_only
def test_clipboard_round_trip_against_the_real_clipboard():
    original = inject.read_clipboard_text()
    try:
        inject.write_clipboard_text("oat notes injection probe ✓")
        assert inject.read_clipboard_text() == "oat notes injection probe ✓"
    finally:
        if original is not None:
            inject.write_clipboard_text(original, private=False)


def _wait_for(predicate, timeout=2.0):
    import time

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")
