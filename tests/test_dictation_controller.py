import pytest

from oat_notes import inject
from oat_notes.config import Config
from oat_notes.dictation import controller as dictation
from oat_notes.dictation.controller import (
    DictationController,
    DictationOptions,
    _Command,
)
from oat_notes.hotkeys import HotkeyListener

CONFIG = Config()


class FakeRecorder:
    def __init__(self, text="hello world"):
        self.text = text
        self.speech_seconds = 5.0
        self.started = 0
        self.cancelled = 0
        self.running = False

    def start(self):
        self.started += 1
        self.running = True

    def stop(self):
        self.running = False
        return self.text

    def cancel(self):
        self.cancelled += 1
        self.running = False

    def close(self):
        pass


@pytest.fixture
def injected(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inject, "inject", lambda text, **kwargs: calls.append((text, kwargs))
    )
    monkeypatch.setattr(inject, "foreground_window", lambda: 4242)
    return calls


def make_controller(recorder=None, **option_overrides):
    phases = []
    controller = DictationController(
        CONFIG,
        options=DictationOptions(**option_overrides),
        on_phase=lambda phase, details: phases.append((phase, details)),
    )
    controller._recorder = recorder or FakeRecorder()
    return controller, phases


def names(phases):
    return [phase for phase, _ in phases]


def arm(controller, hwnd=4242, force_email=False):
    controller._dispatch(_Command(dictation._ARM, hwnd=hwnd, force_email=force_email))


def release(controller, held_seconds=1.0):
    controller._dispatch(_Command(dictation._RELEASE, held_seconds=held_seconds))


def cancel(controller):
    controller._dispatch(_Command(dictation._CANCEL))


# -- push to talk ----------------------------------------------------------


def test_hold_and_release_inserts_the_transcript(injected):
    controller, phases = make_controller()
    arm(controller)
    release(controller, held_seconds=2.0)
    assert injected == [("Hello world", {
        "method": inject.PASTE, "hwnd": 4242, "restore_clipboard": True
    })]
    assert names(phases) == [
        dictation.LISTENING,
        dictation.TRANSCRIBING,
        dictation.FORMATTING,
        dictation.INSERTED,
    ]


def test_silence_inserts_nothing(injected):
    controller, phases = make_controller(FakeRecorder(text=""))
    arm(controller)
    release(controller, held_seconds=2.0)
    assert injected == []
    assert names(phases)[-1] == dictation.IDLE


def test_target_window_is_captured_at_arm_not_at_insert(injected):
    controller, _ = make_controller()
    arm(controller, hwnd=999)
    release(controller, held_seconds=2.0)
    assert injected[0][1]["hwnd"] == 999


def test_other_key_cancels_without_inserting(injected):
    recorder = FakeRecorder()
    controller, phases = make_controller(recorder)
    arm(controller)
    cancel(controller)
    release(controller, held_seconds=2.0)
    assert injected == []
    assert recorder.cancelled == 1
    assert names(phases) == [dictation.LISTENING, dictation.CANCELLED]


def test_release_without_arm_is_ignored(injected):
    controller, phases = make_controller()
    release(controller, held_seconds=2.0)
    assert injected == [] and phases == []


# -- hybrid latching -------------------------------------------------------


def test_quick_tap_latches_instead_of_committing(injected):
    recorder = FakeRecorder()
    recorder.speech_seconds = 0.0
    controller, phases = make_controller(recorder)
    arm(controller)
    release(controller, held_seconds=0.1)
    assert injected == []
    assert recorder.running is True
    assert names(phases) == [dictation.LISTENING, dictation.LATCHED]


def test_second_tap_commits_a_latched_recording(injected):
    recorder = FakeRecorder()
    recorder.speech_seconds = 0.0
    controller, phases = make_controller(recorder)
    arm(controller)
    release(controller, held_seconds=0.1)
    arm(controller)
    release(controller, held_seconds=0.1)
    assert injected == [("Hello world", {
        "method": inject.PASTE, "hwnd": 4242, "restore_clipboard": True
    })]
    assert names(phases)[-1] == dictation.INSERTED


def test_quick_tap_that_caught_speech_still_commits(injected):
    recorder = FakeRecorder()
    recorder.speech_seconds = 2.0
    controller, _ = make_controller(recorder)
    arm(controller)
    release(controller, held_seconds=0.1)
    assert injected != []


def test_hold_mode_never_latches(injected):
    recorder = FakeRecorder()
    recorder.speech_seconds = 0.0
    controller, _ = make_controller(recorder, activation=dictation.HOLD)
    arm(controller)
    release(controller, held_seconds=0.05)
    assert injected != []


def test_toggle_mode_always_latches(injected):
    recorder = FakeRecorder()
    controller, phases = make_controller(recorder, activation=dictation.TOGGLE)
    arm(controller)
    release(controller, held_seconds=5.0)
    assert injected == []
    assert names(phases)[-1] == dictation.LATCHED
    arm(controller)
    assert injected != []


def test_escape_cancels_a_latched_recording(injected):
    recorder = FakeRecorder()
    recorder.speech_seconds = 0.0
    controller, phases = make_controller(recorder)
    arm(controller)
    release(controller, held_seconds=0.1)
    controller._cancel_if_active()
    controller._dispatch(controller._commands.get_nowait())
    assert injected == []
    assert recorder.cancelled == 1


def test_escape_while_idle_does_nothing():
    controller, _ = make_controller()
    controller._cancel_if_active()
    assert controller._commands.empty()


# -- formatting hand-off ---------------------------------------------------


def test_formatter_result_is_what_gets_inserted(injected):
    controller = DictationController(
        CONFIG,
        options=DictationOptions(),
        format_text=lambda text, force_email: (text.upper(), "email"),
    )
    controller._recorder = FakeRecorder()
    arm(controller)
    release(controller, held_seconds=2.0)
    assert injected[0][0] == "HELLO WORLD"


def test_email_chord_requests_email_formatting(injected):
    seen = []

    def formatter(text, force_email):
        seen.append(force_email)
        return text, "email" if force_email else "text"

    controller = DictationController(
        CONFIG, options=DictationOptions(), format_text=formatter
    )
    controller._recorder = FakeRecorder()
    arm(controller, force_email=True)
    release(controller, held_seconds=2.0)
    assert seen == [True]


def test_formatter_dropping_everything_inserts_nothing(injected):
    controller = DictationController(
        CONFIG, options=DictationOptions(), format_text=lambda text, _: ("", "text")
    )
    controller._recorder = FakeRecorder()
    arm(controller)
    release(controller, held_seconds=2.0)
    assert injected == []


# -- failure handling ------------------------------------------------------


def test_arm_before_the_model_loads_reports_loading(injected):
    controller, phases = make_controller()
    controller._recorder = None
    arm(controller)
    assert names(phases) == [dictation.LOADING]
    assert injected == []


def test_injection_failure_leaves_the_controller_idle(injected, monkeypatch):
    def explode(text, **kwargs):
        raise inject.InjectionError("no foreground window")

    monkeypatch.setattr(inject, "inject", explode)
    recorder = FakeRecorder()
    controller, phases = make_controller(recorder)
    arm(controller)
    release(controller, held_seconds=2.0)
    assert names(phases)[-1] == dictation.ERROR
    assert controller._recording is False
    arm(controller)
    assert controller._recording is True


def test_phase_listener_error_does_not_break_dictation(injected):
    def explode(phase, details):
        raise RuntimeError("ui blew up")

    controller = DictationController(CONFIG, on_phase=explode)
    controller._recorder = FakeRecorder()
    arm(controller)
    release(controller, held_seconds=2.0)
    assert injected != []


# -- bindings --------------------------------------------------------------


def test_bind_registers_both_chords_and_escape():
    controller, _ = make_controller()
    listener = HotkeyListener()
    controller.bind(listener)
    assert set(listener._bindings) == {
        "dictation", "dictation-email", "dictation-cancel"
    }
    controller.unbind(listener)
    assert listener._bindings == {}


def test_disabled_dictation_registers_nothing():
    controller, _ = make_controller(enabled=False)
    listener = HotkeyListener()
    controller.bind(listener)
    assert listener._bindings == {}


def test_rebind_cancels_a_recording_in_flight():
    recorder = FakeRecorder()
    controller, _ = make_controller(recorder)
    listener = HotkeyListener()
    controller.bind(listener)
    arm(controller)
    controller.rebind(DictationOptions(modifiers=("ctrl", "shift")))
    controller._dispatch(controller._commands.get_nowait())
    assert recorder.cancelled == 1
    assert controller._recording is False


def test_rebind_while_idle_is_harmless():
    controller, phases = make_controller()
    listener = HotkeyListener()
    controller.bind(listener)
    controller.rebind(DictationOptions(modifiers=("ctrl", "shift")))
    controller._dispatch(controller._commands.get_nowait())
    assert phases == []


def test_rebind_swaps_the_chord():
    controller, _ = make_controller()
    listener = HotkeyListener()
    controller.bind(listener)
    controller.rebind(DictationOptions(modifiers=("ctrl", "shift")))
    assert listener._bindings["dictation"].chord.modifiers == frozenset(
        {"ctrl", "shift"}
    )
