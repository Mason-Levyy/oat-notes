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
        self.reopened = 0
        self.running = False

    def reopen_audio_host(self):
        self.reopened += 1

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


def test_hold_and_release_inserts_the_transcript(injected):
    controller, phases = make_controller()
    arm(controller)
    release(controller, held_seconds=2.0)
    assert injected == [("Hello world", {
        "hwnd": 4242, "restore_clipboard": True
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
        "hwnd": 4242, "restore_clipboard": True
    })]
    assert names(phases)[-1] == dictation.INSERTED


def test_quick_tap_that_caught_speech_still_commits(injected):
    recorder = FakeRecorder()
    recorder.speech_seconds = 2.0
    controller, _ = make_controller(recorder)
    arm(controller)
    release(controller, held_seconds=0.1)
    assert injected != []


def test_escape_cancels_a_latched_recording(injected):
    recorder = FakeRecorder()
    recorder.speech_seconds = 0.0
    controller, _ = make_controller(recorder)
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


def test_bind_registers_every_chord_and_escape():
    controller, _ = make_controller()
    listener = HotkeyListener()
    controller.bind(listener)
    assert set(listener._bindings) == {
        "dictation", "dictation-email", "dictation-replay", "dictation-cancel"
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


def test_rebind_hands_the_new_vocabulary_to_the_recorder():
    recorder = FakeRecorder()
    controller, _ = make_controller(recorder)
    controller.rebind(DictationOptions(vocabulary=(("ada", "Ada"),)))
    assert recorder.hotwords == "Ada"


def test_the_recorder_is_built_with_the_vocabulary_as_hotwords(monkeypatch):
    built = {}

    class RecordingRecorder(FakeRecorder):
        def __init__(self, config, transcriber, **kwargs):
            super().__init__()
            built.update(kwargs)

        def prepare(self):
            pass

    monkeypatch.setattr(dictation, "UtteranceRecorder", RecordingRecorder)
    controller, _ = make_controller(vocabulary=(("oat notes", "Oat Notes"),))
    controller.set_transcriber(object())
    assert built["hotwords"] == "Oat Notes"


def replay(controller, hwnd=4242):
    controller._dispatch(_Command(dictation._REPLAY, hwnd=hwnd))


def test_replay_reinserts_the_last_dictation(injected):
    controller, _ = make_controller()
    arm(controller)
    release(controller, held_seconds=2.0)
    replay(controller, hwnd=777)
    assert [text for text, _ in injected] == ["Hello world", "Hello world"]
    assert injected[1][1]["hwnd"] == 777


def test_replay_targets_the_window_focused_now(injected):
    controller, _ = make_controller()
    arm(controller, hwnd=111)
    release(controller, held_seconds=2.0)
    replay(controller, hwnd=999)
    assert injected[0][1]["hwnd"] == 111
    assert injected[1][1]["hwnd"] == 999


def test_replay_before_any_dictation_does_nothing(injected):
    controller, _ = make_controller()
    replay(controller)
    assert injected == []


def test_replay_while_recording_is_ignored(injected):
    controller, _ = make_controller()
    arm(controller)
    replay(controller)
    assert injected == []


def test_a_dropped_dictation_does_not_become_replayable(injected):
    controller, _ = make_controller(FakeRecorder(text=""))
    arm(controller)
    release(controller, held_seconds=2.0)
    replay(controller)
    assert injected == []


def test_replay_reports_inserted(injected):
    controller, phases = make_controller()
    arm(controller)
    release(controller, held_seconds=2.0)
    replay(controller)
    assert names(phases)[-1] == dictation.INSERTED


def dictate(controller, text):
    controller._recorder = FakeRecorder(text=text)
    arm(controller)
    release(controller, held_seconds=2.0)


def test_history_records_each_dictation_newest_first(injected):
    controller, _ = make_controller()
    for text in ("first one", "second one", "third one"):
        dictate(controller, text)
    assert [entry["text"] for entry in controller.history()] == [
        "Third one", "Second one", "First one"
    ]


def test_history_is_capped(injected):
    controller, _ = make_controller()
    for index in range(dictation.HISTORY_LIMIT + 5):
        dictate(controller, f"line {index}")
    assert len(controller.history()) == dictation.HISTORY_LIMIT
    assert controller.history()[0]["text"] == f"Line {dictation.HISTORY_LIMIT + 4}"


def test_history_entries_carry_a_mode_and_a_timestamp(injected):
    controller, _ = make_controller()
    dictate(controller, "hello there")
    entry = controller.history()[0]
    assert entry["mode"] == "text"
    assert entry["at"] > 0
    assert isinstance(entry["id"], int)


def test_history_ids_stay_unique_past_the_cap(injected):
    controller, _ = make_controller()
    for index in range(dictation.HISTORY_LIMIT + 5):
        dictate(controller, f"line {index}")
    ids = [entry["id"] for entry in controller.history()]
    assert len(set(ids)) == len(ids)


def test_dropped_dictations_are_not_recorded(injected):
    controller, _ = make_controller()
    dictate(controller, "")
    assert controller.history() == []


def test_clearing_history_also_forgets_the_replay_text(injected):
    controller, _ = make_controller()
    dictate(controller, "hello there")
    controller.clear_history()
    assert controller.history() == []
    assert controller.last_text is None
    injected.clear()
    replay(controller)
    assert injected == []


def test_history_snapshots_cannot_mutate_the_deque(injected):
    controller, _ = make_controller()
    dictate(controller, "hello there")
    controller.history()[0]["text"] = "tampered"
    assert controller.history()[0]["text"] == "Hello there"


def test_a_failed_insert_still_leaves_the_text_replayable(monkeypatch):
    def explode(text, **kwargs):
        raise inject.InjectionError("another window took focus")

    monkeypatch.setattr(inject, "inject", explode)
    controller, phases = make_controller()
    arm(controller)
    release(controller, held_seconds=2.0)

    assert names(phases)[-1] == dictation.ERROR
    assert controller.last_text == "Hello world"
    assert [entry["text"] for entry in controller.history()] == ["Hello world"]


def test_the_kept_text_can_then_be_replayed(injected, monkeypatch):
    def explode(text, **kwargs):
        raise inject.InjectionError("another window took focus")

    monkeypatch.setattr(inject, "inject", explode)
    controller, _ = make_controller()
    arm(controller)
    release(controller, held_seconds=2.0)

    monkeypatch.setattr(
        inject, "inject", lambda text, **kwargs: injected.append((text, kwargs))
    )
    replay(controller, hwnd=999)
    assert injected == [("Hello world", {
        "hwnd": 999, "restore_clipboard": True
    })]


def test_rescan_while_idle_reopens_the_audio_handle():
    recorder = FakeRecorder()
    controller, _ = make_controller(recorder)
    assert controller.rescan_audio_devices() is True
    assert recorder.reopened == 1


def test_rescan_while_recording_leaves_the_microphone_alone(injected):
    recorder = FakeRecorder()
    controller, _ = make_controller(recorder)
    arm(controller)
    assert controller.rescan_audio_devices() is False
    assert recorder.reopened == 0
    assert recorder.running


def test_rescan_before_the_model_loads_has_nothing_to_reopen():
    controller, _ = make_controller()
    controller._recorder = None
    assert controller.rescan_audio_devices() is True


def test_rescan_runs_on_the_worker_thread():
    recorder = FakeRecorder()
    controller, _ = make_controller(recorder)
    controller.start()
    try:
        assert controller.rescan_audio_devices() is True
    finally:
        controller.stop()
    assert recorder.reopened == 1


def test_a_failed_rescan_reports_the_error_and_returns_false():
    class BrokenHost(FakeRecorder):
        def reopen_audio_host(self):
            raise OSError("no audio host")

    controller, phases = make_controller(BrokenHost())
    assert controller.rescan_audio_devices() is False
    assert names(phases) == ["error"]
