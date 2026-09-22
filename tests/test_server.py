from pathlib import Path
from types import SimpleNamespace

import pytest

from oat_notes.attribution import Speaker
from oat_notes.config import Config
from oat_notes.enrollment import EnrollmentProgress
from oat_notes.settings import AppSettings
from oat_notes.speaker_store import SpeakerStore
from oat_notes.types import Channel
from oat_notes.webapp import library, meeting, settings_api
from oat_notes.webapp.handler import is_trusted_request
from oat_notes.webapp.hub import EventHub
from oat_notes.webapp.requests import ApiError
from oat_notes.webapp.state import AppState, dictation_options


class FakeSession:
    """Stands in for Session — no real audio hardware or transcriber."""

    def __init__(self, guest_has_spoken: bool = True):
        self.stopped = False
        self.discarded = False
        self.saved_calls = []
        self.roster = [Speaker("Guest 1")]
        self.guest_indices = [0]
        self.log = SimpleNamespace(
            is_empty=False,
            speakers_with_lines=lambda: {"Guest 1"} if guest_has_spoken else set(),
        )
        self.active = {Channel.MIC: None, Channel.LOOPBACK: None}
        self.current_speaker = None
        self.profile_learning = None
        self.channels = (Channel.MIC,)
        self.hotkey_bank = 0

    def elapsed(self):
        return 0.0

    def profile_status(self, index):
        return "untrained", 0.0

    def stop(self):
        self.stopped = True

    def discard(self):
        self.discarded = True

    def save(self, renames=None):
        self.saved_calls.append(renames)
        return Path("fake.txt")


class FakeRecorder:
    """Stands in for VoiceEnrollmentRecorder — no real audio hardware."""

    def __init__(self, store, engine, speaker_id, config, on_progress, **kwargs):
        self.store = store
        self.engine = engine
        self.speaker_id = speaker_id
        self.on_progress = on_progress
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def fake_store(**profiles):
    """A library whose ``find`` knows only the given ids."""
    return SimpleNamespace(
        find=profiles.get,
        library=lambda: {"speakers": [], "groups": [], "ready_seconds": 5.0},
    )


def test_hub_fans_out_to_all_subscribers():
    hub = EventHub()
    first, second = hub.subscribe(), hub.subscribe()
    hub.publish({"type": "line", "text": "hello"})
    assert first.get_nowait()["text"] == "hello"
    assert second.get_nowait()["text"] == "hello"


def test_hub_unsubscribe_stops_delivery():
    hub = EventHub()
    subscriber = hub.subscribe()
    hub.unsubscribe(subscriber)
    hub.publish({"type": "line"})
    assert subscriber.empty()


def test_hub_full_subscriber_does_not_block_publish(caplog):
    hub = EventHub()
    subscriber = hub.subscribe()
    missed = [hub.publish({"type": "line"}) for _ in range(600)]
    assert subscriber.qsize() == 512
    assert missed == [0] * 512 + [1] * 88
    assert caplog.text.count("stopped reading events") == 1


def test_idle_status_shape():
    state = AppState(Config(), transcriber=None)
    status = state.status()
    assert status["recording"] is False
    assert status["lines"] == []
    assert status["backend"] == "faster_whisper"
    assert status["model_status"] == "loading"
    assert status["model_error"] is None
    assert status["settings"]["hotkey_modifiers"] == ["ctrl", "alt"]
    assert status["last_saved"] is None
    assert status["recovered"] == []
    assert status["active_speaker"] is None


def test_model_ready_updates_status_and_notifies_clients():
    state = AppState(Config(), transcriber=None)
    subscriber = state.hub.subscribe()
    transcriber = object()

    state.model_ready(transcriber, elapsed=1.25)

    assert state.transcriber is transcriber
    assert state.status()["model_status"] == "ready"
    assert state.status()["model_load_seconds"] == 1.25
    assert subscriber.get_nowait()["type"] == "status"


def test_model_failure_is_exposed_and_blocks_start():
    state = AppState(Config(), transcriber=None)
    state.model_failed(RuntimeError("NPU unavailable"), elapsed=2.5)

    status = state.status()
    assert status["model_status"] == "error"
    assert status["model_error"] == "RuntimeError: NPU unavailable"
    with pytest.raises(ApiError, match="RuntimeError: NPU unavailable"):
        meeting.start_session(state, {})


def test_start_is_blocked_while_model_loads():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError, match="transcription model is still loading") as raised:
        meeting.start_session(state, {})
    assert raised.value.status == 409


def test_settings_update_is_validated_and_persisted():
    saved = []
    state = AppState(Config(), transcriber=None, save_settings=saved.append)

    status = settings_api.update_settings(state, {"hotkey_modifiers": ["shift", "ctrl"]})

    assert status["settings"]["hotkey_modifiers"] == ["ctrl", "shift"]
    assert status["settings"]["hotkey_label"] == "Ctrl+Shift+1–9"
    assert saved == [AppSettings(("ctrl", "shift"))]


def test_invalid_settings_do_not_replace_current_value():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError) as raised:
        settings_api.update_settings(state, {"hotkey_modifiers": []})
    assert raised.value.status == 400
    assert state.settings == AppSettings()


def test_settings_cannot_change_during_meeting():
    state = AppState(Config(), transcriber=None)
    state.session = object()
    with pytest.raises(ApiError, match="end the meeting before changing hotkeys"):
        settings_api.update_settings(state, {"hotkey_modifiers": ["shift"]})


def test_cleanup_status_lifecycle():
    state = AppState(Config(), transcriber=None)
    assert state.status()["cleanup_status"] == "loading"

    state.cleaner_unavailable("disabled")
    assert state.status()["cleanup_status"] == "unavailable"
    assert state.status()["cleanup_error"] == "disabled"

    cleaner = object()
    state.cleaner_ready(cleaner)
    assert state.status()["cleanup_status"] == "ready"
    assert state.cleaner is cleaner

    state.cleaner_failed(RuntimeError("bad model"))
    assert state.status()["cleanup_status"] == "error"
    assert "RuntimeError" in state.status()["cleanup_error"]
    assert state.cleaner is None


def test_stop_without_session_reports_error():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError, match="not recording"):
        meeting.stop_session(state, {})


def test_stop_with_discard_skips_save_and_backfill():
    state = AppState(Config(), transcriber=None)
    session = FakeSession(guest_has_spoken=True)
    state.session = session

    status = meeting.stop_session(state, {"discard": True})

    assert session.stopped is True
    assert session.discarded is True
    assert session.saved_calls == []
    assert state.awaiting_backfill is None
    assert state.last_saved is None
    assert status["recording"] is False
    assert status["pending_backfill"] is None


def test_stop_without_discard_still_offers_backfill_for_a_guest_who_spoke():
    state = AppState(Config(), transcriber=None)
    session = FakeSession(guest_has_spoken=True)
    state.session = session

    status = meeting.stop_session(state, {"discard": False})

    assert session.saved_calls == []
    assert state.awaiting_backfill is session
    assert status["pending_backfill"] == ["Guest 1"]


def test_switch_without_session_is_harmless():
    state = AppState(Config(), transcriber=None)
    status = meeting.switch_speaker(state, {"index": 2})
    assert status["recording"] is False


def test_switch_index_is_validated_at_the_boundary():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError, match="index must be an integer") as raised:
        meeting.switch_speaker(state, {"index": "2"})
    assert raised.value.status == 400


def test_add_guest_without_session_reports_error():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError, match="not recording"):
        meeting.add_guest(state, {})


def test_finalize_without_pending_reports_error():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError, match="nothing awaiting backfill"):
        meeting.finalize(state, {"renames": {}})


def test_finalize_rejects_malformed_renames():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError, match="renames must be an object") as raised:
        meeting.finalize(state, {"renames": ["Guest 1"]})
    assert raised.value.status == 400


def test_idle_status_has_no_pending_backfill():
    state = AppState(Config(), transcriber=None)
    assert state.status()["pending_backfill"] is None


def test_trusted_host_no_origin():
    assert is_trusted_request("127.0.0.1:8737", None, 8737)
    assert is_trusted_request("localhost:8737", None, 8737)


def test_trusted_host_matching_origin():
    assert is_trusted_request("127.0.0.1:8737", "http://127.0.0.1:8737", 8737)
    assert is_trusted_request("localhost:8737", "http://localhost:8737", 8737)


def test_wrong_port_in_host_is_rejected():
    assert not is_trusted_request("127.0.0.1:9999", None, 8737)


def test_dns_rebinding_host_is_rejected():
    assert not is_trusted_request("evil.example.com:8737", None, 8737)


def test_missing_host_is_rejected():
    assert not is_trusted_request(None, None, 8737)


def test_cross_site_origin_is_rejected():
    assert not is_trusted_request("127.0.0.1:8737", "https://evil.example", 8737)


def test_origin_naming_a_different_loopback_port_is_rejected():
    assert not is_trusted_request("127.0.0.1:8737", "http://127.0.0.1:9999", 8737)


def test_local_speaker_library_crud_and_groups(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    state = AppState(Config(), transcriber=None, speaker_store=store)

    result = library.create_speaker(state, {"name": "Sarah"})
    speaker = result["library"]["speakers"][0]
    assert speaker["profile_state"] == "untrained"

    result = library.create_group(
        state, {"name": "Standup", "members": [{"speaker_id": speaker["id"]}]}
    )
    assert result["library"]["groups"][0]["members"][0]["speaker_id"] == speaker["id"]

    library.update_speaker(state, {"id": speaker["id"], "name": "Sarah K"})
    assert store.profile(speaker["id"]).name == "Sarah K"


def test_library_group_members_must_be_objects(tmp_path):
    state = AppState(Config(), transcriber=None, speaker_store=SpeakerStore(tmp_path / "s.db"))
    with pytest.raises(ApiError, match="group members must be a list of objects") as raised:
        library.create_group(state, {"name": "Standup", "members": ["sp-1"]})
    assert raised.value.status == 400


def test_speaker_library_mutations_are_blocked_during_meeting(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    state = AppState(Config(), transcriber=None, speaker_store=store)
    state.session = object()
    with pytest.raises(ApiError, match="end the meeting before changing the speaker library"):
        library.create_speaker(state, {"name": "Blocked"})
    assert store.list_speakers() == ()


def test_speaker_model_failure_does_not_block_transcription_model(tmp_path):
    state = AppState(
        Config(), transcriber=object(), speaker_store=SpeakerStore(tmp_path / "db.sqlite")
    )
    state.speaker_model_failed(RuntimeError("local model missing"))
    status = state.status()
    assert status["model_status"] == "ready"
    assert status["speaker_model_status"] == "error"


def test_voice_recording_blocked_during_meeting(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    speaker = store.create_speaker("Alice")
    state = AppState(
        Config(), transcriber=None, speaker_store=store, embedding_engine=object()
    )
    state.session = object()
    with pytest.raises(ApiError, match="end the meeting before recording a voice sample"):
        library.start_voice_recording(state, {"id": speaker.speaker_id})


def test_voice_recording_requires_speaker_model(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    speaker = store.create_speaker("Alice")
    state = AppState(Config(), transcriber=None, speaker_store=store)
    with pytest.raises(ApiError, match="speaker recognition is unavailable"):
        library.start_voice_recording(state, {"id": speaker.speaker_id})


def test_voice_recording_requires_existing_speaker(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    state = AppState(
        Config(), transcriber=None, speaker_store=store, embedding_engine=object()
    )
    with pytest.raises(ApiError, match="speaker not found") as raised:
        library.start_voice_recording(state, {"id": "does-not-exist"})
    assert raised.value.status == 404


def test_voice_recording_start_creates_recorder_and_blocks_double_start(tmp_path, monkeypatch):
    monkeypatch.setattr("oat_notes.enrollment.VoiceEnrollmentRecorder", FakeRecorder)
    store = SpeakerStore(tmp_path / "speakers.db")
    speaker = store.create_speaker("Alice")
    state = AppState(
        Config(), transcriber=None, speaker_store=store, embedding_engine=object()
    )

    result = library.start_voice_recording(state, {"id": speaker.speaker_id})
    assert result == {"ok": True, "speaker_id": speaker.speaker_id}
    assert state.enrollment is not None
    assert state.enrollment.started

    with pytest.raises(ApiError, match="a voice recording is already in progress"):
        library.start_voice_recording(state, {"id": speaker.speaker_id})


def test_voice_recording_progress_updates_state_and_clears_on_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr("oat_notes.enrollment.VoiceEnrollmentRecorder", FakeRecorder)
    store = SpeakerStore(tmp_path / "speakers.db")
    speaker = store.create_speaker("Alice")
    state = AppState(
        Config(), transcriber=None, speaker_store=store, embedding_engine=object()
    )
    library.start_voice_recording(state, {"id": speaker.speaker_id})
    recorder = state.enrollment

    recorder.on_progress(EnrollmentProgress(phase="listening", target_seconds=8.0))
    assert state.status()["enrollment"]["phase"] == "listening"

    recorder.on_progress(
        EnrollmentProgress(phase="done", captured_seconds=8.2, target_seconds=8.0)
    )
    assert state.enrollment is None
    assert state.status()["enrollment"] is None


def test_stop_voice_recording_without_active_recording_reports_error(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    state = AppState(Config(), transcriber=None, speaker_store=store)
    with pytest.raises(ApiError, match="no voice recording in progress"):
        library.stop_voice_recording(state, {})


def test_stop_voice_recording_stops_active_recorder(tmp_path, monkeypatch):
    monkeypatch.setattr("oat_notes.enrollment.VoiceEnrollmentRecorder", FakeRecorder)
    store = SpeakerStore(tmp_path / "speakers.db")
    speaker = store.create_speaker("Alice")
    state = AppState(
        Config(), transcriber=None, speaker_store=store, embedding_engine=object()
    )
    library.start_voice_recording(state, {"id": speaker.speaker_id})
    recorder = state.enrollment

    result = library.stop_voice_recording(state, {})
    assert result == {"ok": True}
    assert recorder.stopped


class FakeDictation:
    def __init__(self, enabled=True):
        from oat_notes.dictation.controller import DictationOptions

        self.options = DictationOptions(enabled=enabled)
        self.phase = "idle"
        self.last_error = None
        self.rebinds = []
        self.rescans = 0
        self.rescan_succeeds = True

    def rebind(self, options):
        self.options = options
        self.rebinds.append(options)

    def rescan_audio_devices(self):
        self.rescans += 1
        return self.rescan_succeeds


def dictation_state(**kwargs):
    return AppState(Config(), transcriber=None, dictation=FakeDictation(), **kwargs)


def test_status_reports_dictation():
    status = dictation_state().status()
    assert status["dictation"] == {"enabled": True, "phase": "idle", "error": None}


def test_status_without_dictation_reports_unavailable():
    state = AppState(Config(), transcriber=None)
    assert state.status()["dictation"]["phase"] == "unavailable"


def test_saving_dictation_settings_rebinds_the_chord():
    state = dictation_state(save_settings=lambda settings: None)
    settings_api.update_settings(state, {"dictation_modifiers": ["alt", "win"]})
    assert state.dictation.rebinds[-1].modifiers == ("alt", "win")


def test_partial_update_keeps_the_other_card_intact():
    state = dictation_state(save_settings=lambda settings: None)
    settings_api.update_settings(state, {"hotkey_modifiers": ["alt", "shift"]})
    settings_api.update_settings(state, {"dictation_spoken_punctuation": True})
    assert state.settings.hotkey_modifiers == ("alt", "shift")
    assert state.settings.dictation_spoken_punctuation is True


def start_meeting_with_fake_session(state, monkeypatch):
    rescans_seen_by_session = []

    class StartingSession(FakeSession):
        def __init__(self, config, options, transcriber, events):
            super().__init__()
            rescans_seen_by_session.append(state.dictation.rescans)

        def start(self):
            pass

    monkeypatch.setattr(meeting, "Session", StartingSession)
    state.transcriber = object()
    meeting.start_session(state, {})
    return rescans_seen_by_session


def test_starting_a_meeting_rescans_audio_devices_first(monkeypatch):
    state = dictation_state()
    assert start_meeting_with_fake_session(state, monkeypatch) == [1]


def test_a_busy_dictation_does_not_block_the_meeting(monkeypatch, caplog):
    state = dictation_state()
    state.dictation.rescan_succeeds = False
    assert start_meeting_with_fake_session(state, monkeypatch) == [1]
    assert state.session is not None
    assert "not rescanned" in caplog.text


def test_dictation_settings_may_change_during_a_meeting():
    state = dictation_state(save_settings=lambda settings: None)
    state.session = FakeSession()
    response = settings_api.update_settings(state, {"dictation_spoken_punctuation": True})
    assert response["recording"] is True
    assert state.settings.dictation_spoken_punctuation is True


def test_speaker_hotkey_still_locked_during_a_meeting():
    state = dictation_state(save_settings=lambda settings: None)
    state.session = object()
    with pytest.raises(ApiError, match="end the meeting before changing hotkeys"):
        settings_api.update_settings(state, {"hotkey_modifiers": ["shift"]})


def test_colliding_chords_are_refused():
    state = dictation_state(save_settings=lambda settings: None)
    with pytest.raises(ApiError, match="choose different modifiers"):
        settings_api.update_settings(state, {"dictation_modifiers": ["ctrl", "alt"]})
    assert state.settings.dictation_modifiers == ("ctrl", "win")


def test_toggle_dictation_flips_the_setting():
    state = dictation_state(save_settings=lambda settings: None)
    status = settings_api.toggle_dictation(state, {})
    assert status["settings"]["dictation_enabled"] is False
    assert state.dictation.rebinds[-1].enabled is False

    status = settings_api.toggle_dictation(state, {})
    assert status["settings"]["dictation_enabled"] is True


def test_toggle_dictation_accepts_an_explicit_value():
    state = dictation_state(save_settings=lambda settings: None)
    settings_api.toggle_dictation(state, {"enabled": False})
    assert state.settings.dictation_enabled is False
    settings_api.toggle_dictation(state, {"enabled": False})
    assert state.settings.dictation_enabled is False


def test_toggle_dictation_rejects_a_non_boolean():
    state = dictation_state(save_settings=lambda settings: None)
    with pytest.raises(ApiError, match="enabled must be true or false"):
        settings_api.toggle_dictation(state, {"enabled": "maybe"})


def test_toggle_without_dictation_reports_unavailable():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError, match="dictation is unavailable"):
        settings_api.toggle_dictation(state, {})


def test_settings_map_onto_dictation_options():
    options = dictation_options(
        AppSettings(
            dictation_modifiers=("alt", "win"),
            dictation_tap_ms=250,
            dictation_vocabulary=(("ada", "Ada"),),
            dictation_spoken_punctuation=True,
        )
    )
    assert options.modifiers == ("alt", "win")
    assert options.tap_seconds == 0.25
    assert options.vocabulary == (("ada", "Ada"),)
    assert options.spoken_punctuation is True


def test_the_saved_model_choice_reloads_the_transcriber(monkeypatch):
    saved = []
    state = AppState(Config(), transcriber=object(), save_settings=saved.append)
    reloads = []
    monkeypatch.setattr(
        settings_api, "reload_transcriber", lambda state: reloads.append(state.config.model_name)
    )

    status = settings_api.update_settings(state, {"whisper_model": "medium.en"})

    assert status["settings"]["whisper_model"] == "medium.en"
    assert state.config.model_name == "medium.en"
    assert state.transcriber is None
    assert status["model_status"] == "loading"
    assert reloads == ["medium.en"]


def test_saving_other_settings_does_not_reload_the_model(monkeypatch):
    state = AppState(Config(), transcriber=object())
    reloads = []
    monkeypatch.setattr(settings_api, "reload_transcriber", lambda state: reloads.append(1))

    settings_api.update_settings(state, {"hotkey_modifiers": ["shift"]})

    assert reloads == []
    assert state.transcriber is not None


def test_the_model_cannot_change_during_a_meeting():
    state = AppState(Config(), transcriber=object())
    state.session = object()
    with pytest.raises(ApiError, match="end the meeting before changing the transcription model"):
        settings_api.update_settings(state, {"whisper_model": "medium.en"})
    assert state.settings.whisper_model == "small.en"


def test_an_unknown_model_is_rejected():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError):
        settings_api.update_settings(state, {"whisper_model": "large-v3"})
    assert state.settings.whisper_model == "small.en"


def test_renaming_a_roster_member_passes_the_chosen_identity_through():
    state = AppState(Config(), transcriber=None)
    calls = []

    session = FakeSession()
    session.rename_speaker = lambda index, name, speaker_id=None: (
        calls.append((index, name, speaker_id)) or None
    )
    state.session = session
    state.speaker_store = fake_store(**{"sp-1": SimpleNamespace(name="Sarah")})

    meeting.rename_roster_speaker(state, {"index": 0, "name": "Sarah", "speaker_id": "sp-1"})

    assert calls == [(0, "Sarah", "sp-1")]


def test_renaming_against_an_unknown_identity_is_refused():
    state = AppState(Config(), transcriber=None)
    state.session = FakeSession()
    state.speaker_store = fake_store()

    with pytest.raises(ApiError, match="speaker not found") as raised:
        meeting.rename_roster_speaker(
            state, {"index": 0, "name": "Sarah", "speaker_id": "sp-gone"}
        )
    assert raised.value.status == 404


def test_a_named_guest_is_no_longer_pending_backfill():
    state = AppState(Config(), transcriber=None)
    session = FakeSession()
    session.guest_indices = []
    state.awaiting_backfill = session

    assert state.status()["pending_backfill"] == []


def test_resetting_a_profile_mid_meeting_reaches_the_session():
    state = AppState(Config(), transcriber=None)
    session = FakeSession()
    calls = []
    session.reset_speaker_profile = lambda index: calls.append(index) or None
    state.session = session

    meeting.reset_roster_profile(state, {"index": 0})

    assert calls == [0]


def test_resetting_a_profile_needs_a_meeting_and_an_integer_index():
    state = AppState(Config(), transcriber=None)
    with pytest.raises(ApiError, match="not recording"):
        meeting.reset_roster_profile(state, {"index": 0})

    state.session = FakeSession()
    with pytest.raises(ApiError, match="index must be an integer"):
        meeting.reset_roster_profile(state, {"index": "0"})


def test_a_session_error_reaches_the_browser_unchanged():
    state = AppState(Config(), transcriber=None)
    session = FakeSession()
    session.reset_speaker_profile = lambda index: "speaker already removed"
    state.session = session

    with pytest.raises(ApiError, match="speaker already removed") as raised:
        meeting.reset_roster_profile(state, {"index": 0})
    assert raised.value.status == 409


def _line_state():
    state = AppState(Config(), transcriber=None)
    state.lines.append(
        {"id": 0, "label": "Unknown", "text": "the churn number",
         "speaker_index": None, "attribution": "unknown", "confidence": None}
    )
    state.lines.append(
        {"id": 1, "label": "Alex", "text": "agreed",
         "speaker_index": 1, "attribution": "auto", "confidence": 0.9}
    )
    return state


def _lines(state):
    return state.lines.snapshot()[0]


def test_assigning_a_line_updates_it_and_tells_the_browser():
    state = _line_state()
    session = FakeSession()
    session.roster = [Speaker("Sarah"), Speaker("Alex")]
    session.assign_line = lambda line_id, index: None
    state.session = session
    subscriber = state.hub.subscribe()

    meeting.assign_line(state, {"id": 0, "index": 0})

    lines = _lines(state)
    assert lines[0]["label"] == "Sarah"
    assert lines[0]["speaker_index"] == 0
    assert lines[0]["attribution"] == "manual"
    assert lines[1]["label"] == "Alex"

    event = subscriber.get_nowait()
    assert event["type"] == "line_update"
    assert event["id"] == 0
    assert event["label"] == "Sarah"


def test_assigning_a_line_needs_a_live_meeting():
    state = _line_state()
    with pytest.raises(ApiError, match="not recording"):
        meeting.assign_line(state, {"id": 0, "index": 0})


def test_assigning_a_line_validates_its_arguments():
    state = _line_state()
    state.session = FakeSession()

    with pytest.raises(ApiError, match="id must be an integer"):
        meeting.assign_line(state, {"id": "0", "index": 0})
    with pytest.raises(ApiError, match="index must be an integer or null"):
        meeting.assign_line(state, {"id": 0, "index": True})


def test_a_session_refusal_to_assign_reaches_the_browser():
    state = _line_state()
    session = FakeSession()
    session.assign_line = lambda line_id, index: "line not found"
    state.session = session

    with pytest.raises(ApiError, match="line not found"):
        meeting.assign_line(state, {"id": 9, "index": 0})


def test_assigning_a_line_to_a_new_name_adds_them_and_tells_the_browser():
    state = _line_state()
    session = FakeSession()
    session.roster = [Speaker("Alex")]
    asked = []

    def assign_line_to_name(line_id, name, speaker_id=None):
        asked.append((line_id, name, speaker_id))
        session.roster.append(Speaker(name.strip(), speaker_id="sp-sarah"))
        return 1, None

    session.assign_line_to_name = assign_line_to_name
    state.session = session
    subscriber = state.hub.subscribe()

    response = meeting.assign_line(state, {"id": 0, "name": " Sarah "})

    assert response["recording"] is True
    assert asked == [(0, " Sarah ", None)]
    assert state.roster[1].name == "Sarah"
    assert _lines(state)[0]["label"] == "Sarah"
    assert _lines(state)[0]["speaker_index"] == 1
    events = [subscriber.get_nowait()["type"] for _ in range(2)]
    assert events == ["status", "line_update"]


def test_assigning_a_line_to_a_library_person_uses_their_saved_name():
    state = _line_state()
    state.speaker_store = fake_store(
        **{"sp-priya": SimpleNamespace(name="Priya", speaker_id="sp-priya")}
    )
    session = FakeSession()
    session.roster = [Speaker("Alex")]
    asked = []

    def assign_line_to_name(line_id, name, speaker_id=None):
        asked.append((name, speaker_id))
        session.roster.append(Speaker(name, speaker_id=speaker_id))
        return 1, None

    session.assign_line_to_name = assign_line_to_name
    state.session = session

    meeting.assign_line(state, {"id": 0, "name": "pri", "speaker_id": "sp-priya"})

    assert asked == [("Priya", "sp-priya")]


def test_assigning_a_line_to_an_unknown_library_id_is_refused():
    state = _line_state()
    state.speaker_store = fake_store()
    state.session = FakeSession()

    with pytest.raises(ApiError, match="speaker not found"):
        meeting.assign_line(state, {"id": 0, "name": "x", "speaker_id": "nope"})


def test_a_session_refusal_to_assign_by_name_reaches_the_browser():
    state = _line_state()
    session = FakeSession()
    session.assign_line_to_name = lambda line_id, name, speaker_id=None: (None, "a name is required")
    state.session = session

    with pytest.raises(ApiError, match="a name is required"):
        meeting.assign_line(state, {"id": 0, "name": ""})
