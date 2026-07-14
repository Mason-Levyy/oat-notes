import queue

from oat_notes.config import Config
from oat_notes.server import AppState, EventHub, is_trusted_request
from oat_notes.settings import AppSettings
from oat_notes.speaker_store import SpeakerStore


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


def test_hub_full_subscriber_does_not_block_publish():
    hub = EventHub()
    subscriber = hub.subscribe()
    for _ in range(600):
        hub.publish({"type": "line"})
    assert subscriber.qsize() == 512  # capped, publish never raised


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
    assert state.start_session({}) == {"error": "RuntimeError: NPU unavailable"}


def test_start_is_blocked_while_model_loads():
    state = AppState(Config(), transcriber=None)
    assert state.start_session({}) == {
        "error": "transcription model is still loading"
    }


def test_settings_update_is_validated_and_persisted():
    saved = []
    state = AppState(Config(), transcriber=None, save_settings=saved.append)

    status = state.update_settings({"hotkey_modifiers": ["shift", "ctrl"]})

    assert status["settings"]["hotkey_modifiers"] == ["ctrl", "shift"]
    assert status["settings"]["hotkey_label"] == "Ctrl+Shift+1–9"
    assert saved == [AppSettings(("ctrl", "shift"))]


def test_invalid_settings_do_not_replace_current_value():
    state = AppState(Config(), transcriber=None)
    response = state.update_settings({"hotkey_modifiers": []})
    assert "error" in response
    assert state.settings == AppSettings()


def test_settings_cannot_change_during_meeting():
    state = AppState(Config(), transcriber=None)
    state.session = object()
    response = state.update_settings({"hotkey_modifiers": ["shift"]})
    assert response == {"error": "end the meeting before changing hotkeys"}


def test_stop_without_session_reports_error():
    state = AppState(Config(), transcriber=None)
    assert state.stop_session() == {"error": "not recording"}


def test_switch_without_session_is_harmless():
    state = AppState(Config(), transcriber=None)
    status = state.switch_speaker(2)
    assert status["recording"] is False


def test_add_guest_without_session_reports_error():
    state = AppState(Config(), transcriber=None)
    assert state.add_guest({"remote": False}) == {"error": "not recording"}


def test_finalize_without_pending_reports_error():
    state = AppState(Config(), transcriber=None)
    assert state.finalize({"renames": {}}) == {"error": "nothing awaiting backfill"}


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

    result = state.create_library_speaker({"name": "Sarah"})
    speaker = result["library"]["speakers"][0]
    assert speaker["profile_state"] == "untrained"

    result = state.create_library_group(
        {
            "name": "Standup",
            "members": [{"speaker_id": speaker["id"], "remote": True}],
        }
    )
    assert result["library"]["groups"][0]["members"][0]["remote"] is True

    state.update_library_speaker({"id": speaker["id"], "name": "Sarah K"})
    assert store.profile(speaker["id"]).name == "Sarah K"


def test_speaker_library_mutations_are_blocked_during_meeting(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    state = AppState(Config(), transcriber=None, speaker_store=store)
    state.session = object()
    result = state.create_library_speaker({"name": "Blocked"})
    assert result == {"error": "end the meeting before changing the speaker library"}
    assert store.list_speakers() == ()


def test_speaker_model_failure_does_not_block_transcription_model(tmp_path):
    state = AppState(
        Config(), transcriber=object(), speaker_store=SpeakerStore(tmp_path / "db.sqlite")
    )
    state.speaker_model_failed(RuntimeError("local model missing"))
    status = state.status()
    assert status["model_status"] == "ready"
    assert status["speaker_model_status"] == "error"
