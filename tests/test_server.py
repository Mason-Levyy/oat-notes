import queue

from oat_notes.config import Config
from oat_notes.server import AppState, EventHub


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
    assert status["last_saved"] is None


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
