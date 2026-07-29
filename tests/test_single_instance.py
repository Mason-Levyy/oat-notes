import json
import sys
import uuid

import pytest

from oat_notes.single_instance import SingleInstance

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="the instance lock is a Windows kernel mutex"
)


def lock(name, record):
    return SingleInstance(name=f"Local\\oat-notes-test-{name}", record=record)


def test_the_second_instance_is_turned_away(tmp_path):
    name = uuid.uuid4().hex
    record = tmp_path / "instance.json"

    first = lock(name, record)
    assert first.acquire(8737) is None

    second = lock(name, record)
    assert second.acquire(8737) == "http://127.0.0.1:8737"

    first.release()


def test_a_different_port_does_not_get_around_the_lock(tmp_path):
    # The regression this guards: binding the HTTP port was the only guard,
    # and --port walked straight around it — leaving two keyboard hooks
    # installed, so one chord pasted the same dictation twice.
    name = uuid.uuid4().hex
    record = tmp_path / "instance.json"

    first = lock(name, record)
    assert first.acquire(8737) is None

    second = lock(name, record)
    # Refused, and pointed at the UI that actually exists rather than the one
    # it asked for.
    assert second.acquire(9999) == "http://127.0.0.1:8737"

    first.release()


def test_releasing_lets_the_next_instance_start(tmp_path):
    name = uuid.uuid4().hex
    record = tmp_path / "instance.json"

    first = lock(name, record)
    assert first.acquire(8737) is None
    first.release()

    second = lock(name, record)
    assert second.acquire(8737) is None
    second.release()


def test_the_running_instance_records_where_to_find_it(tmp_path):
    name = uuid.uuid4().hex
    record = tmp_path / "instance.json"

    held = lock(name, record)
    assert held.acquire(8123) is None

    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["port"] == 8123
    assert isinstance(payload["pid"], int)

    held.release()
    assert not record.exists()


def test_a_corrupt_record_falls_back_to_the_requested_port(tmp_path):
    name = uuid.uuid4().hex
    record = tmp_path / "instance.json"

    first = lock(name, record)
    assert first.acquire(8737) is None
    record.write_text("{ not json", encoding="utf-8")

    second = lock(name, record)
    assert second.acquire(8737) == "http://127.0.0.1:8737"

    first.release()


def test_a_missing_record_still_refuses_the_second_instance(tmp_path):
    name = uuid.uuid4().hex
    record = tmp_path / "instance.json"

    first = lock(name, record)
    assert first.acquire(8737) is None
    record.unlink()

    second = lock(name, record)
    assert second.acquire(8737) == "http://127.0.0.1:8737"

    first.release()
