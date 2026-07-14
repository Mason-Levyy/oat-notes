import sqlite3

import numpy as np
import pytest

from oat_notes.speaker_store import (
    MAX_SAMPLES_PER_SOURCE,
    SpeakerStore,
)


def test_speaker_profiles_persist_without_audio(tmp_path):
    path = tmp_path / "speakers.db"
    store = SpeakerStore(path)
    speaker = store.create_speaker("Sarah")
    assert speaker.state == "untrained"

    store.add_sample(speaker.speaker_id, np.array([1.0, 0.0]), "mic", 2.0, 1.0)
    profile = store.add_sample(
        speaker.speaker_id, np.array([0.9, 0.1]), "mic", 2.0, 1.0
    )
    assert profile.state == "ready"
    assert SpeakerStore(path).profile(speaker.speaker_id).enrollment_seconds == 4.0

    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(voice_samples)")
        }
    assert "audio" not in columns
    assert "embedding" in columns


def test_speaker_names_are_case_insensitively_unique(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    store.create_speaker("Mason")
    with pytest.raises(ValueError):
        store.create_speaker("mason")


def test_samples_are_capped_per_source(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    speaker = store.create_speaker("Mason")
    for index in range(MAX_SAMPLES_PER_SOURCE + 3):
        store.add_sample(
            speaker.speaker_id,
            np.array([1.0, index / 100.0]),
            "mic",
            1.0,
            1.0,
        )
    for index in range(5):
        store.add_sample(
            speaker.speaker_id,
            np.array([1.0, index / 100.0]),
            "mic",
            1.0,
            1.0,
            model_key="next-model-version",
        )
    with sqlite3.connect(store.path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM voice_samples").fetchone()[0]
    assert count == MAX_SAMPLES_PER_SOURCE


def test_store_rejects_short_or_invalid_quality_samples(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    speaker = store.create_speaker("Mason")
    with pytest.raises(ValueError, match="at least one second"):
        store.add_sample(
            speaker.speaker_id, np.array([1.0, 0.0]), "mic", 0.99, 1.0
        )
    with pytest.raises(ValueError, match="quality"):
        store.add_sample(
            speaker.speaker_id, np.array([1.0, 0.0]), "mic", 1.0, -0.1
        )


def test_groups_round_trip_order_and_placement(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    a = store.create_speaker("A")
    b = store.create_speaker("B")
    group = store.create_group(
        "Standup",
        [
            {"speaker_id": b.speaker_id, "remote": True},
            {"speaker_id": a.speaker_id, "remote": False},
        ],
    )
    assert [member.name for member in group.members] == ["B", "A"]
    assert [member.remote for member in group.members] == [True, False]

    store.delete_speaker(b.speaker_id)
    assert [member.name for member in store.group(group.group_id).members] == ["A"]


def test_match_vectors_prefer_same_source_then_fall_back_global(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    speaker = store.create_speaker("A")
    store.add_sample(speaker.speaker_id, np.array([1.0, 0.0]), "mic", 4.0, 1.0)

    mic = store.match_vectors([speaker.speaker_id], "mic")[0]
    loopback = store.match_vectors([speaker.speaker_id], "loopback")[0]
    assert mic.source_specific is True
    assert loopback.source_specific is False


def test_schema_is_versioned_and_delete_cascades(tmp_path):
    path = tmp_path / "speakers.db"
    store = SpeakerStore(path)
    speaker = store.create_speaker("Temporary")
    store.add_sample(speaker.speaker_id, np.array([1.0, 0.0]), "mic", 4.0, 1.0)
    store.create_group(
        "Temporary group",
        [{"speaker_id": speaker.speaker_id, "remote": False}],
    )
    store.delete_speaker(speaker.speaker_id)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM voice_samples").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM group_members").fetchone()[0] == 0


def test_default_database_uses_local_app_data(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    store = SpeakerStore()
    assert store.path == tmp_path / "oat-notes" / "speakers.db"
