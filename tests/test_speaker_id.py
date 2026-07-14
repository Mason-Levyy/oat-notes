import math

import numpy as np

from oat_notes.attribution import Speaker
from oat_notes.speaker_id import (
    AttributionDecision,
    RollingSpeakerBuffer,
    SpeakerChangeGate,
    SpeakerEmbeddingEngine,
    SpeakerResolver,
)
from oat_notes.speaker_store import SpeakerStore
from oat_notes.types import AudioChunk, Channel


class FakeEngine(SpeakerEmbeddingEngine):
    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)

    def embed(self, samples, sample_rate):
        return self.embedding


def chunk(channel=Channel.MIC, manual=None, seconds=2.0):
    return AudioChunk(
        samples=np.full(int(16_000 * seconds), 0.2, dtype=np.float32),
        channel=channel,
        start=0.0,
        end=seconds,
        manual_speaker_index=manual,
        speech_seconds=seconds,
    )


def test_manual_turn_enrolls_then_automatic_turn_matches(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    alice = store.create_speaker("Alice")
    bob = store.create_speaker("Bob")
    store.add_sample(alice.speaker_id, np.array([1.0, 0.0]), "mic", 4.0, 1.0)
    store.add_sample(bob.speaker_id, np.array([0.0, 1.0]), "mic", 4.0, 1.0)
    roster = [
        Speaker("Alice", speaker_id=alice.speaker_id),
        Speaker("Bob", speaker_id=bob.speaker_id),
    ]
    resolver = SpeakerResolver(roster, store, FakeEngine([0.0, 1.0]), 16_000)

    decision = resolver.resolve(chunk(), np.array([0.0, 1.0], dtype=np.float32))
    assert decision.name == "Bob"
    assert decision.source == "auto"

    manual = resolver.resolve(
        chunk(manual=0), np.array([1.0, 0.0], dtype=np.float32)
    )
    assert manual.name == "Alice"
    assert manual.source == "manual"
    assert manual.profile.enrollment_seconds == 6.0


def test_ambiguous_match_is_unknown(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    a = store.create_speaker("A")
    b = store.create_speaker("B")
    store.add_sample(a.speaker_id, np.array([1.0, 0.0]), "mic", 4.0, 1.0)
    store.add_sample(b.speaker_id, np.array([0.99, 0.01]), "mic", 4.0, 1.0)
    resolver = SpeakerResolver(
        [Speaker("A", speaker_id=a.speaker_id), Speaker("B", speaker_id=b.speaker_id)],
        store,
        FakeEngine([1.0, 0.0]),
        16_000,
    )
    decision = resolver.resolve(chunk(), np.array([1.0, 0.0], dtype=np.float32))
    assert decision.source == "unknown"


def test_matching_never_considers_people_outside_roster(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    roster_person = store.create_speaker("Roster")
    outsider = store.create_speaker("Outsider")
    store.add_sample(roster_person.speaker_id, np.array([1.0, 0.0]), "mic", 4.0, 1.0)
    store.add_sample(outsider.speaker_id, np.array([0.0, 1.0]), "mic", 4.0, 1.0)
    resolver = SpeakerResolver(
        [
            Speaker("Roster", speaker_id=roster_person.speaker_id),
            Speaker("New untrained"),
        ],
        store,
        FakeEngine([0.0, 1.0]),
        16_000,
    )
    decision = resolver.resolve(chunk(), np.array([0.0, 1.0], dtype=np.float32))
    assert decision.source == "unknown"


def test_guest_embeddings_stay_temporary_until_linked(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    person = store.create_speaker("Visitor")
    resolver = SpeakerResolver([Speaker("Guest 1")], store, FakeEngine([1, 0]), 16_000)
    resolver.resolve(chunk(manual=0), np.array([1.0, 0.0], dtype=np.float32))
    assert resolver.profile_status(0) == ("learning", 2.0)
    assert store.profile(person.speaker_id).enrollment_seconds == 0.0

    resolver.persist_guest(0, person.speaker_id)
    assert store.profile(person.speaker_id).enrollment_seconds == 2.0


def test_enrollment_and_matching_need_no_network(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    store = SpeakerStore(tmp_path / "speakers.db")
    person = store.create_speaker("Local")
    resolver = SpeakerResolver(
        [Speaker("Local", speaker_id=person.speaker_id)],
        store,
        FakeEngine([1.0, 0.0]),
        16_000,
    )
    decision = resolver.resolve(
        chunk(manual=0, seconds=4.0), np.array([1.0, 0.0], dtype=np.float32)
    )
    assert decision.profile.state == "ready"


def test_source_specific_and_cross_source_thresholds(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    same_source = store.create_speaker("Same source")
    untrained = store.create_speaker("Untrained")
    vector_62 = np.array([0.62, math.sqrt(1.0 - 0.62**2)], dtype=np.float32)
    store.add_sample(same_source.speaker_id, vector_62, "mic", 4.0, 1.0)

    mic = SpeakerResolver(
        [
            Speaker("Same source", speaker_id=same_source.speaker_id),
            Speaker("Untrained", speaker_id=untrained.speaker_id),
        ],
        store,
        FakeEngine([1.0, 0.0]),
        16_000,
    )
    assert mic.resolve(chunk(), np.array([1.0, 0.0], dtype=np.float32)).source == "auto"

    cross_source = SpeakerResolver(
        [
            Speaker("Same source", speaker_id=same_source.speaker_id),
            Speaker("Untrained", speaker_id=untrained.speaker_id),
        ],
        store,
        FakeEngine([1.0, 0.0]),
        16_000,
    )
    decision = cross_source.resolve(
        chunk(channel=Channel.LOOPBACK), np.array([1.0, 0.0], dtype=np.float32)
    )
    assert decision.source == "unknown"


def test_matching_considers_every_rostered_person_on_either_source(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    mic_a = store.create_speaker("Mic A")
    mic_b = store.create_speaker("Mic B")
    system_speaker = store.create_speaker("Taylor")
    store.add_sample(mic_a.speaker_id, np.array([0.0, 1.0]), "mic", 4.0, 1.0)
    store.add_sample(mic_b.speaker_id, np.array([0.0, -1.0]), "mic", 4.0, 1.0)
    store.add_sample(system_speaker.speaker_id, np.array([1.0, 0.0]), "loopback", 4.0, 1.0)
    resolver = SpeakerResolver(
        [
            Speaker("Mic A", speaker_id=mic_a.speaker_id),
            Speaker("Mic B", speaker_id=mic_b.speaker_id),
            Speaker("Taylor", speaker_id=system_speaker.speaker_id),
        ],
        store,
        FakeEngine([1.0, 0.0]),
        16_000,
    )

    decision = resolver.resolve(chunk(), np.array([1.0, 0.0], dtype=np.float32))
    assert decision.name == "Taylor"
    assert decision.source == "auto"


def test_enrollment_rejects_short_or_clipped_turns(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    person = store.create_speaker("Local")
    resolver = SpeakerResolver(
        [Speaker("Local", speaker_id=person.speaker_id)],
        store,
        FakeEngine([1.0, 0.0]),
        16_000,
    )
    assert resolver.wants_embedding(chunk(manual=0, seconds=0.99)) is False
    clipped = AudioChunk(
        samples=np.ones(32_000, dtype=np.float32),
        channel=Channel.MIC,
        start=0.0,
        end=2.0,
        manual_speaker_index=0,
        speech_seconds=2.0,
    )
    assert resolver.wants_embedding(clipped) is False
    assert resolver.wants_embedding(chunk(manual=0, seconds=1.0)) is True


def test_rolling_speaker_buffer_checks_overlapping_windows_on_hop():
    buffer = RollingSpeakerBuffer(Channel.MIC, 100, 1.5, 0.5, 1.0)
    block = np.full(10, 0.2, dtype=np.float32)

    emitted = [buffer.push(index / 10, block, True) for index in range(15)]
    first = emitted[-1]
    assert first is not None
    assert first.duration == 1.5
    assert first.speech_seconds == 1.5

    assert all(buffer.push(1.5 + index / 10, block, True) is None for index in range(4))
    second = buffer.push(1.9, block, True)
    assert second is not None
    assert second.start == 0.5
    assert second.duration == 1.5


def test_rolling_speaker_buffer_skips_windows_without_one_second_of_speech():
    buffer = RollingSpeakerBuffer(Channel.MIC, 100, 1.5, 0.5, 1.0)
    block = np.full(10, 0.2, dtype=np.float32)

    result = None
    for index in range(15):
        result = buffer.push(index / 10, block, index < 9)
    assert result is None


def test_speaker_change_gate_requires_two_consecutive_matches():
    gate = SpeakerChangeGate(confirmations=2)
    alice = AttributionDecision("Alice", 0, "alice", "auto", 0.9)
    bob = AttributionDecision("Bob", 1, "bob", "auto", 0.9)
    unknown = AttributionDecision("Unknown", None, None, "unknown")

    assert gate.observe(alice) is None
    assert gate.observe(unknown) is None
    assert gate.observe(alice) is None
    assert gate.observe(alice) == alice
    assert gate.observe(alice) is None
    assert gate.observe(bob) is None
    assert gate.observe(bob) == bob
