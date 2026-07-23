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
    store.add_sample(alice.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(bob.speaker_id, np.array([0.0, 1.0]), "mic", 5.0, 1.0)
    roster = [
        Speaker("Alice", speaker_id=alice.speaker_id),
        Speaker("Bob", speaker_id=bob.speaker_id),
    ]
    resolver = SpeakerResolver(roster, store, FakeEngine([0.0, 1.0]), 16_000)

    decision = resolver.resolve(chunk(), np.array([0.0, 1.0], dtype=np.float32))
    assert decision.name == "Bob"
    assert decision.source == "auto"

    manual = resolver.resolve(chunk(manual=0), None)
    assert manual.name == "Alice"
    assert manual.source == "manual"
    assert store.profile(alice.speaker_id).enrollment_seconds == 5.0

    sample = resolver.add_manual_sample(
        0, chunk(manual=0, seconds=3.0), np.array([1.0, 0.0], dtype=np.float32)
    )
    assert sample.accepted is True
    assert sample.profile.enrollment_seconds == 8.0


def test_ambiguous_match_is_unknown(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    a = store.create_speaker("A")
    b = store.create_speaker("B")
    store.add_sample(a.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(b.speaker_id, np.array([0.99, 0.01]), "mic", 5.0, 1.0)
    resolver = SpeakerResolver(
        [Speaker("A", speaker_id=a.speaker_id), Speaker("B", speaker_id=b.speaker_id)],
        store,
        FakeEngine([1.0, 0.0]),
        16_000,
    )
    decision = resolver.resolve(chunk(), np.array([1.0, 0.0], dtype=np.float32))
    assert decision.source == "unknown"


def test_nearest_fallback_assigns_closest_in_call_below_threshold(tmp_path):
    # Best match (0.50) is below the confident bar but clearly closest — it
    # should be a "nearest" guess rather than "Unknown".
    store = SpeakerStore(tmp_path / "speakers.db")
    a = store.create_speaker("A")
    b = store.create_speaker("B")
    store.add_sample(a.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(b.speaker_id, np.array([0.0, 1.0]), "mic", 5.0, 1.0)
    resolver = SpeakerResolver(
        [Speaker("A", speaker_id=a.speaker_id), Speaker("B", speaker_id=b.speaker_id)],
        store,
        FakeEngine([0.5, 0.1]),
        16_000,
    )
    decision = resolver.resolve(chunk(), np.array([0.5, 0.1], dtype=np.float32))
    assert decision.source == "nearest"
    assert decision.name == "A"
    assert math.isclose(decision.confidence, 0.5, abs_tol=1e-6)


def test_weak_match_below_the_nearest_floor_stays_unknown(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    a = store.create_speaker("A")
    b = store.create_speaker("B")
    store.add_sample(a.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(b.speaker_id, np.array([0.0, 1.0]), "mic", 5.0, 1.0)
    resolver = SpeakerResolver(
        [Speaker("A", speaker_id=a.speaker_id), Speaker("B", speaker_id=b.speaker_id)],
        store,
        FakeEngine([0.3, 0.1]),
        16_000,
    )
    decision = resolver.resolve(chunk(), np.array([0.3, 0.1], dtype=np.float32))
    assert decision.source == "unknown"


def test_near_tie_below_threshold_is_left_unknown_not_guessed(tmp_path):
    # Two enrolled people are almost equally close and neither clears the
    # confident bar — a coin flip is worse than "Unknown", so stay unknown.
    store = SpeakerStore(tmp_path / "speakers.db")
    a = store.create_speaker("A")
    b = store.create_speaker("B")
    store.add_sample(a.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(b.speaker_id, np.array([0.0, 1.0]), "mic", 5.0, 1.0)
    resolver = SpeakerResolver(
        [Speaker("A", speaker_id=a.speaker_id), Speaker("B", speaker_id=b.speaker_id)],
        store,
        FakeEngine([0.5, 0.49]),
        16_000,
    )
    decision = resolver.resolve(chunk(), np.array([0.5, 0.49], dtype=np.float32))
    assert decision.source == "unknown"


def test_matching_never_considers_people_outside_roster(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    roster_person = store.create_speaker("Roster")
    outsider = store.create_speaker("Outsider")
    store.add_sample(roster_person.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(outsider.speaker_id, np.array([0.0, 1.0]), "mic", 5.0, 1.0)
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
    result = resolver.add_manual_sample(
        0, chunk(manual=0, seconds=3.0), np.array([1.0, 0.0], dtype=np.float32)
    )
    assert result.accepted is True
    assert resolver.profile_status(0) == ("learning", 3.0)
    assert store.profile(person.speaker_id).enrollment_seconds == 0.0

    resolver.persist_guest(0, person.speaker_id)
    assert store.profile(person.speaker_id).enrollment_seconds == 3.0


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
    first = resolver.add_manual_sample(
        0, chunk(manual=0, seconds=3.0), np.array([1.0, 0.0], dtype=np.float32)
    )
    assert first.profile.state == "learning"
    second = resolver.add_manual_sample(
        0, chunk(manual=0, seconds=3.0), np.array([1.0, 0.0], dtype=np.float32)
    )
    assert second.profile.state == "ready"


def test_source_specific_and_cross_source_thresholds(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    same_source = store.create_speaker("Same source")
    untrained = store.create_speaker("Untrained")
    vector_62 = np.array([0.62, math.sqrt(1.0 - 0.62**2)], dtype=np.float32)
    store.add_sample(same_source.speaker_id, vector_62, "mic", 5.0, 1.0)

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
    # 0.62 clears the source-specific bar (0.60) but not the stricter
    # cross-source bar (0.65), so it never becomes a confident "auto" match.
    # It's now offered as a lower-confidence "nearest" guess instead of unknown.
    assert decision.source == "nearest"


def test_matching_considers_every_rostered_person_on_either_source(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    mic_a = store.create_speaker("Mic A")
    mic_b = store.create_speaker("Mic B")
    system_speaker = store.create_speaker("Taylor")
    store.add_sample(mic_a.speaker_id, np.array([0.0, 1.0]), "mic", 5.0, 1.0)
    store.add_sample(mic_b.speaker_id, np.array([0.0, -1.0]), "mic", 5.0, 1.0)
    store.add_sample(system_speaker.speaker_id, np.array([1.0, 0.0]), "loopback", 5.0, 1.0)
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
    assert resolver.wants_embedding(chunk(manual=0, seconds=3.0)) is False
    assert resolver.add_manual_sample(
        0, chunk(manual=0, seconds=2.99), np.array([1.0, 0.0], dtype=np.float32)
    ).reason == "too_short"
    clipped = AudioChunk(
        samples=np.ones(48_000, dtype=np.float32),
        channel=Channel.MIC,
        start=0.0,
        end=3.0,
        manual_speaker_index=0,
        speech_seconds=3.0,
    )
    assert resolver.add_manual_sample(
        0, clipped, np.array([1.0, 0.0], dtype=np.float32)
    ).reason == "clipped"


def test_ready_profile_accepts_consistent_manual_sample(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    person = store.create_speaker("Local")
    store.add_sample(person.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    resolver = SpeakerResolver(
        [Speaker("Local", speaker_id=person.speaker_id)],
        store,
        FakeEngine([1.0, 0.0]),
        16_000,
    )

    result = resolver.add_manual_sample(
        0, chunk(manual=0, seconds=3.0), np.array([0.95, 0.05], dtype=np.float32)
    )

    assert result.accepted is True
    assert result.profile.enrollment_seconds == 8.0


def test_ready_profile_rejects_inconsistent_manual_sample(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    person = store.create_speaker("Local")
    store.add_sample(person.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    resolver = SpeakerResolver(
        [Speaker("Local", speaker_id=person.speaker_id)],
        store,
        FakeEngine([0.0, 1.0]),
        16_000,
    )

    result = resolver.add_manual_sample(
        0, chunk(manual=0, seconds=3.0), np.array([0.0, 1.0], dtype=np.float32)
    )

    assert result.accepted is False
    assert result.reason == "inconsistent"
    assert store.profile(person.speaker_id).enrollment_seconds == 5.0


def test_ready_profile_rejects_sample_that_is_ambiguous_with_roster(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    alice = store.create_speaker("Alice")
    bob = store.create_speaker("Bob")
    store.add_sample(alice.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(bob.speaker_id, np.array([0.99, 0.01]), "mic", 5.0, 1.0)
    resolver = SpeakerResolver(
        [
            Speaker("Alice", speaker_id=alice.speaker_id),
            Speaker("Bob", speaker_id=bob.speaker_id),
        ],
        store,
        FakeEngine([1.0, 0.0]),
        16_000,
    )

    result = resolver.add_manual_sample(
        0, chunk(manual=0, seconds=3.0), np.array([1.0, 0.0], dtype=np.float32)
    )

    assert result.accepted is False
    assert result.reason == "ambiguous_profile"


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

    assert gate.current is None
    assert gate.observe(alice) is None
    assert gate.observe(unknown) is None
    assert gate.observe(alice) is None
    assert gate.observe(alice) == alice
    assert gate.current == 0
    assert gate.observe(alice) is None
    assert gate.observe(bob) is None
    assert gate.observe(bob) == bob
    assert gate.current == 1


def test_speaker_resolver_tracking_engine_avoids_shared_lock(tmp_path):
    """embed() (turn attribution) and embed_for_tracking() (rolling
    tracker) must not serialize on one lock when each gets its own engine
    instance, or one hot path stalls behind the other."""
    import threading

    barrier = threading.Barrier(2, timeout=3.0)

    class BarrierEngine(SpeakerEmbeddingEngine):
        def embed(self, samples, sample_rate):
            barrier.wait()
            return np.array([1.0, 0.0], dtype=np.float32)

    store = SpeakerStore(tmp_path / "speakers.db")
    roster = [Speaker("Alice")]
    resolver = SpeakerResolver(
        roster, store, BarrierEngine(), 16_000, tracking_engine=BarrierEngine()
    )
    results: dict[str, np.ndarray] = {}

    def run_embed():
        results["embed"] = resolver.embed(chunk())

    def run_tracking():
        results["tracking"] = resolver.embed_for_tracking(chunk())

    embed_thread = threading.Thread(target=run_embed)
    tracking_thread = threading.Thread(target=run_tracking)
    embed_thread.start()
    tracking_thread.start()
    embed_thread.join(timeout=3.0)
    tracking_thread.join(timeout=3.0)

    assert not embed_thread.is_alive() and not tracking_thread.is_alive()
    assert "embed" in results and "tracking" in results
