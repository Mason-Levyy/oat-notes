"""Pipeline threading and multi-channel routing, with no real audio or model."""

import math
import threading

import numpy as np

from oat_notes.clock import SessionClock
from oat_notes.config import Config
from oat_notes.pipeline import Pipeline
from oat_notes.transcriber import Transcriber
from oat_notes.types import Channel, TranscriptSegment

CONFIG = Config()
WINDOW = CONFIG.vad_window_samples
SILENCE_WINDOWS = math.ceil(
    CONFIG.silence_split_seconds * CONFIG.sample_rate / WINDOW
)


class EnergyFakeVad:
    """Speech wherever the samples are loud — lets tests shape audio directly."""

    window_samples = WINDOW

    def speech_probability(self, window):
        return 1.0 if float(np.abs(window).max()) > 0.5 else 0.0


class FakeTranscriber(Transcriber):
    def transcribe(self, chunk):
        return TranscriptSegment(
            text=f"heard {chunk.duration:.1f}s",
            channel=chunk.channel,
            start=chunk.start,
            end=chunk.end,
        )


def speech(windows):
    return np.ones(windows * WINDOW, dtype=np.float32)


def silence(windows):
    return np.zeros(windows * WINDOW, dtype=np.float32)


def make_pipeline(channels):
    received = []
    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        FakeTranscriber(),
        lambda segment, latency, embedding=None: received.append(segment),
        channels=channels,
        vad_factory=EnergyFakeVad,
    )
    return pipeline, received


def test_channels_route_to_separate_chunkers():
    pipeline, received = make_pipeline((Channel.MIC, Channel.LOOPBACK))
    pipeline.start()

    pipeline.frame_queue.put((Channel.MIC, 0.0, speech(31)))
    pipeline.frame_queue.put((Channel.MIC, 31 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS)))
    pipeline.frame_queue.put((Channel.LOOPBACK, 5.0, speech(31)))
    pipeline.frame_queue.put((Channel.LOOPBACK, 5.0 + 31 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS)))
    pipeline.finish()

    assert [segment.channel for segment in received] == [
        Channel.MIC,
        Channel.LOOPBACK,
    ]
    mic_segment, loopback_segment = received
    assert mic_segment.start < 1.0
    assert 4.5 < loopback_segment.start <= 5.0


def test_finish_flushes_all_channels():
    pipeline, received = make_pipeline((Channel.MIC, Channel.LOOPBACK))
    pipeline.start()

    pipeline.frame_queue.put((Channel.MIC, 0.0, speech(20)))
    pipeline.frame_queue.put((Channel.LOOPBACK, 0.0, speech(20)))
    pipeline.finish()

    assert {segment.channel for segment in received} == {
        Channel.MIC,
        Channel.LOOPBACK,
    }


def test_worker_applies_attributor():
    from oat_notes.attribution import Attributor, SwitchLog, parse_speakers

    log = SwitchLog()
    log.record(1.2, 1)
    received = []
    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        FakeTranscriber(),
        lambda segment, latency, embedding=None: received.append(segment),
        channels=(Channel.MIC, Channel.LOOPBACK),
        vad_factory=EnergyFakeVad,
        attributor=Attributor(
            parse_speakers("Mason, Sarah, Priya"),
            mic_log=log,
            loopback_log=SwitchLog(initial=2),
        ),
    )
    pipeline.start()
    pipeline.frame_queue.put((Channel.MIC, 0.0, speech(31)))
    pipeline.frame_queue.put(
        (Channel.MIC, 31 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    pipeline.frame_queue.put((Channel.LOOPBACK, 5.0, speech(31)))
    pipeline.frame_queue.put(
        (Channel.LOOPBACK, 5.0 + 31 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    pipeline.finish()

    speakers = {segment.channel: segment.speaker for segment in received}
    assert speakers[Channel.MIC] == "Mason"
    assert speakers[Channel.LOOPBACK] == "Priya"


def test_hotkey_split_attributes_back_to_back_speakers():
    """Two speakers with NO pause between them: the press must cut the chunk."""
    from oat_notes.attribution import Attributor, SwitchLog, parse_speakers

    log = SwitchLog()
    received = []
    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        FakeTranscriber(),
        lambda segment, latency, embedding=None: received.append(segment),
        channels=(Channel.MIC,),
        vad_factory=EnergyFakeVad,
        attributor=Attributor(
            parse_speakers("Alice, Bob"), mic_log=log, loopback_log=SwitchLog()
        ),
    )
    pipeline.start()

    alice_end = 31 * WINDOW / CONFIG.sample_rate
    pipeline.frame_queue.put((Channel.MIC, 0.0, speech(31)))
    log.record(alice_end, 1)
    pipeline.split_channel(Channel.MIC)
    pipeline.frame_queue.put((Channel.MIC, alice_end, speech(31)))
    pipeline.frame_queue.put(
        (Channel.MIC, 2 * alice_end, silence(SILENCE_WINDOWS))
    )
    pipeline.finish()

    assert [segment.speaker for segment in received] == ["Alice", "Bob"]
    first, second = received
    assert second.start == first.end


def test_a_full_frame_queue_reports_the_dropped_command(caplog):
    pipeline, _ = make_pipeline((Channel.MIC,))
    while not pipeline.frame_queue.full():
        pipeline.frame_queue.put_nowait((Channel.MIC, 0.0, silence(1)))

    assert pipeline.split_channel(Channel.MIC) is False
    assert pipeline.manual_override(Channel.MIC, 0) is False
    assert pipeline.cancel_profile_learning() is False
    assert "split dropped" in caplog.text
    assert "manual override dropped" in caplog.text


def test_mic_only_pipeline_still_works():
    pipeline, received = make_pipeline((Channel.MIC,))
    pipeline.start()
    pipeline.frame_queue.put((Channel.MIC, 0.0, speech(31)))
    pipeline.frame_queue.put(
        (Channel.MIC, 31 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    pipeline.finish()
    assert len(received) == 1
    assert received[0].text.startswith("heard")


def test_embedding_and_transcription_run_in_parallel(tmp_path):
    from oat_notes.attribution import Speaker
    from oat_notes.speaker_id import SpeakerEmbeddingEngine, SpeakerResolver
    from oat_notes.speaker_store import SpeakerStore

    barrier = threading.Barrier(2, timeout=3.0)

    class BarrierEngine(SpeakerEmbeddingEngine):
        def embed(self, samples, sample_rate):
            barrier.wait()
            return np.array([1.0, 0.0], dtype=np.float32)

    class BarrierTranscriber(Transcriber):
        def transcribe(self, chunk):
            barrier.wait()
            return TranscriptSegment("parallel", chunk.channel, chunk.start, chunk.end)

    store = SpeakerStore(tmp_path / "speakers.db")
    saved = store.create_speaker("Alice")
    other = store.create_speaker("Bob")
    store.add_sample(saved.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(other.speaker_id, np.array([0.0, 1.0]), "mic", 5.0, 1.0)
    roster = [
        Speaker("Alice", speaker_id=saved.speaker_id),
        Speaker("Bob", speaker_id=other.speaker_id),
    ]
    resolver = SpeakerResolver(roster, store, BarrierEngine(), CONFIG.sample_rate)
    received = []
    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        BarrierTranscriber(),
        lambda segment, latency, embedding=None: received.append(segment),
        channels=(Channel.MIC,),
        vad_factory=EnergyFakeVad,
        speaker_resolver=resolver,
    )
    pipeline.start()
    pipeline.frame_queue.put((Channel.MIC, 0.0, np.full(40 * WINDOW, 0.75, dtype=np.float32)))
    pipeline.frame_queue.put(
        (Channel.MIC, 40 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    pipeline.finish()

    assert received[0].speaker == "Alice"
    assert received[0].attribution == "auto"
    assert store.profile(saved.speaker_id).state == "ready"


def test_rolling_speaker_tracking_first_identification_splits(tmp_path):
    """A channel's opening chunk is exactly where two people answering each
    other get merged onto one line — nobody is established yet, and there is
    no silence to split on. The first confirmed identification has to cut."""
    import threading as _threading

    from oat_notes.attribution import Speaker
    from oat_notes.speaker_id import SpeakerEmbeddingEngine, SpeakerResolver
    from oat_notes.speaker_store import SpeakerStore

    class BobEngine(SpeakerEmbeddingEngine):
        def embed(self, samples, sample_rate):
            return np.array([0.0, 1.0], dtype=np.float32)

    store = SpeakerStore(tmp_path / "speakers.db")
    alice = store.create_speaker("Alice")
    bob = store.create_speaker("Bob")
    store.add_sample(alice.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(bob.speaker_id, np.array([0.0, 1.0]), "mic", 5.0, 1.0)
    resolver = SpeakerResolver(
        [
            Speaker("Alice", speaker_id=alice.speaker_id),
            Speaker("Bob", speaker_id=bob.speaker_id),
        ],
        store,
        BobEngine(),
        CONFIG.sample_rate,
    )
    received = []
    tracked = []
    identified = _threading.Event()

    def on_tracking(*event):
        tracked.append(event)
        identified.set()

    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        FakeTranscriber(),
        lambda segment, latency, embedding=None: received.append(segment),
        channels=(Channel.MIC,),
        vad_factory=EnergyFakeVad,
        speaker_resolver=resolver,
        speaker_tracking_sink=on_tracking,
    )
    splits = []
    real_split = pipeline.split_channel

    def spy_split(channel):
        splits.append(channel)
        real_split(channel)

    pipeline.split_channel = spy_split

    pipeline.start()
    pipeline.frame_queue.put(
        (Channel.MIC, 0.0, np.full(90 * WINDOW, 0.75, dtype=np.float32))
    )
    assert identified.wait(timeout=3.0)
    pipeline.frame_queue.put(
        (
            Channel.MIC,
            90 * WINDOW / CONFIG.sample_rate,
            np.full(40 * WINDOW, 0.75, dtype=np.float32),
        )
    )
    pipeline.finish()

    assert splits == [Channel.MIC]
    assert tracked == [(1, Channel.MIC, "auto", 1.0)]
    assert [segment.speaker for segment in received] == ["Bob", "Bob"]
    assert received[1].start == received[0].end


def test_rolling_speaker_tracking_confirmed_switch_splits_transcript(tmp_path):
    """A confirmed switch away from an already-established speaker must cut
    the transcript immediately instead of waiting for silence or the 15s
    force-split cap (previously the only re-check points)."""
    import threading as _threading

    from oat_notes.attribution import Speaker
    from oat_notes.speaker_id import SpeakerEmbeddingEngine, SpeakerResolver
    from oat_notes.speaker_store import SpeakerStore

    class BobEngine(SpeakerEmbeddingEngine):
        def embed(self, samples, sample_rate):
            return np.array([0.0, 1.0], dtype=np.float32)

    store = SpeakerStore(tmp_path / "speakers.db")
    alice = store.create_speaker("Alice")
    bob = store.create_speaker("Bob")
    store.add_sample(alice.speaker_id, np.array([1.0, 0.0]), "mic", 5.0, 1.0)
    store.add_sample(bob.speaker_id, np.array([0.0, 1.0]), "mic", 5.0, 1.0)
    resolver = SpeakerResolver(
        [
            Speaker("Alice", speaker_id=alice.speaker_id),
            Speaker("Bob", speaker_id=bob.speaker_id),
        ],
        store,
        BobEngine(),
        CONFIG.sample_rate,
    )
    received = []
    tracked = []
    confirmed_switch = _threading.Event()

    def on_tracking(*event):
        tracked.append(event)
        confirmed_switch.set()

    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        FakeTranscriber(),
        lambda segment, latency, embedding=None: received.append(segment),
        channels=(Channel.MIC,),
        vad_factory=EnergyFakeVad,
        speaker_resolver=resolver,
        speaker_tracking_sink=on_tracking,
    )
    pipeline.start()
    pipeline.manual_override(Channel.MIC, 0)

    def unclipped_speech(windows):
        return np.full(windows * WINDOW, 0.75, dtype=np.float32)

    elapsed = 0.0

    def push(windows, block):
        nonlocal elapsed
        pipeline.frame_queue.put((Channel.MIC, elapsed, block))
        elapsed += windows * WINDOW / CONFIG.sample_rate

    push(100, unclipped_speech(100))
    assert confirmed_switch.wait(timeout=3.0)
    push(40, unclipped_speech(40))
    push(SILENCE_WINDOWS, silence(SILENCE_WINDOWS))
    pipeline.finish()

    assert [segment.speaker for segment in received] == ["Alice", "Bob"]
    first, second = received
    assert second.start == first.end
    assert first.attribution == "manual"
    assert second.attribution == "auto"
    assert tracked == [(1, Channel.MIC, "auto", 1.0)]


def test_transcription_errors_never_log_backend_message(caplog):
    class PrivateFailureTranscriber(Transcriber):
        def transcribe(self, chunk):
            raise RuntimeError("captured words must stay private")

    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        PrivateFailureTranscriber(),
        lambda segment, latency, embedding=None: None,
        channels=(Channel.MIC,),
        vad_factory=EnergyFakeVad,
    )
    pipeline.start()
    pipeline.frame_queue.put((Channel.MIC, 0.0, speech(31)))
    pipeline.frame_queue.put(
        (Channel.MIC, 31 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    pipeline.finish()

    assert "RuntimeError" in caplog.text
    assert "captured words must stay private" not in caplog.text


def _learning_pipeline(tmp_path, channels=(Channel.MIC,)):
    from oat_notes.attribution import Speaker
    from oat_notes.speaker_id import SpeakerEmbeddingEngine, SpeakerResolver
    from oat_notes.speaker_store import SpeakerStore

    class StableEngine(SpeakerEmbeddingEngine):
        def embed(self, samples, sample_rate):
            return np.array([1.0, 0.0], dtype=np.float32)

    store = SpeakerStore(tmp_path / "speakers.db")
    alice = store.create_speaker("Alice")
    bob = store.create_speaker("Bob")
    resolver = SpeakerResolver(
        [
            Speaker("Alice", speaker_id=alice.speaker_id),
            Speaker("Bob", speaker_id=bob.speaker_id),
        ],
        store,
        StableEngine(),
        CONFIG.sample_rate,
    )
    updates = []
    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        FakeTranscriber(),
        lambda segment, latency, embedding=None: None,
        channels=channels,
        vad_factory=EnergyFakeVad,
        speaker_resolver=resolver,
        speaker_tracking_sink=lambda *event: None,
        profile_learning_sink=updates.append,
    )
    return pipeline, store, alice, bob, updates


def _clean_speech(windows):
    return np.full(windows * WINDOW, 0.75, dtype=np.float32)


def test_manual_profile_learning_saves_one_three_second_source_sample(tmp_path):
    pipeline, store, alice, _, updates = _learning_pipeline(
        tmp_path, (Channel.MIC, Channel.LOOPBACK)
    )
    pipeline.start()
    pipeline.manual_override(Channel.MIC, 0)
    pipeline.manual_override(Channel.LOOPBACK, 0)
    pipeline.begin_profile_learning(0, 0.0)
    pipeline.frame_queue.put((Channel.MIC, 0.0, _clean_speech(100)))
    pipeline.frame_queue.put((Channel.LOOPBACK, 0.0, silence(100)))
    pipeline.finish()

    assert store.profile(alice.speaker_id).enrollment_seconds == 3.0
    assert store.profile_vector(alice.speaker_id, "mic").source_specific is True
    assert store.profile_vector(alice.speaker_id, "loopback").source_specific is False
    assert [event.phase for event in updates][-1] == "saved"
    assert [event.phase for event in updates].count("saved") == 1


def test_new_manual_selection_cancels_the_previous_profile_candidate(tmp_path):
    pipeline, store, alice, bob, updates = _learning_pipeline(tmp_path)
    pipeline.start()
    pipeline.manual_override(Channel.MIC, 0)
    pipeline.begin_profile_learning(0, 0.0)
    pipeline.frame_queue.put((Channel.MIC, 0.0, _clean_speech(40)))
    pipeline.manual_override(Channel.MIC, 1)
    pipeline.begin_profile_learning(1, 1.3)
    pipeline.frame_queue.put((Channel.MIC, 1.3, _clean_speech(100)))
    pipeline.finish()

    assert store.profile(alice.speaker_id).enrollment_seconds == 0.0
    assert store.profile(bob.speaker_id).enrollment_seconds == 3.0
    assert any(
        event.phase == "saved" and event.speaker_index == 1 for event in updates
    )


def test_manual_profile_learning_times_out_without_three_seconds_of_speech(tmp_path):
    pipeline, store, alice, _, updates = _learning_pipeline(tmp_path)
    pipeline.start()
    pipeline.begin_profile_learning(0, 0.0)
    pipeline.frame_queue.put((Channel.MIC, 11.0, silence(1)))
    pipeline.finish()

    assert store.profile(alice.speaker_id).enrollment_seconds == 0.0
    assert any(event.phase == "skipped" and event.reason == "timeout" for event in updates)


def test_profile_learning_skips_an_ambiguous_active_source(tmp_path):
    pipeline, store, alice, _, updates = _learning_pipeline(
        tmp_path, (Channel.MIC, Channel.LOOPBACK)
    )
    pipeline.start()
    pipeline.frame_queue.put((Channel.MIC, 0.0, _clean_speech(1)))
    pipeline.frame_queue.put((Channel.LOOPBACK, 0.0, _clean_speech(1)))
    pipeline.begin_profile_learning(0, 0.03)
    pipeline.finish()

    assert store.profile(alice.speaker_id).enrollment_seconds == 0.0
    assert any(
        event.phase == "skipped" and event.reason == "ambiguous_source"
        for event in updates
    )


def test_cancel_profile_learning_stops_the_in_progress_capture(tmp_path):
    pipeline, store, alice, _, updates = _learning_pipeline(tmp_path)
    pipeline.start()
    pipeline.begin_profile_learning(0, 0.0)
    pipeline.frame_queue.put((Channel.MIC, 0.0, _clean_speech(10)))
    pipeline.cancel_profile_learning()
    pipeline.frame_queue.put((Channel.MIC, 10.0, _clean_speech(100)))
    pipeline.finish()

    assert store.profile(alice.speaker_id).enrollment_seconds == 0.0
    assert any(
        event.phase == "skipped" and event.reason == "cancelled" for event in updates
    )


def test_cancel_profile_learning_without_a_candidate_is_a_no_op(tmp_path):
    pipeline, _, _, _, updates = _learning_pipeline(tmp_path)
    pipeline.start()
    pipeline.cancel_profile_learning()
    pipeline.finish()

    assert updates == []
