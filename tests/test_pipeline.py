"""Pipeline threading and multi-channel routing, with no real audio or model."""

import threading

import numpy as np

from oat_notes.clock import SessionClock
from oat_notes.config import Config
from oat_notes.pipeline import Pipeline
from oat_notes.transcriber import Transcriber
from oat_notes.types import Channel, TranscriptSegment

CONFIG = Config()
WINDOW = CONFIG.vad_window_samples
SILENCE_WINDOWS = 16


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
        lambda segment, latency: received.append(segment),
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
    assert 4.5 < loopback_segment.start <= 5.0  # pre-roll may pull it slightly early


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
        lambda segment, latency: received.append(segment),
        channels=(Channel.MIC, Channel.LOOPBACK),
        vad_factory=EnergyFakeVad,
        attributor=Attributor(
            parse_speakers("Mason, Sarah, Priya"),
            mic_log=log,
            loopback_log=SwitchLog(initial=2),
        ),
    )
    pipeline.start()
    # Mic chunk spans ~0..1.5s (incl. trailing silence); Sarah only from 1.2s
    # => minority overlap, Mason wins.
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
        lambda segment, latency: received.append(segment),
        channels=(Channel.MIC,),
        vad_factory=EnergyFakeVad,
        attributor=Attributor(
            parse_speakers("Alice, Bob"), mic_log=log, loopback_log=SwitchLog()
        ),
    )
    pipeline.start()

    alice_end = 31 * WINDOW / CONFIG.sample_rate  # ~0.992s
    pipeline.frame_queue.put((Channel.MIC, 0.0, speech(31)))
    log.record(alice_end, 1)  # hotkey pressed as Bob starts
    pipeline.split_channel(Channel.MIC)
    pipeline.frame_queue.put((Channel.MIC, alice_end, speech(31)))
    pipeline.frame_queue.put(
        (Channel.MIC, 2 * alice_end, silence(SILENCE_WINDOWS))
    )
    pipeline.finish()

    assert [segment.speaker for segment in received] == ["Alice", "Bob"]
    first, second = received
    assert second.start == first.end  # no audio lost at the boundary


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
    roster = [Speaker("Alice", speaker_id=saved.speaker_id)]
    resolver = SpeakerResolver(roster, store, BarrierEngine(), CONFIG.sample_rate)
    received = []
    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        BarrierTranscriber(),
        lambda segment, latency: received.append(segment),
        channels=(Channel.MIC,),
        vad_factory=EnergyFakeVad,
        speaker_resolver=resolver,
    )
    pipeline.start()
    pipeline.manual_override(Channel.MIC, 0)
    pipeline.frame_queue.put((Channel.MIC, 0.0, np.full(40 * WINDOW, 0.75, dtype=np.float32)))
    pipeline.frame_queue.put(
        (Channel.MIC, 40 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    pipeline.finish()

    assert received[0].speaker == "Alice"
    assert received[0].attribution == "manual"
    assert store.profile(saved.speaker_id).state == "learning"


def test_rolling_speaker_tracking_updates_without_splitting_transcript(tmp_path):
    from oat_notes.attribution import Speaker
    from oat_notes.speaker_id import SpeakerEmbeddingEngine, SpeakerResolver
    from oat_notes.speaker_store import SpeakerStore

    class BobEngine(SpeakerEmbeddingEngine):
        def embed(self, samples, sample_rate):
            return np.array([0.0, 1.0], dtype=np.float32)

    store = SpeakerStore(tmp_path / "speakers.db")
    alice = store.create_speaker("Alice")
    bob = store.create_speaker("Bob")
    store.add_sample(alice.speaker_id, np.array([1.0, 0.0]), "mic", 4.0, 1.0)
    store.add_sample(bob.speaker_id, np.array([0.0, 1.0]), "mic", 4.0, 1.0)
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
    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        FakeTranscriber(),
        lambda segment, latency: received.append(segment),
        channels=(Channel.MIC,),
        vad_factory=EnergyFakeVad,
        speaker_resolver=resolver,
        speaker_tracking_sink=lambda *event: tracked.append(event),
    )
    pipeline.start()
    duration_windows = 90
    pipeline.frame_queue.put(
        (
            Channel.MIC,
            0.0,
            np.full(duration_windows * WINDOW, 0.75, dtype=np.float32),
        )
    )
    pipeline.finish()

    assert len(received) == 1
    assert tracked == [(1, Channel.MIC, "auto", 1.0)]


def test_transcription_errors_never_log_backend_message(capsys):
    class PrivateFailureTranscriber(Transcriber):
        def transcribe(self, chunk):
            raise RuntimeError("captured words must stay private")

    pipeline = Pipeline(
        CONFIG,
        SessionClock(),
        PrivateFailureTranscriber(),
        lambda segment, latency: None,
        channels=(Channel.MIC,),
        vad_factory=EnergyFakeVad,
    )
    pipeline.start()
    pipeline.frame_queue.put((Channel.MIC, 0.0, speech(31)))
    pipeline.frame_queue.put(
        (Channel.MIC, 31 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    pipeline.finish()

    error_log = capsys.readouterr().err
    assert "RuntimeError" in error_log
    assert "captured words must stay private" not in error_log
