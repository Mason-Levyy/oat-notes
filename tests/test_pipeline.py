"""Pipeline threading and multi-channel routing, with no real audio or model."""

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
