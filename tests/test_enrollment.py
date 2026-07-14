"""Tests for the standalone voice-enrollment recorder — no real audio
hardware or ONNX model involved; frame_queue is driven directly, the same
seam tests use for Pipeline."""

import threading

import numpy as np

from oat_notes.config import Config
from oat_notes.enrollment import VoiceEnrollmentRecorder
from oat_notes.speaker_id import SpeakerEmbeddingEngine
from oat_notes.speaker_store import SpeakerStore
from oat_notes.types import Channel

CONFIG = Config()
WINDOW = CONFIG.vad_window_samples
SILENCE_WINDOWS = 16


class EnergyFakeVad:
    """Speech wherever the samples are loud — lets tests shape audio directly."""

    window_samples = WINDOW

    def speech_probability(self, window):
        return 1.0 if float(np.abs(window).max()) > 0.5 else 0.0


class ConstantEngine(SpeakerEmbeddingEngine):
    def embed(self, samples, sample_rate):
        return np.array([1.0, 0.0], dtype=np.float32)


def speech(windows, amplitude=0.75):
    return np.full(windows * WINDOW, amplitude, dtype=np.float32)


def silence(windows):
    return np.zeros(windows * WINDOW, dtype=np.float32)


def make_recorder(store, speaker_id, on_progress, target_seconds, engine=None):
    return VoiceEnrollmentRecorder(
        store,
        engine or ConstantEngine(),
        speaker_id,
        CONFIG,
        on_progress,
        target_seconds=target_seconds,
        vad_factory=EnergyFakeVad,
    )


def run_without_hardware(recorder):
    """Drive the recorder's background loop without opening real audio
    (start() is the only place real PyAudio/AudioCapture get touched)."""
    thread = threading.Thread(target=recorder._run, daemon=True)
    thread.start()
    return thread


def test_reaches_target_and_reports_done(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    alice = store.create_speaker("Alice")
    events = []
    recorder = make_recorder(store, alice.speaker_id, events.append, target_seconds=1.0)
    thread = run_without_hardware(recorder)

    recorder.frame_queue.put((Channel.MIC, 0.0, speech(50)))
    recorder.frame_queue.put(
        (Channel.MIC, 50 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    phases = [event.phase for event in events]
    assert phases[0] == "listening"
    assert phases[-1] == "done"
    assert events[-1].captured_seconds >= 1.0
    assert events[-1].profile is not None
    assert store.profile(alice.speaker_id).state in ("learning", "ready")


def test_multiple_turns_accumulate_toward_target(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    bob = store.create_speaker("Bob")
    events = []
    recorder = make_recorder(store, bob.speaker_id, events.append, target_seconds=3.0)
    thread = run_without_hardware(recorder)

    recorder.frame_queue.put((Channel.MIC, 0.0, speech(50)))
    recorder.frame_queue.put(
        (Channel.MIC, 50 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    turn2_start = (50 + SILENCE_WINDOWS) * WINDOW / CONFIG.sample_rate
    recorder.frame_queue.put((Channel.MIC, turn2_start, speech(50)))
    turn2_silence_start = turn2_start + 50 * WINDOW / CONFIG.sample_rate
    recorder.frame_queue.put(
        (Channel.MIC, turn2_silence_start, silence(SILENCE_WINDOWS))
    )
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    captured_events = [event for event in events if event.phase == "captured"]
    assert len(captured_events) == 2
    assert captured_events[1].captured_seconds > captured_events[0].captured_seconds
    assert events[-1].phase == "done"


def test_clipped_audio_is_rejected_and_never_reaches_target(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    carol = store.create_speaker("Carol")
    events = []
    recorder = make_recorder(store, carol.speaker_id, events.append, target_seconds=1.0)
    thread = run_without_hardware(recorder)

    recorder.frame_queue.put((Channel.MIC, 0.0, speech(50, amplitude=1.0)))  # clipped
    recorder.frame_queue.put(
        (Channel.MIC, 50 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    recorder.frame_queue.put(None)
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    phases = [event.phase for event in events]
    assert "rejected" in phases
    assert "captured" not in phases
    assert events[-1].phase == "stopped"
    assert store.profile(carol.speaker_id).state == "untrained"


def test_stop_before_target_still_saves_partial_progress(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    dee = store.create_speaker("Dee")
    events = []
    recorder = make_recorder(store, dee.speaker_id, events.append, target_seconds=10.0)
    thread = run_without_hardware(recorder)

    recorder.frame_queue.put((Channel.MIC, 0.0, speech(50)))
    recorder.frame_queue.put(
        (Channel.MIC, 50 * WINDOW / CONFIG.sample_rate, silence(SILENCE_WINDOWS))
    )
    recorder.frame_queue.put(None)
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert events[-1].phase == "done"
    assert 0 < events[-1].captured_seconds < 10.0
    assert store.profile(dee.speaker_id).state == "learning"


def test_stop_with_no_speech_reports_stopped_not_done(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    erin = store.create_speaker("Erin")
    events = []
    recorder = make_recorder(store, erin.speaker_id, events.append, target_seconds=10.0)
    thread = run_without_hardware(recorder)

    recorder.frame_queue.put(None)
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert events[-1].phase == "stopped"
    assert events[-1].captured_seconds == 0.0
    assert events[-1].profile is None
