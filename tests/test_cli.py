"""The ``--wav`` path end to end: decoded audio in, a transcript file out."""

import argparse
import functools
import math

import numpy as np

from oat_notes import cli, pipeline
from oat_notes.config import Config
from oat_notes.transcriber import NO_HINTS, Transcriber
from oat_notes.types import TranscriptSegment

CONFIG = Config()
WINDOW = CONFIG.vad_window_samples
SILENCE_WINDOWS = math.ceil(CONFIG.silence_split_seconds * CONFIG.sample_rate / WINDOW)


class EnergyFakeVad:
    window_samples = WINDOW

    def speech_probability(self, window):
        return 1.0 if float(np.abs(window).max()) > 0.5 else 0.0


class FakeTranscriber(Transcriber):
    def transcribe(self, chunk, hints=NO_HINTS):
        return TranscriptSegment(
            text="hello from the file",
            channel=chunk.channel,
            start=chunk.start,
            end=chunk.end,
        )


def test_wav_transcription_writes_a_transcript(tmp_path, monkeypatch, capsys):
    speech = np.ones(WINDOW * 40, dtype=np.float32)
    silence = np.zeros(WINDOW * (SILENCE_WINDOWS + 4), dtype=np.float32)
    monkeypatch.setattr(
        "faster_whisper.audio.decode_audio",
        lambda path, sampling_rate: np.concatenate([speech, silence]),
    )
    monkeypatch.setattr(
        pipeline, "Pipeline", functools.partial(pipeline.Pipeline, vad_factory=EnergyFakeVad)
    )
    args = argparse.Namespace(wav="meeting.wav", out_dir=tmp_path, no_file=False, name="meeting")

    cli._run_file(args, CONFIG, FakeTranscriber())

    transcript = next(tmp_path.glob("*.txt")).read_text(encoding="utf-8")
    assert "hello from the file" in transcript
    assert "hello from the file" in capsys.readouterr().out
