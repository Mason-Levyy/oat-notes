import sys
import types

import numpy as np

from oat_notes.config import Config
from oat_notes.transcriber import (
    NO_HINTS,
    FasterWhisperTranscriber,
    TranscriptionHints,
    _openvino_hint_kwargs,
)
from oat_notes.types import AudioChunk, Channel


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def transcribe(self, samples, **options):
        self.calls.append(options)
        return iter([types.SimpleNamespace(text=" hello ")]), None


def chunk():
    return AudioChunk(np.zeros(1600, dtype=np.float32), Channel.MIC, 0.0, 0.1)


def test_faster_whisper_passes_the_hints_through(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    )
    transcriber = FasterWhisperTranscriber(Config())
    hints = TranscriptionHints(prompt="so far", hotwords="Mason")
    assert transcriber.transcribe(chunk(), hints).text == "hello"
    assert transcriber.transcribe(chunk()).text == "hello"
    with_hints, without = transcriber._model.calls
    assert (with_hints["initial_prompt"], with_hints["hotwords"]) == ("so far", "Mason")
    assert (without["initial_prompt"], without["hotwords"]) == (None, None)


def test_openvino_only_receives_the_hints_that_are_set():
    assert _openvino_hint_kwargs(NO_HINTS) == {}
    assert _openvino_hint_kwargs(TranscriptionHints(prompt="so far")) == {
        "initial_prompt": "so far"
    }
    assert _openvino_hint_kwargs(TranscriptionHints(hotwords="Mason")) == {
        "hotwords": "Mason"
    }
