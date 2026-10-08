import sys
import types
from pathlib import Path

import numpy as np
import pytest

from oat_notes import transcriber as transcriber_module
from oat_notes.config import Config
from oat_notes.transcriber import (
    NO_HINTS,
    FasterWhisperTranscriber,
    OpenVinoTranscriber,
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


@pytest.mark.parametrize("device", ["NPU", "NPU.0", "CPU", "GPU"])
def test_openvino_hints_match_the_loaded_device(monkeypatch, device):
    class FakePipeline:
        def __init__(self, model_dir, **options):
            self.calls = []

        def generate(self, samples, **options):
            if device.startswith("NPU") and options:
                raise RuntimeError("'initial_prompt' parameter is not supported on NPU device.")
            self.calls.append(options)
            return types.SimpleNamespace(texts=["The next sentence."])

    monkeypatch.setitem(
        sys.modules, "openvino_genai", types.SimpleNamespace(WhisperPipeline=FakePipeline)
    )
    monkeypatch.setattr(transcriber_module, "resolve_openvino_model", lambda *_: "model")
    monkeypatch.setattr(transcriber_module, "openvino_cache_dir", lambda: Path("model-cache"))
    # The requested NPU may have fallen back during loading. Use the device
    # that actually loaded, so CPU fallback still receives useful hints.
    monkeypatch.setattr(
        transcriber_module, "load_on_device_or_fall_back",
        lambda build, *_: (build(device), device),
    )
    transcriber = OpenVinoTranscriber(Config(backend="openvino", openvino_device="NPU"))
    hints = TranscriptionHints(prompt="The first sentence.", hotwords="Mason")
    assert transcriber.transcribe(chunk(), hints).text == "The next sentence."
    assert transcriber._pipeline.calls == [
        {} if device.startswith("NPU") else {
            "initial_prompt": "The first sentence.", "hotwords": "Mason"
        }
    ]
