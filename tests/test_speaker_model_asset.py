import hashlib
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oat_notes.speaker_id import SherpaOnnxEmbeddingEngine, bundled_model_path


def test_incomplete_windows_runtime_fails_cleanly(monkeypatch):
    import oat_notes.speaker_id as speaker_id

    missing_package = Path(__file__).with_name("missing_sherpa_onnx") / "__init__.py"
    fake_package = SimpleNamespace(__file__=str(missing_package))
    monkeypatch.setitem(__import__("sys").modules, "sherpa_onnx", fake_package)
    monkeypatch.setattr(speaker_id.sys, "platform", "win32")
    try:
        SherpaOnnxEmbeddingEngine()
    except RuntimeError as error:
        assert "native runtime is incomplete" in str(error)
    else:
        raise AssertionError("incomplete native runtime was accepted")


@pytest.mark.slow
def test_bundled_speaker_model_loads_and_embeds_without_network(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("outbound network attempted")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    path = bundled_model_path()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "c59158379255ad66e161679cca6af8d52d51e389e3224ab7d7a7baae295c2db5"
    )

    engine = SherpaOnnxEmbeddingEngine()
    time = np.arange(32_000, dtype=np.float32) / 16_000
    samples = 0.2 * np.sin(2 * np.pi * 220 * time)
    embedding = engine.embed(samples, 16_000)
    assert embedding.shape == (192,)
    assert np.linalg.norm(embedding) == np.float32(1.0)
