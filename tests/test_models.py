import pytest

from oat_notes.models import (
    ensure_tls_trust,
    load_on_device_or_fall_back,
    openvino_model_cached,
)


def test_openvino_model_cached_for_a_local_directory(tmp_path):
    assert openvino_model_cached(str(tmp_path)) is True


def test_openvino_model_cached_tracks_the_download_target(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert openvino_model_cached("Org/Some-Model") is False

    target = tmp_path / "oat-notes" / "models" / "Org--Some-Model"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    assert openvino_model_cached("Org/Some-Model") is True


def test_ensure_tls_trust_is_idempotent_and_never_raises():
    ensure_tls_trust()
    ensure_tls_trust()


def test_a_device_that_takes_the_model_is_used_as_is():
    pipeline, device = load_on_device_or_fall_back(
        lambda chosen: f"pipeline-on-{chosen}", "NPU", "test model"
    )
    assert (pipeline, device) == ("pipeline-on-NPU", "NPU")


def test_a_refusing_device_falls_back_to_cpu(caplog):
    attempts = []

    def build(chosen: str) -> str:
        attempts.append(chosen)
        if chosen != "CPU":
            raise RuntimeError("unsupported op")
        return "pipeline-on-CPU"

    pipeline, device = load_on_device_or_fall_back(build, "NPU", "test model")
    assert (pipeline, device) == ("pipeline-on-CPU", "CPU")
    assert attempts == ["NPU", "CPU"]
    assert "NPU rejected the test model" in caplog.text


def test_cpu_refusing_the_model_is_not_survivable():
    def build(_chosen: str) -> str:
        raise RuntimeError("no room")

    with pytest.raises(RuntimeError, match="no room"):
        load_on_device_or_fall_back(build, "CPU", "test model")
