from oat_notes.transcriber import ensure_tls_trust, openvino_model_cached


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
    # Safe to call repeatedly; a missing/broken truststore must not blow up.
    ensure_tls_trust()
    ensure_tls_trust()
