import sys

import oat_notes.app as app


def test_windowed_logging_never_persists_stdout_transcripts(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "log_dir", lambda: tmp_path)
    monkeypatch.setattr(app, "_log_file", None)
    monkeypatch.setattr(app, "_stdout_sink", None)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    app._configure_windowed_logging()
    print("private captured transcript", file=sys.stdout, flush=True)
    print("sanitized diagnostic", file=sys.stderr, flush=True)

    log_path = next(tmp_path.glob("oat-notes_*.log"))
    logged = log_path.read_text(encoding="utf-8")
    assert "sanitized diagnostic" in logged
    assert "private captured transcript" not in logged
    app._stdout_sink.close()
    app._log_file.close()
