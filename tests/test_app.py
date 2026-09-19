import sys

import oat_notes.app as app


def test_windowed_logging_never_persists_stdout_transcripts(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "log_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    app._configure_windowed_logging()
    stdout_sink, log_file = sys.stdout, sys.stderr
    print("private captured transcript", file=stdout_sink, flush=True)
    print("sanitized diagnostic", file=log_file, flush=True)

    log_path = next(tmp_path.glob("oat-notes_*.log"))
    logged = log_path.read_text(encoding="utf-8")
    assert "sanitized diagnostic" in logged
    assert "private captured transcript" not in logged
    stdout_sink.close()
    log_file.close()


def test_bare_launch_runs_the_ui_on_the_npu():
    argv = app.default_argv(["oat-notes.exe"])
    assert "--ui" in argv and "--backend" in argv and "openvino" in argv


def test_startup_shortcut_still_gets_the_ui_defaults():
    argv = app.default_argv(["oat-notes.exe", "--background"])
    assert argv[1] == "--background"
    assert "--ui" in argv and "openvino" in argv


def test_launch_flags_combine():
    argv = app.default_argv(["oat-notes.exe", "--background", "--no-overlay"])
    assert "--ui" in argv


def test_explicit_commands_are_left_alone():
    for command in (["--list-devices"], ["--wav", "a.mp3"], ["--ui", "--port", "9000"]):
        argv = ["oat-notes.exe", *command]
        assert app.default_argv(argv) == argv
