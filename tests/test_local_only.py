"""Dictation must never reach the network.

The README states a hard invariant: the only permitted network traffic is the
one-time model download, and ``--offline`` disables even that. These are
tripwires in the style of test_build_environment — cheap to run, and they
fail loudly the moment a well-meaning change adds a remote call.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src" / "oat_notes"

DICTATION_MODULES = [
    PACKAGE / "dictation" / "controller.py",
    PACKAGE / "dictation" / "format.py",
    PACKAGE / "dictation" / "recorder.py",
    PACKAGE / "inject.py",
    PACKAGE / "overlay.py",
    PACKAGE / "hotkeys.py",
    PACKAGE / "llm.py",
]

_NETWORK_MODULES = {
    "requests", "urllib", "urllib3", "httpx", "aiohttp", "socket", "ssl",
    "http", "websocket", "websockets", "ftplib", "smtplib", "telnetlib",
    "openai", "anthropic", "huggingface_hub",
}
_IMPORT = re.compile(r"^\s*(?:import|from)\s+([\w.]+)", re.MULTILINE)


@pytest.mark.parametrize("path", DICTATION_MODULES, ids=lambda p: p.name)
def test_no_network_client_in_the_dictation_path(path):
    imported = {
        name.split(".")[0]
        for name in _IMPORT.findall(path.read_text(encoding="utf-8"))
    }
    hits = sorted(imported & _NETWORK_MODULES)
    assert hits == [], f"{path.name} reaches the network via {hits}"


def test_model_downloads_are_confined_to_one_module():
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        if path.name == "models.py":
            continue
        if "huggingface_hub" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []


def test_downloads_respect_the_offline_flag():
    source = (PACKAGE / "models.py").read_text(encoding="utf-8")
    resolve = source[source.index("def resolve_openvino_model") :]
    assert resolve.index("if offline:") < resolve.index("snapshot_download")


def test_the_llm_engine_is_local_only():
    source = (PACKAGE / "llm.py").read_text(encoding="utf-8")
    assert "openvino_genai" in source
    assert "api_key" not in source and "base_url" not in source


def test_dictation_stays_out_of_clipboard_history():
    source = (PACKAGE / "inject.py").read_text(encoding="utf-8")
    assert "ExcludeClipboardContentFromMonitorProcessing" in source
    assert "CanIncludeInClipboardHistory" in source


def test_captured_words_are_never_logged():
    for path in (PACKAGE / "dictation" / "recorder.py", PACKAGE / "cleanup.py"):
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            if "file=sys.stderr" in line or "error:" in line and "print" in line:
                assert "{error}" not in line or "type(error)" in line
