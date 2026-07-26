"""Locating and fetching local inference models.

Shared by the transcriber and the LLM engine — both pull pre-converted
OpenVINO models from the same cache. The only network traffic the
application ever makes lives here, and ``offline`` turns it off.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_tls_trusted = False


def ensure_tls_trust() -> None:
    """Make HTTPS model downloads trust the OS certificate store so a
    corporate TLS-interception proxy doesn't fail first-run fetches. A no-op
    when already applied or when truststore isn't available."""
    global _tls_trusted
    if _tls_trusted:
        return
    _tls_trusted = True
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as error:  # never let this block model loading
        print(f"could not enable OS certificate trust: {error}", file=sys.stderr)


def _openvino_model_dir(source: str) -> Path:
    return (
        Path(os.environ["LOCALAPPDATA"])
        / "oat-notes"
        / "models"
        / source.replace("/", "--")
    )


def openvino_model_cached(source: str) -> bool:
    """True if the model is a local dir or already downloaded — i.e. loading
    it will not hit the network."""
    if Path(source).is_dir():
        return True
    return (_openvino_model_dir(source) / "config.json").exists()


def resolve_openvino_model(source: str, offline: bool) -> str:
    """Local directory or HF repo id; repos land as plain files under
    %LOCALAPPDATA% because the HF cache's symlinks need Developer Mode."""
    if Path(source).is_dir():
        return source
    target = _openvino_model_dir(source)
    if (target / "config.json").exists():
        return str(target)
    if offline:
        raise RuntimeError(
            f"model {source!r} is not cached locally and --offline forbids"
            " downloading it"
        )
    print(f"downloading {source}…", file=sys.stderr)
    ensure_tls_trust()
    from huggingface_hub import snapshot_download

    return snapshot_download(source, local_dir=target)


def openvino_cache_dir() -> Path:
    """Where OpenVINO drops compiled device blobs. Keeping these means an NPU
    load is seconds rather than minutes on every launch."""
    return Path(os.environ["LOCALAPPDATA"]) / "oat-notes" / "openvino-cache"
