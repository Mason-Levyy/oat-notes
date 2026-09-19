"""Locating and fetching local inference models.

Shared by the transcriber and the LLM engine — both pull pre-converted
OpenVINO models from the same cache. The only network traffic the
application ever makes lives here, and ``offline`` turns it off.
"""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path


@cache
def ensure_tls_trust() -> None:
    """Make HTTPS model downloads trust the OS certificate store so a
    corporate TLS-interception proxy doesn't fail first-run fetches. A no-op
    when already applied or when truststore isn't available."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as error:
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


def load_on_device_or_fall_back(build, device: str, what: str):
    """Build a pipeline on ``device``, dropping to CPU if it refuses.

    An NPU or GPU can reject a model for reasons that only surface at load
    time — an unsupported op, a driver that is too old, another process
    already holding the device. CPU always works, so a refusal costs speed
    rather than the feature. Returns the pipeline and the device that took it.
    """
    try:
        return build(device), device
    except Exception as error:
        if device == "CPU":
            raise
        print(
            f"warning: {device} rejected the {what} ({error});"
            " falling back to CPU",
            file=sys.stderr,
        )
        return build("CPU"), "CPU"


def openvino_cache_dir() -> Path:
    """Where OpenVINO drops compiled device blobs. Keeping these means an NPU
    load is seconds rather than minutes on every launch."""
    return Path(os.environ["LOCALAPPDATA"]) / "oat-notes" / "openvino-cache"
