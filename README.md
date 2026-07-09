# oat-notes

Live meeting transcription for Windows ("Muesli" — a Granola analogue). Captures audio via WASAPI, chunks it with Silero VAD at natural pauses, and transcribes with faster-whisper — all local, no cloud.

**Current state:** live mic + system-audio loopback → timestamped console transcript with `Me:` / `Remote:` labels. On a Zoom/Teams call, the remote side is captured from the loopback of your output device — no mic pickup needed. Wear headphones, or the mic will also hear the remote speaker and produce duplicate lines. Later phases add hotkey speaker attribution for in-person meetings, an end-of-meeting `.txt` writer, an OpenVINO/NPU backend, and a UI + packaging.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is pinned and auto-installed):

```
uv sync
```

The first run downloads `distil-small.en` (a few hundred MB) to the HuggingFace cache.

## Usage

```
uv run oat-notes                     # mic + system audio live; Ctrl+C to stop
uv run oat-notes --no-loopback       # mic only
uv run oat-notes --list-devices      # enumerate input devices (loopback endpoints marked)
uv run oat-notes --device-index 5    # pick a specific mic
uv run oat-notes --loopback-index 10 # pick a specific loopback endpoint
uv run oat-notes --seconds 30        # auto-stop (handy for testing)
uv run oat-notes --wav clip.m4a      # transcribe a file (any format PyAV decodes)
uv run oat-notes --debug             # show per-chunk transcription latency
uv run oat-notes --model small.en    # swap whisper models
```

Output, one line per speech chunk:

```
[00:03:12] Me: Let's walk through the model assumptions.
[00:03:28] Remote: The churn number looks high to me.
```

## Architecture

```
capture.py    WASAPI capture (pyaudiowpatch); callback only stamps + enqueues
clock.py      one monotonic session clock — all timestamps stamped at capture
vad.py        streaming Silero VAD (ONNX model bundled with faster-whisper)
chunker.py    state machine: split at 0.5s silence, force-split at 15s, drop blips
transcriber.py  Transcriber ABC + faster-whisper backend (OpenVINO slot reserved)
pipeline.py   capture → chunker thread → transcription worker → sink, via queues
```

Chunks carry a channel tag (`MIC`/`LOOPBACK`) and segments carry a `speaker` field from day one, so the dual-stream and attribution phases slot in without restructuring.

## Tests

```
uv run pytest
```

Chunker and clock tests run against a fake VAD — no audio hardware needed.
