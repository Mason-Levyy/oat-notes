# oat-notes

Live meeting transcription for Windows ("Muesli" — a Granola analogue). Captures audio via WASAPI, chunks it with Silero VAD at natural pauses, and transcribes with faster-whisper — all local, no cloud.

Live mic + system-audio loopback → timestamped transcript with speaker labels, in the console or a local 8-bit web UI. On a Zoom/Teams call, the remote side is captured from the loopback of your output device — no mic pickup needed. Wear headphones, or the mic will also hear the remote speaker and produce duplicate lines.

## Web UI

```
uv run oat-notes --ui
```

Opens `http://127.0.0.1:8737`: enter in-person speakers and a remote name, START MEETING, and the transcript streams onto the screen live. Speaker chips (P1/P2/…) switch the mic owner with a click — or the global 1..N hotkeys, which stay live and keep the UI in sync. END MEETING writes the `.txt` and shows the path. `--port` and `--no-browser` available.

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
uv run oat-notes --speakers "Mason,Sarah"   # in-person speakers; press 1/2 to switch
uv run oat-notes --remote-name Priya        # name the Zoom/Teams side
uv run oat-notes --out-dir D:\notes  # transcript folder (default: ./transcripts)
uv run oat-notes --no-file           # console only, skip the transcript file
uv run oat-notes --wav clip.m4a      # transcribe a file (any format PyAV decodes)
uv run oat-notes --debug             # show per-chunk transcription latency
uv run oat-notes --model small.en    # swap whisper models
uv run oat-notes --backend openvino  # offload to the Intel NPU (see below)
uv run oat-notes --backend openvino --ov-device GPU   # or the Arc GPU
```

Output, one line per speech chunk:

```
[00:03:12] Me: Let's walk through the model assumptions.
[00:03:28] Remote: The churn number looks high to me.
```

On exit the session is also written to `transcripts/meeting_YYYY-MM-DD_HHMM.txt`, merged across channels and sorted by timestamp — ready to paste into OneNote or Claude for summarization.

With `--speakers "Mason,Sarah"`, number keys 1..N (any app focused — the hook is global) switch who owns the mic. A press also cuts the in-flight chunk on the spot: the previous speaker's words go straight to transcription and the new speaker starts a fresh chunk, so rapid handoffs with no pause between speakers still attribute cleanly. Where no key is pressed, a chunk goes to whoever held the majority of its span. The remote (loopback) side never needs a key. With a single name in `--speakers`, all mic audio is yours with no hotkeys involved.

## Architecture

```
capture.py    WASAPI capture (pyaudiowpatch); callback only stamps + enqueues
clock.py      one monotonic session clock — all timestamps stamped at capture
vad.py        streaming Silero VAD (ONNX model bundled with faster-whisper)
chunker.py    state machine: split at 0.5s silence, force-split at 15s, drop blips
transcriber.py  Transcriber ABC + faster-whisper and OpenVINO backends
pipeline.py   capture → chunker thread → transcription worker → sink, via queues
attribution.py  switch log + majority-overlap speaker attribution
hotkeys.py    global 1..N listener (session-scoped)
session.py    one meeting: capture + pipeline + attribution + transcript log
server.py     stdlib HTTP + SSE serving the web UI (web/index.html)
```

Chunks carry a channel tag (`MIC`/`LOOPBACK`) and segments carry a `speaker` field from day one, so the dual-stream and attribution phases slot in without restructuring.

## OpenVINO backend (NPU/GPU offload)

Install the optional extra, then pick a device:

```
uv sync --extra openvino
uv run oat-notes --backend openvino               # NPU (default)
uv run oat-notes --backend openvino --ov-device GPU
```

Uses a pre-converted `OpenVINO/whisper-small.en-int8-ov` from HuggingFace (no torch/conversion toolchain), stored under `%LOCALAPPDATA%\oat-notes\models`. If the device rejects the model it falls back to OpenVINO-on-CPU. Benchmarks on a Core Ultra (11 s clip, distil-small.en vs whisper-small.en int8):

| Backend | Load | Speed |
|---|---|---|
| faster-whisper CPU int8 | 1.4 s | 9× realtime |
| OpenVINO **NPU** | 3.6 s | 21× realtime, CPU stays free |
| OpenVINO GPU (Arc) | 18 s | 38× realtime |
| OpenVINO CPU | 1.3 s | 18× realtime |

## Packaged exe

```
.\scripts\build_exe.ps1
```

Produces `dist\oat-notes.exe` (PyInstaller onefile, console app). Double-click → web UI opens; all CLI flags work too. Model weights download to the user cache on first run so the exe stays smaller. The OpenVINO backend is excluded from the exe — run from source for NPU transcription.

## Tests

```
uv run pytest
```

Chunker and clock tests run against a fake VAD — no audio hardware needed.
