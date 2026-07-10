# oat-notes

Live meeting transcription for Windows ("Muesli" — a Granola analogue). Captures audio via WASAPI, chunks it with Silero VAD at natural pauses, and transcribes with faster-whisper — all local, no cloud.

Live mic + system-audio loopback → timestamped transcript with speaker labels, in the console or a local 8-bit web UI. On a Zoom/Teams call, the remote side is captured from the loopback of your output device — no mic pickup needed. Wear headphones, or the mic will also hear the remote speaker and produce duplicate lines.

## Web UI

```
uv run oat-notes --ui
```

Opens `http://127.0.0.1:8737`: name the meeting, list everyone in one row (`Mason, Sarah, Priya*, Dev*` — `*` marks who's on the call), START MEETING, and the transcript streams onto the screen live. Chips or the global 1..N hotkeys switch the active speaker; a press routes to the speaker's own channel, so in-person names attribute the mic and `*` names attribute the call audio, independently.

Someone unexpected joins? Click the dashed `+ WALK-IN` (in the room) or `+ CALL-IN` (on the call) chip — it creates `Guest 1`, switches to them instantly, and another empty slot appears for the next surprise. When you END MEETING with guests in the roster, a backfill panel asks who they were before writing the file; every line they own gets the real name. END MEETING writes the named `.txt` and shows the path. `--port` and `--no-browser` available.

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
uv run oat-notes --speakers "Mason,Sarah,Priya*"  # * = remote; keys 1..N switch
uv run oat-notes --name "stand-up"          # names the transcript file
uv run oat-notes --out-dir D:\notes  # transcript folder (default: ./transcripts)
uv run oat-notes --no-file           # console only, skip the transcript file
uv run oat-notes --wav clip.m4a      # transcribe a file (any format PyAV decodes)
uv run oat-notes --debug             # show per-chunk transcription latency
uv run oat-notes --model small.en    # swap whisper models
uv run oat-notes --backend openvino  # offload to the Intel NPU (see below)
uv run oat-notes --backend openvino --ov-device GPU   # or the Arc GPU
uv run oat-notes --offline           # never touch the network; fails if the model isn't cached yet
```

Output, one line per speech chunk:

```
[00:03:12] Me: Let's walk through the model assumptions.
[00:03:28] Remote: The churn number looks high to me.
```

On exit the session is written to `transcripts/<name>_YYYY-MM-DD_HHMM.txt` (`--name "stand-up"` → `stand-up_…​.txt`), merged across channels and sorted by timestamp — ready to paste into OneNote or Claude for summarization.

`--speakers` takes everyone in one list; a `*` suffix marks remote people on the call. **Ctrl+Alt+1..9** (any app focused — the hook is global, and plain digits still type normally everywhere) switches the active speaker, and the press routes to that speaker's own channel: in-person names attribute mic audio, `*` names attribute call audio, each channel tracking its own active speaker. A chord for a number with no speaker yet auto-adds an in-person `Guest N` and switches to them — name them in the backfill panel afterwards; guests who never spoke are dropped silently. A press also cuts the in-flight chunk on the spot, so rapid handoffs with no pause still attribute cleanly; otherwise a chunk goes to whoever held the majority of its span. A channel with a single listed speaker never needs a key (one remote person = fully automatic).

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

## Local-only by design

Audio, transcription, and the transcript file never leave the machine. The web UI binds to `127.0.0.1` only and rejects requests whose `Host`/`Origin` headers don't name that address (blocks DNS-rebinding and cross-site POSTs from a browser tab). The only network traffic oat-notes ever makes is the one-time HuggingFace model download — cached after that, and `--offline` disables it entirely, failing immediately if a model isn't already on disk. HF telemetry and implicit-token lookups are disabled by default.

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
