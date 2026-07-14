# oat-notes

Live meeting transcription for Windows ("Muesli" — a Granola analogue). Captures audio via WASAPI, chunks it with Silero VAD at natural pauses, and transcribes with faster-whisper — all local, no cloud.

Live mic + system-audio loopback → timestamped transcript with speaker labels, in the console or a local 8-bit web UI. On a Zoom/Teams call, other participants are captured from the loopback of your output device — no mic pickup needed. Wear headphones to prevent the mic from capturing the same speech and producing duplicate lines.

## Web UI

```
uv run oat-notes --ui
```

Opens `http://127.0.0.1:8737`: name the meeting, build the roster from saved people or groups, START MEETING, and the transcript streams onto the screen live. New people are gray until hotkey-labeled turns provide four seconds of usable enrollment speech. Voice embedding and transcription run locally in parallel on separate model instances; ready profiles are checked in overlapping 1.5-second windows about every 0.25 seconds, with two consecutive matches required before the live speaker changes. Once a change is confirmed, the transcript cuts to the new speaker immediately rather than waiting for a pause or the 15-second chunk cap.

The **Settings** tab holds the local speaker directory and reusable groups. Groups remember order but stay editable per meeting. Someone unexpected joins? Click `+ GUEST`, then link the Guest to an existing profile or create a new saved person during backfill. END MEETING writes the named `.txt` and shows the path. `--port` and `--no-browser` are available.

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
uv run oat-notes --speakers "Mason,Sarah,Priya"   # keys 1..N switch
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
[00:03:12] Microphone: Let's walk through the model assumptions.
[00:03:28] System audio: The churn number looks high to me.
```

On exit the session is written to `transcripts/<name>_YYYY-MM-DD_HHMM.txt` (`--name "stand-up"` → `stand-up_…​.txt`), merged across channels and sorted by timestamp — ready to paste into OneNote or Claude for summarization.

`--speakers` accepts a comma-separated roster. In the UI, the saved roster determines stable hotkey slots. **Ctrl+Alt+1..9** by default selects a speaker in the active bank, while **Ctrl+Alt+[** and **Ctrl+Alt+]** page through larger rosters. The Settings tab can change the modifier combination. A manual press is authoritative for the current VAD turn on either audio source, cuts in-flight audio cleanly, and is the only event allowed to improve a saved voice profile. Automatic low-confidence matches are labeled `Unknown` rather than guessed.

## Architecture

```
capture.py    WASAPI capture (pyaudiowpatch); callback only stamps + enqueues
clock.py      one monotonic session clock — all timestamps stamped at capture
vad.py        streaming Silero VAD (ONNX model bundled with faster-whisper)
chunker.py    state machine: split at 0.5s silence, force-split at 15s, drop blips
transcriber.py  Transcriber ABC + faster-whisper and OpenVINO backends
pipeline.py   capture → chunker thread → transcription worker → sink, via queues
attribution.py  switch log + majority-overlap speaker attribution
speaker_store.py  local SQLite speaker/group directory + embedding centroids
speaker_id.py  bundled sherpa-onnx embeddings, enrollment, roster matching
hotkeys.py    configurable global modifier+1..9 listener (session-scoped)
session.py    one meeting: capture + pipeline + attribution + transcript log
server.py     stdlib HTTP + SSE serving the web UI (web/index.html)
```

Chunks carry a channel tag (`MIC`/`LOOPBACK`) and segments carry a `speaker` field from day one, so the dual-stream and attribution phases slot in without restructuring.

## Local-only by design

Captured audio, transcripts, voice embeddings, identities, groups, and recognition results never leave the machine. Profiles are stored as embeddings only—never replayable audio—under `%LOCALAPPDATA%\oat-notes\speakers.db`, outside the normal OneDrive Documents tree. Speaker inference uses a bundled ONNX model and performs no download. The web UI binds to `127.0.0.1` only and rejects requests whose `Host`/`Origin` headers don't name that address. The only permitted network traffic is the existing one-time transcription-model download; `--offline` disables that too. HF telemetry and implicit-token lookups are disabled by default.

## OpenVINO backend (NPU/GPU offload)

Install the optional extra, then pick a device:

```
uv sync --extra openvino
uv run oat-notes --backend openvino               # NPU (default)
uv run oat-notes --backend openvino --ov-device GPU
```

Uses a pre-converted `OpenVINO/whisper-small.en-int8-ov` from HuggingFace (no torch/conversion toolchain), stored under `%LOCALAPPDATA%\oat-notes\models`. If the device rejects the model it falls back to OpenVINO-on-CPU. Benchmarks on a Core Ultra (11 s clip, distil-small.en vs whisper-small.en int8):

Compiled OpenVINO models are cached under `%LOCALAPPDATA%\oat-notes\openvino-cache`, avoiding a minute-long NPU compilation on later starts. The cache uses about 850 MB on the tested Core Ultra system.

| Backend | Load | Speed |
|---|---|---|
| faster-whisper CPU int8 | 1.4 s | 9× realtime |
| OpenVINO **NPU** | 3.6 s | 21× realtime, CPU stays free |
| OpenVINO GPU (Arc) | 18 s | 38× realtime |
| OpenVINO CPU | 1.3 s | 18× realtime |

## Installed Windows app

```
uv sync --extra openvino
.\scripts\build_app.ps1
```

Produces an unpacked, windowed application under
`%LOCALAPPDATA%\oat-notes-build\dist\oat-notes`. Unlike the old one-file
executable, the installed layout does not decompress 170+ MB into `%TEMP%` on
every launch. Double-clicking `oat-notes.exe` starts the UI first, then loads
and warms OpenVINO in the background. The START button enables when the model
is ready.

For a Start menu shortcut, uninstaller, and optional desktop shortcut, install
[Inno Setup 6](https://jrsoftware.org/isinfo.php) and run:

```
.\scripts\build_installer.ps1
```

The installer is written to `dist\installer\OatNotes-Setup.exe` and installs
per-user under `%LOCALAPPDATA%\Programs\Oat Notes`, so it does not require an
administrator prompt. Installed launches save transcripts under
`Documents\Oat Notes`; windowed-app logs live under
`%LOCALAPPDATA%\oat-notes\logs`. Model weights and compiled OpenVINO caches
remain in `%LOCALAPPDATA%\oat-notes` and survive application upgrades.

The legacy `.\scripts\build_exe.ps1` command remains as an alias for
`build_app.ps1`. All CLI flags still work. Model weights download to the user
cache on first run, keeping the installer smaller.

## Tests

```
uv run pytest
```

Chunker and clock tests run against a fake VAD — no audio hardware needed.
