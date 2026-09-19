# oat-notes

Live meeting transcription for Windows ("Muesli" — a Granola analogue). Captures audio via WASAPI, chunks it with Silero VAD at natural pauses, and transcribes with faster-whisper — all local, no cloud.

It also does **dictation**: hold Ctrl+Win anywhere in Windows, speak, and the cleaned text is inserted into whatever has focus — with automatic email formatting. See [Dictation](#dictation).

Live mic + system-audio loopback → timestamped transcript with speaker labels, in the console or a local 8-bit web UI. On a Zoom/Teams call, other participants are captured from the loopback of your output device — no mic pickup needed. Wear headphones to prevent the mic from capturing the same speech and producing duplicate lines.

## First-time setup

This guide is for a development checkout on Windows 10 or 11. You need
[Git](https://git-scm.com/downloads/win) to clone the project and
[uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the
runtime. You do not need to install Python separately: the project pins
Python 3.12 and uv installs it when needed.

1. Clone the project and enter its folder:

   ```powershell
   git clone https://github.com/Mason-Levyy/oat-notes.git
   cd oat-notes
   ```

2. Install the pinned Python version and application dependencies:

   ```powershell
   uv sync
   ```

   To use Intel OpenVINO acceleration (NPU or GPU), install that optional
   extra instead:

   ```powershell
   uv sync --extra openvino
   ```

3. Start the local meeting UI:

   ```powershell
   uv run oat-notes --ui
   ```

   Open `http://127.0.0.1:8737` if the browser does not open automatically.
   The first start downloads the transcription model (a few hundred MB), so
   wait for the model to report ready before starting a meeting.

4. In the UI, name the meeting, add people or a saved group to the roster, and
   select **START MEETING**. Use a headset during calls: your microphone is
   recorded together with system-audio loopback, and headphones prevent the
   microphone from duplicating other participants' speech.

5. Select a speaker by clicking their roster row or using **Ctrl+Alt+1** through
   **Ctrl+Alt+9**. End the meeting to write the transcript. Development runs
   save to `./transcripts` by default; the installed app saves to
   `Documents\Oat Notes`.

### Choose audio devices

The default command listens to the default microphone and default output-device
loopback. To inspect or override those choices, use:

```powershell
uv run oat-notes --list-devices
uv run oat-notes --device-index 5 --loopback-index 10
uv run oat-notes --no-loopback
```

`--no-loopback` records only the microphone. This is useful for in-person
meetings or when a virtual audio device is already mixing all sources.

## Web UI

```
uv run oat-notes --ui
```

Opens `http://127.0.0.1:8737`: name the meeting, build the roster from saved people or groups, START MEETING, and the transcript streams onto the screen live. New people are gray until they have five seconds of usable enrollment speech. A manual speaker press starts a one-time, source-specific profile sample: it waits for three seconds of detected speech within ten seconds, then saves one quality-checked embedding. Voice embedding and transcription run locally in parallel on separate model instances; ready profiles are checked in overlapping 1-second windows about every 0.15 seconds, with two consecutive matches required before the live speaker changes. Once a match is confirmed the transcript cuts to that speaker immediately, rather than waiting for a pause or the 15-second chunk cap — including the first identification on a channel, since people answering each other leave no pause to split on.

The **Settings** tab holds the transcription model, the local speaker directory
and reusable groups. Transcription defaults to `small.en`; `distil-small.en` is
faster and `medium.en` copes better with accents, and the choice applies to the
next meeting. `--model` overrides it for one run. Groups remember order but stay editable per meeting. Someone unexpected joins? Click `+ GUEST`, then link the Guest to an existing profile or create a new saved person during backfill. END MEETING writes the named `.txt` and shows the path. `--port` and `--no-browser` are available.

## Dictation

Hold **Ctrl+Win** anywhere in Windows, speak, release — the cleaned text is
inserted wherever your cursor is. A small pill above the taskbar shows the
level and what stage it's at; it never takes focus, so the text always lands in
the window you were typing in.

- **Hold** to talk, or **tap** to latch recording on until you tap again.
  **Esc** cancels without inserting. `--activation` equivalents live in Settings.
- **Cleanup is deterministic and instant** by default — filler words, stutters,
  spoken `new line`/`new paragraph`, and your own vocabulary substitutions.
  Whisper already punctuates, so no model runs on the common path. Speech is
  transcribed in fragments while you talk; each fragment hears the words
  before it, and the length of the pause between them decides whether a
  boundary is a breath or a full stop, so a sentence never ends mid-thought.
- **Emails are detected and reformatted.** Text that opens with a salutation and
  reads like a message ("Hey Sarah, following up on…") is rewritten into a
  greeting, paragraphs and a sign-off by the local LLM. A free heuristic decides
  the clear cases; only genuinely ambiguous text costs a model call.
  **Ctrl+Shift+Win** forces email formatting regardless.
- **Vocabulary** is the setting worth filling in first. Whisper is reliable on
  ordinary English and hopeless on colleague and product names. The written
  forms are also handed to Whisper as hotwords, so names are usually right
  before the substitution ever runs.

Everything runs on this machine. Dictated text goes to the target window and
nowhere else — the clipboard write is even marked excluded from Win+V history
and cloud clipboard sync.

Settings → **DICTATION** changes the chords, activation style, insertion method
(paste or simulated typing), sign-off name and vocabulary. Installing with the
*Start with Windows* option adds a `--background` startup shortcut; launching
the app again while it's running opens the UI on that instance.

Two Windows caveats worth knowing: a non-elevated app cannot send input to an
elevated window, so dictating into an admin console silently does nothing; and
chords Windows reserves (Ctrl+Win+D, Ctrl+Win+arrows) cancel the recording
rather than starting one.

Only one copy runs at a time. Each instance installs a process-wide keyboard
hook, so a second one arms on the same chord and dictates everything twice —
launching the app again just opens the UI of the copy already running, whatever
port it was started on.

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
uv run oat-notes --model medium.en   # swap whisper models for this run
uv run oat-notes --backend openvino  # offload to the Intel NPU (see below)
uv run oat-notes --backend openvino --ov-device GPU   # or the Arc GPU
uv run oat-notes --offline           # never touch the network; fails if the model isn't cached yet
uv run oat-notes --ui --background   # no browser tab; dictation hotkey only
uv run oat-notes --ui --no-overlay   # no floating dictation HUD
uv run oat-notes --llm-device GPU    # run the cleanup/formatting LLM on the iGPU
```

Output, one line per speech chunk:

```
[00:03:12] Microphone: Let's walk through the model assumptions.
[00:03:28] System audio: The churn number looks high to me.
```

On exit the session is written to `transcripts/<name>_YYYY-MM-DD_HHMM.txt` (`--name "stand-up"` → `stand-up_…​.txt`), merged across channels and sorted by timestamp — ready to paste into OneNote or Claude for summarization.

`--speakers` accepts a comma-separated roster. In the UI, the saved roster determines stable hotkey slots. **Ctrl+Alt+1..9** by default selects a speaker in the active bank, while **Ctrl+Alt+[** and **Ctrl+Alt+]** page through larger rosters. The Settings tab can change the modifier combination. A manual press is authoritative for the current VAD turn on either audio source, cuts in-flight audio cleanly, and captures a quality-checked profile sample. Automatic low-confidence matches are labeled `Unknown` rather than guessed.

An `Unknown` line doesn't have to stay that way. Each unnamed turn keeps its
voice embedding in memory for the meeting, and once a profile firms up the
worker re-scores them and renames the ones it is now sure about — to a stricter
bar than live attribution, since it is rewriting a line you have already read.
Click any transcript line to put it on someone by hand: pick a name from the
roster, or type a new one and press Enter — that person is saved to the
library and added to the roster without switching the live speaker. The
correction is also a lesson: the turn's voice goes into that person's profile
and any other `Unknown` line it now explains is renamed too. Putting a line
back to `Unknown` teaches nothing. All corrections reach the saved `.txt`.

## Architecture

```
capture.py    WASAPI capture (pyaudiowpatch); callback only stamps + enqueues
clock.py      one monotonic session clock — all timestamps stamped at capture
vad.py        streaming Silero VAD (ONNX model bundled with faster-whisper)
chunker.py    state machine: split at 0.35s silence, force-split at 15s, drop blips
transcriber.py  Transcriber ABC + faster-whisper and OpenVINO backends
models.py     model download/cache resolution shared by whisper and the LLM
llm.py        one lock-serialized local LLM shared by cleanup and dictation
cleanup.py    delayed per-line transcript cleanup on top of that LLM
pipeline.py   capture → chunker thread → transcription worker → sink, via queues
attribution.py  switch log + majority-overlap speaker attribution
speaker_store.py  local SQLite speaker/group directory + embedding centroids
speaker_id.py  bundled sherpa-onnx embeddings, enrollment, roster matching
hotkeys.py    one process-wide chord hook: modifier+1..9 and modifier-only chords
inject.py     Win32 clipboard paste / unicode SendInput into the focused window
overlay.py    frameless, always-on-top, never-focused dictation HUD (tkinter)
dictation/    recorder (mic → VAD → whisper), formatter (rules + email), controller
session.py    one meeting: capture + pipeline + attribution + transcript log
single_instance.py  named kernel mutex: one keyboard hook, one microphone
line_feed.py  the transcript lines and notes the browser sees, behind one lock
log.py        the stderr logger; error_kind() keeps captured speech out of it
webapp/       the web UI: requests (validation), state, meeting/library/settings
              handlers, bridge (session callbacks → events), handler (HTTP + SSE),
              bootstrap (serve); the page itself is web/index.html
```

Chunks carry a channel tag (`MIC`/`LOOPBACK`) and segments carry a `speaker` field from day one, so the dual-stream and attribution phases slot in without restructuring.

## Local-only by design

Captured audio, transcripts, dictated text, voice embeddings, identities, groups, and recognition results never leave the machine. Profiles are stored as embeddings only—never replayable audio—under `%LOCALAPPDATA%\oat-notes\speakers.db`, outside the normal OneDrive Documents tree. Speaker inference uses a bundled ONNX model and performs no download. The web UI binds to `127.0.0.1` only and rejects requests whose `Host`/`Origin` headers don't name that address. Dictation adds nothing here: transcription, cleanup and email formatting all run on local models, and the dictated text is written only to the target window — the clipboard write is marked excluded from Win+V history and cloud clipboard sync. The only permitted network traffic is the existing one-time model download; `--offline` disables that too. HF telemetry and implicit-token lookups are disabled by default. `tests/test_local_only.py` fails the build if any module on the dictation path imports a network client.

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

To produce an installable Windows build, first complete the development setup
above and install [Inno Setup 6](https://jrsoftware.org/isinfo.php). Then run:

```powershell
uv sync --extra openvino
uv run pytest
.\scripts\build_installer.ps1
```

The script runs the test suite again by default, builds the application, and
then writes the installer to `dist\installer\OatNotes-Setup.exe`. Run that
installer to add the Start menu shortcut, uninstaller, and optional desktop
shortcut. It installs per-user under `%LOCALAPPDATA%\Programs\Oat Notes`, so
no administrator prompt is needed.

For a faster unpacked build while iterating on the application, run:

```powershell
.\scripts\build_app.ps1
```

Produces an unpacked, windowed application under
`%LOCALAPPDATA%\oat-notes-build\dist\oat-notes`. Unlike the old one-file
executable, the installed layout does not decompress 170+ MB into `%TEMP%` on
every launch. Double-clicking `oat-notes.exe` starts the UI first, then loads
and warms OpenVINO in the background. The START button enables when the model
is ready.

To upgrade an existing installation, close Oat Notes and run the newly built
installer again. Do not delete `%LOCALAPPDATA%\oat-notes\speakers.db`: it holds
the local speaker directory and embeddings and is preserved across upgrades.
Installed launches save transcripts under
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
