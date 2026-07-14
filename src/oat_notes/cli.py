"""Command-line entry point: live mic(+loopback) transcription or file playback."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    from .output import MeetingLog


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="oat-notes", description="Live meeting transcription for Windows."
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="list input devices and exit"
    )
    parser.add_argument(
        "--device-index", type=int, default=None, help="input device index"
    )
    parser.add_argument(
        "--no-loopback",
        action="store_true",
        help="capture the mic only, skip system audio",
    )
    parser.add_argument(
        "--loopback-index",
        type=int,
        default=None,
        help="loopback device index (default: loopback of the default output)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="stop automatically after this many seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("transcripts"),
        help="folder for the end-of-meeting transcript (default: ./transcripts)",
    )
    parser.add_argument(
        "--no-file",
        action="store_true",
        help="don't write a transcript file on exit",
    )
    parser.add_argument(
        "--speakers",
        default=None,
        metavar="NAMES",
        help="comma-separated speakers, * marks remote: \"Mason,Sarah,Priya*\";"
        " keys 1..N switch between them",
    )
    parser.add_argument(
        "--remote-name",
        default=None,
        metavar="NAME",
        help="shorthand for adding one remote speaker to --speakers",
    )
    parser.add_argument(
        "--name",
        default="meeting",
        metavar="TITLE",
        help="meeting name used in the transcript filename (default: meeting)",
    )
    parser.add_argument(
        "--model", default=None, help="whisper model name (default: distil-small.en)"
    )
    parser.add_argument(
        "--backend",
        choices=["faster_whisper", "openvino"],
        default=None,
        help="transcription engine (default: faster_whisper on CPU)",
    )
    parser.add_argument(
        "--ov-device",
        default=None,
        metavar="DEVICE",
        help="OpenVINO device: NPU (default), GPU, or CPU",
    )
    parser.add_argument(
        "--ov-model",
        default=None,
        metavar="DIR_OR_REPO",
        help="OpenVINO model dir or HF repo (default: OpenVINO/whisper-small.en-int8-ov)",
    )
    parser.add_argument(
        "--wav",
        metavar="PATH",
        default=None,
        help="transcribe an audio file instead of the mic (any format PyAV decodes)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="show per-chunk transcription latency"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never reach the network; fail if the model isn't already cached",
    )
    parser.add_argument(
        "--ui", action="store_true", help="launch the web UI instead of the console"
    )
    parser.add_argument(
        "--port", type=int, default=8737, help="web UI port (default: 8737)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="with --ui: don't open the browser automatically",
    )
    args = parser.parse_args()

    if args.list_devices:
        _print_devices()
        return

    config_overrides = {"debug": args.debug, "offline": args.offline}
    if args.model:
        config_overrides["model_name"] = args.model
    if args.backend:
        config_overrides["backend"] = args.backend
    if args.ov_device:
        config_overrides["openvino_device"] = args.ov_device
    if args.ov_model:
        config_overrides["openvino_model"] = args.ov_model
    config = Config(**config_overrides)

    if args.ui:
        from .server import serve

        serve(config, args, open_browser=not args.no_browser)
        return

    from .transcriber import create_transcriber

    if config.backend == "openvino":
        print(
            f"Loading {config.openvino_model} on {config.openvino_device} (OpenVINO)",
            flush=True,
        )
    else:
        print(f"Loading {config.model_name} ({config.compute_type})", flush=True)
    if not config.offline:
        print("(first run downloads the model to the HuggingFace cache)", flush=True)
    transcriber = create_transcriber(config)

    if args.wav:
        _run_file(args, config, transcriber)
    else:
        _run_live(args, config, transcriber)


def _save_transcript(args: argparse.Namespace, log: MeetingLog, started_at: datetime) -> None:
    if args.no_file:
        return
    if log.is_empty:
        print("No speech detected — no transcript written.", flush=True)
        return
    path = log.save(args.out_dir, started_at, args.name)
    print(f"Transcript saved: {path}", flush=True)


def warm_up(config: Config, transcriber) -> None:
    """One dummy transcription so the first real chunk isn't slowed by lazy init."""
    import numpy as np

    from .types import AudioChunk, Channel

    transcriber.transcribe(
        AudioChunk(
            samples=np.zeros(config.sample_rate // 2, dtype=np.float32),
            channel=Channel.MIC,
            start=0.0,
            end=0.5,
        )
    )


def _run_live(args: argparse.Namespace, config: Config, transcriber) -> None:
    from .attribution import parse_speakers
    from .output import format_timestamp
    from .session import Session, SessionEvents, SessionOptions
    from .types import Channel, TranscriptSegment

    warm_up(config, transcriber)
    roster = parse_speakers(args.speakers or "")
    if args.remote_name:
        roster = roster + (Speaker(args.remote_name.strip(), remote=True),)

    def print_segment(segment: TranscriptSegment, label: str, latency: float) -> None:
        prefix = f"{label}: " if label else ""
        line = f"[{format_timestamp(segment.start)}] {prefix}{segment.text}"
        if config.debug:
            line += f"   (+{latency:.1f}s)"
        print(line, flush=True)

    def print_speaker(index: int) -> None:
        if 0 <= index < len(session.roster):
            print(f"  → active speaker: {session.roster[index].name}", flush=True)

    session = Session(
        config,
        SessionOptions(
            speakers=roster,
            meeting_name=args.name,
            use_loopback=not args.no_loopback,
            device_index=args.device_index,
            loopback_index=args.loopback_index,
            out_dir=args.out_dir,
            save_file=not args.no_file,
        ),
        transcriber,
        SessionEvents(on_segment=print_segment, on_speaker=print_speaker),
    )
    session.start()
    for capture in session.captures:
        role = "Me" if capture.channel is Channel.MIC else "Remote"
        print(f"{role:>6}: {capture.device_name}", flush=True)
    if len(roster) > 1:
        mapping = "  ".join(
            f"[{number}] {speaker.name}{'*' if speaker.remote else ''}"
            for number, speaker in enumerate(roster, start=1)
        )
        print(
            f"Speakers: {mapping} — Ctrl+Alt+number to switch; brackets page",
            flush=True,
        )
    print("Listening — Ctrl+C to stop\n", flush=True)

    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print("\nStopping…", flush=True)
    session.stop()
    saved = session.save()
    if saved is not None:
        print(f"Transcript saved: {saved}", flush=True)
    elif not args.no_file:
        print("No speech detected — no transcript written.", flush=True)
    if session.dropped_blocks:
        print(
            f"warning: dropped {session.dropped_blocks} audio blocks",
            file=sys.stderr,
        )


def _run_file(args: argparse.Namespace, config: Config, transcriber) -> None:
    from faster_whisper.audio import decode_audio

    from .clock import SessionClock
    from .output import MeetingLog, line_for
    from .pipeline import Pipeline
    from .types import Channel

    samples = decode_audio(args.wav, sampling_rate=config.sample_rate)
    duration = samples.size / config.sample_rate
    print(f"Transcribing {args.wav} ({duration:.1f}s)\n", flush=True)

    started_at = datetime.now()
    clock = SessionClock()
    log = MeetingLog(label_channels=False)

    def sink(segment: TranscriptSegment, latency: float) -> None:
        log.add(segment)
        print(line_for(segment, label_channels=False), flush=True)

    pipeline = Pipeline(config, clock, transcriber, sink)
    pipeline.start()
    block_size = 4096
    for offset in range(0, samples.size, block_size):
        block = samples[offset : offset + block_size]
        pipeline.frame_queue.put((Channel.MIC, offset / config.sample_rate, block))
    pipeline.finish()
    _save_transcript(args, log, started_at)


def _print_devices() -> None:
    import pyaudiowpatch as pyaudio

    from .capture import list_input_devices

    pa = pyaudio.PyAudio()
    try:
        for device in list_input_devices(pa):
            tags = []
            if device["is_default"]:
                tags.append("default")
            if device["is_loopback"]:
                tags.append("loopback")
            suffix = f"  [{', '.join(tags)}]" if tags else ""
            print(
                f"{device['index']:3d}  {device['name']}"
                f"  ({device['channels']}ch @ {device['rate']} Hz){suffix}"
            )
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
