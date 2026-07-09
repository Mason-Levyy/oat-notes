"""Command-line entry point: live mic(+loopback) transcription or file playback."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .attribution import Attributor, SwitchLog
from .capture import AudioCapture, find_default_loopback, list_input_devices
from .clock import SessionClock
from .config import Config
from .hotkeys import HotkeyListener
from .output import MeetingLog, line_for
from .pipeline import Pipeline
from .transcriber import create_transcriber
from .types import AudioChunk, Channel, TranscriptSegment


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
        help="comma-separated in-person speaker names; keys 1..N switch between them",
    )
    parser.add_argument(
        "--remote-name",
        default=None,
        metavar="NAME",
        help="name for the remote (loopback) party instead of 'Remote'",
    )
    parser.add_argument(
        "--model", default=None, help="whisper model name (default: distil-small.en)"
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
    args = parser.parse_args()

    if args.list_devices:
        _print_devices()
        return

    config_overrides = {"debug": args.debug}
    if args.model:
        config_overrides["model_name"] = args.model
    config = Config(**config_overrides)

    print(f"Loading {config.model_name} ({config.compute_type})", flush=True)
    print("(first run downloads the model to the HuggingFace cache)", flush=True)
    transcriber = create_transcriber(config)

    if args.wav:
        _run_file(args, config, transcriber)
    else:
        _run_live(args, config, transcriber)


def _make_sink(
    config: Config,
    log: MeetingLog,
    show_latency: bool = True,
    label_channels: bool = False,
):
    def sink(segment: TranscriptSegment, latency: float) -> None:
        log.add(segment)
        line = line_for(segment, label_channels)
        if config.debug and show_latency:
            line += f"   (+{latency:.1f}s)"
        print(line, flush=True)

    return sink


def _save_transcript(args: argparse.Namespace, log: MeetingLog, started_at: datetime) -> None:
    if args.no_file:
        return
    if log.is_empty:
        print("No speech detected — no transcript written.", flush=True)
        return
    path = log.save(args.out_dir, started_at)
    print(f"Transcript saved: {path}", flush=True)


def _run_live(args: argparse.Namespace, config: Config, transcriber) -> None:
    import pyaudiowpatch as pyaudio

    # Warm up so the first real chunk doesn't absorb lazy-init cost.
    transcriber.transcribe(
        AudioChunk(
            samples=np.zeros(config.sample_rate // 2, dtype=np.float32),
            channel=Channel.MIC,
            start=0.0,
            end=0.5,
        )
    )

    started_at = datetime.now()
    clock = SessionClock()
    pa = pyaudio.PyAudio()
    try:
        loopback_info = None
        if not args.no_loopback:
            if args.loopback_index is not None:
                loopback_info = pa.get_device_info_by_index(args.loopback_index)
            else:
                loopback_info = find_default_loopback(pa)
                if loopback_info is None:
                    print(
                        "warning: no loopback device found — capturing mic only",
                        file=sys.stderr,
                    )

        channels = (
            (Channel.MIC, Channel.LOOPBACK) if loopback_info else (Channel.MIC,)
        )
        speakers = (
            [name.strip() for name in args.speakers.split(",") if name.strip()]
            if args.speakers
            else []
        )
        switch_log = SwitchLog()
        attributor = None
        if speakers or args.remote_name:
            attributor = Attributor(speakers, switch_log, args.remote_name)

        log = MeetingLog(label_channels=len(channels) > 1)
        pipeline = Pipeline(
            config,
            clock,
            transcriber,
            _make_sink(config, log, label_channels=len(channels) > 1),
            channels=channels,
            attributor=attributor,
        )

        captures = [
            AudioCapture(
                pa, args.device_index, Channel.MIC, clock, config,
                pipeline.frame_queue,
            )
        ]
        if loopback_info is not None:
            captures.append(
                AudioCapture(
                    pa, int(loopback_info["index"]), Channel.LOOPBACK, clock,
                    config, pipeline.frame_queue,
                )
            )

        hotkeys = None
        if len(speakers) > 1:

            def on_switch(index: int) -> None:
                switch_log.record(clock.now(), index)
                print(f"  → active speaker: {speakers[index]}", flush=True)

            hotkeys = HotkeyListener(len(speakers), on_switch)
            hotkeys.start()

        pipeline.start()
        for capture in captures:
            capture.start()
            role = "Me" if capture.channel is Channel.MIC else "Remote"
            print(f"{role:>6}: {capture.device_name}", flush=True)
        if len(speakers) > 1:
            mapping = "  ".join(
                f"[{number}] {name}" for number, name in enumerate(speakers, start=1)
            )
            print(f"Speakers: {mapping} — press the number key to switch", flush=True)
            print(f"Active speaker: {speakers[0]}", flush=True)
        print("Listening — Ctrl+C to stop\n", flush=True)

        deadline = time.monotonic() + args.seconds if args.seconds else None
        try:
            while deadline is None or time.monotonic() < deadline:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        print("\nStopping…", flush=True)
        if hotkeys is not None:
            hotkeys.stop()
        for capture in captures:
            capture.stop()
        pipeline.finish()
        _save_transcript(args, log, started_at)
        dropped = sum(capture.dropped_blocks for capture in captures)
        if dropped:
            print(f"warning: dropped {dropped} audio blocks", file=sys.stderr)
    finally:
        pa.terminate()


def _run_file(args: argparse.Namespace, config: Config, transcriber) -> None:
    from faster_whisper.audio import decode_audio

    samples = decode_audio(args.wav, sampling_rate=config.sample_rate)
    duration = samples.size / config.sample_rate
    print(f"Transcribing {args.wav} ({duration:.1f}s)\n", flush=True)

    # File blocks are fed faster than realtime, so latency numbers are meaningless.
    started_at = datetime.now()
    clock = SessionClock()
    log = MeetingLog(label_channels=False)
    pipeline = Pipeline(
        config, clock, transcriber, _make_sink(config, log, show_latency=False)
    )
    pipeline.start()
    block_size = 4096
    for offset in range(0, samples.size, block_size):
        block = samples[offset : offset + block_size]
        pipeline.frame_queue.put((Channel.MIC, offset / config.sample_rate, block))
    pipeline.finish()
    _save_transcript(args, log, started_at)


def _print_devices() -> None:
    import pyaudiowpatch as pyaudio

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
