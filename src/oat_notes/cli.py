"""Command-line entry point: live mic(+loopback) transcription or file playback."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .capture import AudioCapture, find_default_loopback, list_input_devices
from .clock import SessionClock
from .config import Config
from .pipeline import Pipeline
from .transcriber import create_transcriber
from .types import AudioChunk, Channel, TranscriptSegment

CHANNEL_LABELS = {Channel.MIC: "Me", Channel.LOOPBACK: "Remote"}


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
        _run_file(args.wav, config, transcriber)
    else:
        _run_live(args, config, transcriber)


def _make_sink(config: Config, show_latency: bool = True, label_channels: bool = False):
    def sink(segment: TranscriptSegment, latency: float) -> None:
        label = f"{CHANNEL_LABELS[segment.channel]}: " if label_channels else ""
        line = f"[{_format_timestamp(segment.start)}] {label}{segment.text}"
        if config.debug and show_latency:
            line += f"   (+{latency:.1f}s)"
        print(line, flush=True)

    return sink


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
        pipeline = Pipeline(
            config,
            clock,
            transcriber,
            _make_sink(config, label_channels=len(channels) > 1),
            channels=channels,
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

        pipeline.start()
        for capture in captures:
            capture.start()
            role = CHANNEL_LABELS[capture.channel]
            print(f"{role:>6}: {capture.device_name}", flush=True)
        print("Listening — Ctrl+C to stop\n", flush=True)

        deadline = time.monotonic() + args.seconds if args.seconds else None
        try:
            while deadline is None or time.monotonic() < deadline:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        print("\nStopping…", flush=True)
        for capture in captures:
            capture.stop()
        pipeline.finish()
        dropped = sum(capture.dropped_blocks for capture in captures)
        if dropped:
            print(f"warning: dropped {dropped} audio blocks", file=sys.stderr)
    finally:
        pa.terminate()


def _run_file(path: str, config: Config, transcriber) -> None:
    from faster_whisper.audio import decode_audio

    samples = decode_audio(path, sampling_rate=config.sample_rate)
    duration = samples.size / config.sample_rate
    print(f"Transcribing {path} ({duration:.1f}s)\n", flush=True)

    # File blocks are fed faster than realtime, so latency numbers are meaningless.
    clock = SessionClock()
    pipeline = Pipeline(
        config, clock, transcriber, _make_sink(config, show_latency=False)
    )
    pipeline.start()
    block_size = 4096
    for offset in range(0, samples.size, block_size):
        block = samples[offset : offset + block_size]
        pipeline.frame_queue.put((Channel.MIC, offset / config.sample_rate, block))
    pipeline.finish()


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


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    main()
