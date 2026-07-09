"""Command-line entry point: live mic transcription or file playback."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .capture import AudioCapture, list_input_devices
from .clock import SessionClock
from .config import Config
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
        _run_live(args.device_index, config, transcriber)


def _make_sink(config: Config, show_latency: bool = True):
    def sink(segment: TranscriptSegment, latency: float) -> None:
        line = f"[{_format_timestamp(segment.start)}] {segment.text}"
        if config.debug and show_latency:
            line += f"   (+{latency:.1f}s)"
        print(line, flush=True)

    return sink


def _run_live(device_index: int | None, config: Config, transcriber) -> None:
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
    pipeline = Pipeline(config, clock, transcriber, _make_sink(config))
    pa = pyaudio.PyAudio()
    try:
        capture = AudioCapture(
            pa, device_index, Channel.MIC, clock, config, pipeline.frame_queue
        )
        pipeline.start()
        capture.start()
        print(f"Listening on: {capture.device_name} — Ctrl+C to stop\n", flush=True)
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopping…", flush=True)
        capture.stop()
        pipeline.finish()
        if capture.dropped_blocks:
            print(
                f"warning: dropped {capture.dropped_blocks} audio blocks",
                file=sys.stderr,
            )
    finally:
        pa.terminate()


def _run_file(path: str, config: Config, transcriber) -> None:
    from faster_whisper.audio import decode_audio

    samples = decode_audio(path, sampling_rate=config.sample_rate)
    duration = samples.size / config.sample_rate
    print(f"Transcribing {path} ({duration:.1f}s)\n", flush=True)

    # File blocks are fed faster than realtime, so latency numbers are meaningless.
    clock = SessionClock()
    pipeline = Pipeline(config, clock, transcriber, _make_sink(config, show_latency=False))
    pipeline.start()
    block_size = 4096
    for offset in range(0, samples.size, block_size):
        block = samples[offset : offset + block_size]
        pipeline.frame_queue.put((offset / config.sample_rate, block))
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
