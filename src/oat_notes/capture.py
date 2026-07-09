"""WASAPI audio capture via pyaudiowpatch.

The PortAudio callback does nothing but stamp, convert, and enqueue — all
real work happens downstream. The same class will serve the loopback stream
in Phase 3 (different device index, Channel.LOOPBACK).

Capture is attempted at the pipeline rate (16 kHz); if the device refuses,
it falls back to the device's default rate and downsamples in the callback
via linear interpolation, which is adequate for speech ASR.
"""

from __future__ import annotations

import queue

import numpy as np
import pyaudiowpatch as pyaudio

from .clock import SessionClock
from .config import Config
from .types import Channel

FrameBlock = tuple[float, np.ndarray]


class AudioCapture:
    def __init__(
        self,
        pa: pyaudio.PyAudio,
        device_index: int | None,
        channel: Channel,
        clock: SessionClock,
        config: Config,
        frame_queue: queue.Queue[FrameBlock | None],
    ) -> None:
        self._pa = pa
        self._channel = channel
        self._clock = clock
        self._config = config
        self._frame_queue = frame_queue
        self._stream: pyaudio.Stream | None = None
        self.dropped_blocks = 0

        if device_index is None:
            device_info = pa.get_default_input_device_info()
        else:
            device_info = pa.get_device_info_by_index(device_index)
        self._device_index = int(device_info["index"])
        self.device_name = str(device_info["name"])
        self._device_channels = max(1, int(device_info["maxInputChannels"]))
        self._device_default_rate = int(device_info["defaultSampleRate"])

    def start(self) -> None:
        self._capture_rate = self._config.sample_rate
        try:
            self._stream = self._open_stream(self._capture_rate)
        except OSError:
            self._capture_rate = self._device_default_rate
            self._stream = self._open_stream(self._capture_rate)
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def _open_stream(self, rate: int) -> pyaudio.Stream:
        blocks_per_second = self._config.sample_rate / self._config.vad_window_samples
        frames_per_buffer = int(rate / blocks_per_second)
        return self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._device_channels,
            rate=rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=frames_per_buffer,
            stream_callback=self._callback,
            start=False,
        )

    def _callback(self, in_data, frame_count, time_info, status):
        block_start = self._clock.now() - frame_count / self._capture_rate
        samples = np.frombuffer(in_data, dtype=np.float32)
        if self._device_channels > 1:
            samples = samples.reshape(-1, self._device_channels).mean(axis=1)
        if self._capture_rate != self._config.sample_rate:
            samples = _resample(samples, self._capture_rate, self._config.sample_rate)
        try:
            self._frame_queue.put_nowait((block_start, samples))
        except queue.Full:
            self.dropped_blocks += 1
        return (None, pyaudio.paContinue)


def _resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    target_length = int(round(samples.size * to_rate / from_rate))
    positions = np.linspace(0, samples.size - 1, target_length)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def list_input_devices(pa: pyaudio.PyAudio) -> list[dict]:
    """All input-capable devices, including WASAPI loopback endpoints."""
    devices = []
    try:
        default_index = int(pa.get_default_input_device_info()["index"])
    except OSError:
        default_index = -1
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info["maxInputChannels"]) < 1:
            continue
        devices.append(
            {
                "index": int(info["index"]),
                "name": str(info["name"]),
                "rate": int(info["defaultSampleRate"]),
                "channels": int(info["maxInputChannels"]),
                "is_loopback": bool(info.get("isLoopbackDevice", False)),
                "is_default": int(info["index"]) == default_index,
            }
        )
    return devices
