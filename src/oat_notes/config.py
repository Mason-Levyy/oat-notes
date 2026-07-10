"""Pipeline configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16_000
    vad_window_samples: int = 512
    vad_threshold: float = 0.5
    silence_split_seconds: float = 0.5
    max_chunk_seconds: float = 15.0
    min_speech_seconds: float = 0.25
    pre_roll_windows: int = 2
    model_name: str = "distil-small.en"
    compute_type: str = "int8"
    language: str = "en"
    backend: str = "faster_whisper"
    openvino_model: str = "OpenVINO/whisper-small.en-int8-ov"
    openvino_device: str = "NPU"
    offline: bool = False
    debug: bool = False

    @property
    def window_seconds(self) -> float:
        return self.vad_window_samples / self.sample_rate
