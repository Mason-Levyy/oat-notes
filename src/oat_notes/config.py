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
    speaker_window_seconds: float = 1.5
    speaker_hop_seconds: float = 0.25
    speaker_confirmations: int = 2
    manual_enrollment_seconds: float = 3.0
    manual_enrollment_timeout_seconds: float = 10.0
    model_name: str = "distil-small.en"
    compute_type: str = "int8"
    language: str = "en"
    # Greedy decoding will happily loop a phrase to the end of a chunk. The
    # penalty discourages it; the compression bar is deliberately below
    # faster-whisper's default 2.4, which a phrase repeated three to five
    # times does not reach — so the temperature fallback never re-decoded it.
    repetition_penalty: float = 1.1
    compression_ratio_threshold: float = 2.0
    backend: str = "faster_whisper"
    openvino_model: str = "OpenVINO/whisper-small.en-int8-ov"
    openvino_device: str = "NPU"
    cleanup_enabled: bool = True
    cleanup_model: str = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
    cleanup_device: str = "CPU"
    cleanup_context_before: int = 2
    cleanup_context_after: int = 2
    cleanup_max_wait_seconds: float = 15.0
    dictation_max_chunk_seconds: float = 8.0
    offline: bool = False
    debug: bool = False

    @property
    def window_seconds(self) -> float:
        return self.vad_window_samples / self.sample_rate
