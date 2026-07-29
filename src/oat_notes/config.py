"""Pipeline configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16_000
    vad_window_samples: int = 512
    vad_threshold: float = 0.5
    silence_split_seconds: float = 0.35
    max_chunk_seconds: float = 15.0
    min_speech_seconds: float = 0.25
    pre_roll_windows: int = 2
    # People answering each other leave no silence to split on, so the turn
    # boundary has to come from the voice itself. A shorter window and hop
    # notice the change in about a third of a second.
    speaker_window_seconds: float = 1.0
    speaker_hop_seconds: float = 0.15
    # How much of a tracking window has to be speech before it is worth
    # embedding. Distinct from the enrollment floor: inside a 1.0s window,
    # demanding a full second of speech would mean 100% and stop tracking
    # producing windows at all.
    speaker_min_speech_seconds: float = 0.6
    speaker_confirmations: int = 2
    manual_enrollment_seconds: float = 3.0
    manual_enrollment_timeout_seconds: float = 10.0
    # The distil-* family is trained down for speed on standard English and
    # is measurably weaker on an accent it wasn't distilled against — which
    # is most of them. Settings can trade back for speed.
    model_name: str = "small.en"
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
