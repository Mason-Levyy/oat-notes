"""One local open-weight LLM, shared by every consumer.

Transcript cleanup and dictation formatting want the same small instruct
model, and loading it twice would cost about a gigabyte and a second cold
start. The lock is what makes sharing safe: ``start_chat``/``finish_chat``
mutate pipeline state, so two threads generating at once would splice each
other's conversations together.
"""

from __future__ import annotations

import threading

from .config import Config
from .models import (
    load_on_device_or_fall_back,
    openvino_cache_dir,
    openvino_model_cached,
    resolve_openvino_model,
)


class LlmEngine:
    def __init__(
        self, model: str, device: str = "CPU", offline: bool = False
    ) -> None:
        try:
            import openvino_genai
        except ImportError as error:
            raise RuntimeError(
                "The local LLM is an optional extra — install it with:"
                " uv sync --extra openvino"
            ) from error

        model_dir = resolve_openvino_model(model, offline)
        self.model = model
        self._lock = threading.Lock()
        cache_dir = str(openvino_cache_dir())
        self._pipeline, self.device = load_on_device_or_fall_back(
            lambda chosen: openvino_genai.LLMPipeline(
                model_dir, device=chosen, CACHE_DIR=cache_dir
            ),
            device,
            "language model",
        )

    @classmethod
    def from_config(cls, config: Config) -> LlmEngine:
        return cls(config.cleanup_model, config.cleanup_device, config.offline)

    def generate(self, system_prompt: str, prompt: str, max_new_tokens: int) -> str:
        """Greedy single-turn generation. Each call is its own conversation,
        so no state carries between a cleanup line and a dictation."""
        with self._lock:
            self._pipeline.start_chat(system_prompt)
            try:
                output = self._pipeline.generate(
                    prompt, max_new_tokens=max_new_tokens, do_sample=False
                )
            finally:
                self._pipeline.finish_chat()
        return str(output)

    def warm_up(self, system_prompt: str = "You are a helpful assistant.") -> None:
        """One dummy generation so the first real request isn't slowed by
        lazy compilation."""
        self.generate(system_prompt, "hello", max_new_tokens=4)


def model_is_cached(config: Config) -> bool:
    return openvino_model_cached(config.cleanup_model)
