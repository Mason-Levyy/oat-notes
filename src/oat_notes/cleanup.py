"""Delayed transcript cleanup: a local open-weight LLM re-writes each
finalized line a few seconds after it lands, stripping filler words and
dropping hallucinated gibberish. All inference stays on this machine."""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import Config

DROP_TOKEN = "[DROP]"
# A cleanup pass only ever removes or repairs words. Output that grows past
# this ratio means the model started inventing content — keep the original.
MAX_GROWTH_RATIO = 1.2

SYSTEM_PROMPT = (
    "You clean up speech-to-text transcript lines. Remove filler words "
    "(um, uh, like, you know), stutters, repeated fragments, and "
    "transcription artifacts. Fix obvious transcription errors only when "
    "the intent is clear. Never add information and never paraphrase "
    "beyond removing fillers; preserve the speaker's wording otherwise. "
    f"If the entire line is meaningless gibberish or a transcription "
    f"artifact, reply with exactly {DROP_TOKEN}. "
    "Reply with the cleaned line only."
)

OnCleaned = Callable[[int, "str | None"], None]


class LineCleaner(Protocol):
    def clean(self, text: str) -> str | None: ...


def postprocess_output(output: str, original: str) -> str | None:
    """Apply the anti-hallucination guardrails to one model reply.

    Returns ``None`` to drop the line, otherwise the text to keep — falling
    back to ``original`` whenever the reply looks untrustworthy.
    """
    cleaned = output.strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 1:
        cleaned = cleaned[1:-1].strip()
    if DROP_TOKEN in cleaned and len(cleaned) <= len(DROP_TOKEN) + 4:
        return None
    if not cleaned:
        return original
    if len(cleaned) > MAX_GROWTH_RATIO * len(original) + 8:
        return original
    return cleaned


class TranscriptCleaner:
    """One ``openvino_genai.LLMPipeline`` used from the cleanup worker
    thread only — calls are serialized by that single consumer."""

    def __init__(self, config: Config, model_dir: str | None = None) -> None:
        import openvino_genai

        from .transcriber import resolve_openvino_model

        source = model_dir or resolve_openvino_model(
            config.cleanup_model, config.offline
        )
        self._pipeline = openvino_genai.LLMPipeline(
            source, device=config.cleanup_device
        )

    def clean(self, text: str) -> str | None:
        original = text.strip()
        if not original:
            return None
        max_new_tokens = min(256, 16 + 3 * len(original.split()))
        self._pipeline.start_chat(SYSTEM_PROMPT)
        try:
            output = self._pipeline.generate(
                original, max_new_tokens=max_new_tokens, do_sample=False
            )
        finally:
            self._pipeline.finish_chat()
        return postprocess_output(str(output), original)

    def warm_up(self) -> None:
        """One dummy generation so the first real line isn't slowed by
        lazy compilation (mirrors the whisper warm_up)."""
        self.clean("um, hello there")


@dataclass(frozen=True)
class _Job:
    line_id: int
    text: str
    submitted_at: float  # time.monotonic()


class CleanupWorker:
    """Delay-and-clean queue between the transcription sink and the LLM.

    Lines are cleaned no earlier than ``delay_seconds`` after they were
    submitted so the raw line is always rendered first. ``finish`` drains
    the backlog without the delay; a full queue silently skips the task and
    the line stays raw (same non-blocking stance as the audio pipeline).
    """

    def __init__(
        self,
        cleaner: LineCleaner,
        delay_seconds: float,
        on_cleaned: OnCleaned,
    ) -> None:
        self._cleaner = cleaner
        self._delay = delay_seconds
        self._on_cleaned = on_cleaned
        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._results: dict[int, str | None] = {}
        self._results_lock = threading.Lock()
        self._draining = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="transcript-cleanup", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def submit(self, line_id: int, text: str) -> None:
        try:
            self._queue.put_nowait(_Job(line_id, text, time.monotonic()))
        except queue.Full:
            pass

    def finish(self) -> None:
        """Signal end of input and drain the remaining backlog promptly."""
        self._draining.set()
        self._queue.put(None)
        self._thread.join()

    def results(self) -> dict[int, str | None]:
        """Snapshot of cleaned/dropped lines, keyed by MeetingLog line id."""
        with self._results_lock:
            return dict(self._results)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            wait = job.submitted_at + self._delay - time.monotonic()
            if wait > 0:
                # Interruptible: finish() releases the remaining delay.
                self._draining.wait(wait)
            try:
                cleaned = self._cleaner.clean(job.text)
            except Exception as error:
                # Message deliberately omitted — never echo captured words
                # into the durable application log.
                print(
                    f"transcript cleanup error: {type(error).__name__}",
                    file=sys.stderr,
                )
                continue
            if cleaned == job.text:
                continue
            with self._results_lock:
                self._results[job.line_id] = cleaned
            self._on_cleaned(job.line_id, cleaned)
