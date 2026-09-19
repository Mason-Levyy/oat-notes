"""Delayed transcript cleanup: a local open-weight LLM re-writes each
finalized line using a few neighbouring lines as context, stripping filler
words and dropping hallucinated gibberish. All inference stays on this
machine."""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .llm import LlmEngine

DROP_TOKEN = "[DROP]"
MAX_GROWTH_RATIO = 1.2
_LABEL_PREFIXES = ("LINE:", "PREV:", "NEXT:")

SYSTEM_PROMPT = (
    "You clean up speech-to-text transcript lines. Remove filler words "
    "(um, uh, like, you know), stutters, repeated fragments, and "
    "transcription artifacts. Fix obvious transcription errors only when "
    "the surrounding lines make the intent clear. Never add information and "
    "never paraphrase beyond removing fillers; preserve the speaker's "
    "wording otherwise. You are given nearby lines only as context — never "
    "rewrite or repeat them, only the target line. If the entire target "
    f"line is meaningless gibberish or a transcription artifact, reply with "
    f"exactly {DROP_TOKEN}. Reply with the cleaned target line only."
)

OnCleaned = Callable[[int, "str | None"], None]


class LineCleaner(Protocol):
    def clean(
        self, text: str, before: Sequence[str] = (), after: Sequence[str] = ()
    ) -> str | None: ...


def build_prompt(
    target: str, before: Sequence[str] = (), after: Sequence[str] = ()
) -> str:
    """Frame the target line between its neighbours. With no context this is
    just the line itself, so a context-free cleaner behaves as before."""
    if not before and not after:
        return target
    lines = [f"PREV: {text}" for text in before]
    lines.append(f"LINE: {target}")
    lines.extend(f"NEXT: {text}" for text in after)
    return (
        "Clean only the line labelled LINE, using the PREV/NEXT lines as "
        "context. Do not output the PREV or NEXT lines.\n\n" + "\n".join(lines)
    )


def postprocess_output(output: str, original: str) -> str | None:
    """Apply the anti-hallucination guardrails to one model reply.

    Returns ``None`` to drop the line, otherwise the text to keep — falling
    back to ``original`` whenever the reply looks untrustworthy.
    """
    cleaned = output.strip()
    for prefix in _LABEL_PREFIXES:
        if cleaned.upper().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
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
    """Cleanup prompts on top of the shared ``LlmEngine``. The engine
    serializes generation, so dictation formatting can use the same model
    concurrently without the two conversations interleaving."""

    def __init__(self, engine: LlmEngine) -> None:
        self._engine = engine

    def clean(
        self, text: str, before: Sequence[str] = (), after: Sequence[str] = ()
    ) -> str | None:
        original = text.strip()
        if not original:
            return None
        max_new_tokens = min(256, 16 + 3 * len(original.split()))
        output = self._engine.generate(
            SYSTEM_PROMPT, build_prompt(original, before, after), max_new_tokens
        )
        return postprocess_output(output, original)

    def warm_up(self) -> None:
        """One dummy generation so the first real line isn't slowed by
        lazy compilation (mirrors the whisper warm_up)."""
        self.clean("um, hello there")


@dataclass
class _PendingLine:
    line_id: int
    text: str | None
    submitted_at: float


class CleanupWorker:
    """Context-window cleanup between the transcription sink and the LLM.

    A finalized line is cleaned once ``context_after`` later lines have
    arrived (so the raw line renders first and the model sees what came
    next), or ``max_wait_seconds`` after it landed if the conversation goes
    quiet. The model is given the already-cleaned previous lines and the raw
    following lines as context. ``finish`` drains the backlog with whatever
    context exists; a full input queue silently drops the task and the line
    stays raw (the same non-blocking stance as the audio pipeline).
    """

    def __init__(
        self,
        cleaner: LineCleaner,
        on_cleaned: OnCleaned,
        context_before: int = 2,
        context_after: int = 2,
        max_wait_seconds: float = 15.0,
    ) -> None:
        self._cleaner = cleaner
        self._on_cleaned = on_cleaned
        self._before = max(0, context_before)
        self._after = max(0, context_after)
        self._max_wait = max_wait_seconds
        self._queue: queue.Queue = queue.Queue(maxsize=128)
        self._results: dict[int, str | None] = {}
        self._results_lock = threading.Lock()
        self._buffer: list[_PendingLine] = []
        self._cursor = 0
        self._thread = threading.Thread(
            target=self._run, name="transcript-cleanup", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def submit(self, line_id: int, text: str) -> None:
        try:
            self._queue.put_nowait(_PendingLine(line_id, text, time.monotonic()))
        except queue.Full:
            pass

    def finish(self) -> None:
        """Signal end of input and drain the remaining backlog promptly."""
        self._queue.put(None)
        self._thread.join()

    def results(self) -> dict[int, str | None]:
        """Snapshot of cleaned/dropped lines, keyed by MeetingLog line id."""
        with self._results_lock:
            return dict(self._results)

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self._poll_timeout())
            except queue.Empty:
                self._process(draining=False)
                continue
            if item is None:
                self._process(draining=True)
                return
            self._buffer.append(item)
            self._process(draining=False)

    def _poll_timeout(self) -> float | None:
        return 1.0 if self._cursor < len(self._buffer) else None

    def _process(self, draining: bool) -> None:
        while self._cursor < len(self._buffer):
            index = self._cursor
            has_future = (len(self._buffer) - 1 - index) >= self._after
            aged = (time.monotonic() - self._buffer[index].submitted_at) >= self._max_wait
            if not (draining or has_future or aged):
                break
            self._clean_index(index)
            self._cursor += 1

    def _clean_index(self, index: int) -> None:
        pending = self._buffer[index]
        raw = pending.text
        before = self._context_before(index)
        after = self._context_after(index)
        try:
            cleaned = self._cleaner.clean(raw, before=before, after=after)
        except Exception as error:
            # Message deliberately omitted — never echo captured words into
            # the durable application log.
            print(
                f"transcript cleanup error: {type(error).__name__}",
                file=sys.stderr,
            )
            return
        pending.text = cleaned
        if cleaned == raw:
            return
        with self._results_lock:
            self._results[pending.line_id] = cleaned
        self._on_cleaned(pending.line_id, cleaned)

    def _context_before(self, index: int) -> list[str]:
        out: list[str] = []
        position = index - 1
        while position >= 0 and len(out) < self._before:
            text = self._buffer[position].text
            if text:
                out.append(text)
            position -= 1
        out.reverse()
        return out

    def _context_after(self, index: int) -> list[str]:
        end = min(len(self._buffer), index + 1 + self._after)
        return [self._buffer[position].text for position in range(index + 1, end)]
