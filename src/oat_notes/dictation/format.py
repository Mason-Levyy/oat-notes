"""Turning a raw dictation transcript into the text to insert.

Every dictation gets ``apply_rules`` — pure regex work, and enough on its own
because Whisper already emits punctuation and capitalisation. The local LLM
is reserved for email formatting.

Deciding whether something is an email has the same shape: a free scoring
heuristic answers the clear-cut cases, and only ambiguous text costs a call.
"""

from __future__ import annotations

import re
import sys
from typing import Sequence

from ..llm import LlmEngine
from ..repetition import collapse_repeated_runs

_FILLERS = ("um", "uhm", "uh", "erm", "er", "ah", "hmm", "mm", "mhm")
_FILLER_WITH_SURROUNDING_COMMAS = re.compile(
    r"\s*,?\s*\b(?:" + "|".join(_FILLERS) + r")\b\s*,?\s*", re.IGNORECASE
)
_WORDS_THAT_LEGITIMATELY_DOUBLE = {"had", "that", "is", "no", "very", "really"}
_REPEAT_PATTERN = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)

_LINE_COMMANDS = (
    (re.compile(r"\s*(?<!\w)new\s+paragraph(?!\w)[.,]?\s*", re.IGNORECASE), "\n\n"),
    (re.compile(r"\s*(?<!\w)new\s+line(?!\w)[.,]?\s*", re.IGNORECASE), "\n"),
)
_SPOKEN_PUNCTUATION = (
    (re.compile(r"\s*(?<!\w)question\s+mark(?!\w)", re.IGNORECASE), "?"),
    (re.compile(r"\s*(?<!\w)exclamation\s+(?:mark|point)(?!\w)", re.IGNORECASE), "!"),
    (re.compile(r"\s*(?<!\w)semicolon(?!\w)", re.IGNORECASE), ";"),
    (re.compile(r"\s*(?<!\w)full\s+stop(?!\w)", re.IGNORECASE), "."),
    (re.compile(r"\s*(?<!\w)period(?!\w)", re.IGNORECASE), "."),
    (re.compile(r"\s*(?<!\w)comma(?!\w)", re.IGNORECASE), ","),
    (re.compile(r"\s*(?<!\w)colon(?!\w)", re.IGNORECASE), ":"),
    (re.compile(r"\s*(?<!\w)(?:close|end)\s+quotes?(?!\w)", re.IGNORECASE), '"'),
    (re.compile(r"(?<!\w)(?:open|begin)\s+quotes?(?!\w)\s*", re.IGNORECASE), '"'),
    (re.compile(r"\s*(?<!\w)close\s+(?:paren|parenthesis|bracket)(?!\w)", re.IGNORECASE), ")"),
    (re.compile(r"(?<!\w)open\s+(?:paren|parenthesis|bracket)(?!\w)\s*", re.IGNORECASE), "("),
    (re.compile(r"\s*(?<!\w)(?:dash|hyphen)(?!\w)\s*", re.IGNORECASE), "-"),
)
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?)])")
_COMMA_AFTER_PUNCTUATION = re.compile(r"([,.;:!?])\s*,")
_COMMA_BEFORE_STOP = re.compile(r",\s*([.!?])")
_DOUBLE_STOP = re.compile(r"(?<!\.)\.\s*\.(?!\.)")
_SENTENCE_START = re.compile(r"(^|\n\n)([a-z])")
_MIN_WORDS_FOR_FULL_STOP = 3


def _end_paragraphs(text: str) -> str:
    """A spoken "new paragraph" ends a sentence, whether or not Whisper
    heard a full stop before it. Fragments too short to be sentences —
    a greeting, a sign-off — are left as they are."""
    paragraphs = text.split("\n\n")
    for position, paragraph in enumerate(paragraphs[:-1]):
        is_sentence_length = len(paragraph.split()) >= _MIN_WORDS_FOR_FULL_STOP
        if paragraph and paragraph[-1].isalnum() and is_sentence_length:
            paragraphs[position] = paragraph + "."
    return "\n\n".join(paragraphs)


def _collapse_repeats(match: re.Match) -> str:
    word = match.group(1)
    if word.lower() in _WORDS_THAT_LEGITIMATELY_DOUBLE:
        return match.group(0)
    return word


def vocabulary_hotwords(vocabulary: Sequence[tuple[str, str]]) -> str | None:
    """The written forms, for Whisper to bias towards while decoding —
    cheaper than mis-hearing a name and patching it afterwards."""
    written = [
        item.strip() for heard, item in vocabulary if heard.strip() and item.strip()
    ]
    return ", ".join(dict.fromkeys(written)) or None


def apply_vocabulary(text: str, vocabulary: Sequence[tuple[str, str]]) -> str:
    """Whole-word, case-insensitive substitutions.

    Worth more than everything else here put together: Whisper is reliable on
    ordinary English and hopeless on colleague and product names.
    """
    for heard, written in vocabulary:
        heard = heard.strip()
        if not heard:
            continue
        text = re.sub(
            r"(?<!\w)" + re.escape(heard) + r"(?!\w)",
            written.replace("\\", "\\\\"),
            text,
            flags=re.IGNORECASE,
        )
    return text


def apply_rules(
    text: str,
    vocabulary: Sequence[tuple[str, str]] = (),
    spoken_punctuation: bool = False,
) -> str:
    """The default dictation path: deterministic and effectively free."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = _FILLER_WITH_SURROUNDING_COMMAS.sub(" ", cleaned)
    cleaned = _REPEAT_PATTERN.sub(_collapse_repeats, cleaned)
    cleaned = collapse_repeated_runs(cleaned)
    for pattern, replacement in _LINE_COMMANDS:
        cleaned = pattern.sub(replacement, cleaned)
    if spoken_punctuation:
        for pattern, replacement in _SPOKEN_PUNCTUATION:
            cleaned = pattern.sub(replacement, cleaned)
    cleaned = apply_vocabulary(cleaned, vocabulary)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)
    cleaned = _COMMA_AFTER_PUNCTUATION.sub(r"\1", cleaned)
    cleaned = _COMMA_BEFORE_STOP.sub(r"\1", cleaned)
    cleaned = _DOUBLE_STOP.sub(".", cleaned)
    cleaned = _end_paragraphs(cleaned.strip())
    return _SENTENCE_START.sub(lambda match: match.group(1) + match.group(2).upper(), cleaned)


_EXPLICIT_EMAIL = re.compile(
    r"(?<!\w)(?:write|send|draft|compose)\s+(?:me\s+)?an?\s+email(?!\w)", re.IGNORECASE
)
_GREETING_ADDRESSEE_AND_COMMA = re.compile(
    r"^\s*(?:hey|hi|hello|dear|good\s+(?:morning|afternoon|evening))"
    r"[^,.!?\n]{0,30},",
    re.IGNORECASE,
)
_CLOSINGS = (
    "thanks", "thank you", "best", "regards", "cheers", "let me know",
    "looking forward", "talk soon", "speak soon", "appreciate it",
)
_SECOND_PERSON = re.compile(r"(?<!\w)(?:you|your|you're|yours)(?!\w)", re.IGNORECASE)

EMAIL_SCORE_HIGH = 0.8
EMAIL_SCORE_LOW = 0.3

_CLASSIFY_SYSTEM = (
    "You label dictated text. Reply with exactly EMAIL if the text is meant "
    "to be sent to someone as an email or message with a greeting and a "
    "sign-off. Reply with exactly TEXT for anything else — notes, search "
    "queries, code, chat messages, or thinking out loud. Reply with one word."
)


def email_score(text: str) -> float:
    """0 means certainly not an email, 1 certainly is."""
    stripped = text.strip()
    if not stripped:
        return 0.0
    if _EXPLICIT_EMAIL.search(stripped):
        return 1.0

    words = stripped.split()
    score = 0.0
    if _GREETING_ADDRESSEE_AND_COMMA.search(stripped):
        score += 0.45
    tail = " ".join(words[-12:]).lower()
    if any(closing in tail for closing in _CLOSINGS):
        score += 0.3
    elif any(closing in stripped.lower() for closing in _CLOSINGS):
        score += 0.15
    if len(words) >= 25:
        score += 0.15
    if len(words) >= 60:
        score += 0.05
    if _SECOND_PERSON.search(stripped):
        score += 0.1
    return min(1.0, score)


def classify_email(text: str, engine: LlmEngine) -> bool:
    """One short model call for the ambiguous band. Any failure means 'not an
    email' — inserting plain text where an email was wanted is a much smaller
    annoyance than the reverse."""
    try:
        reply = engine.generate(_CLASSIFY_SYSTEM, text.strip()[:600], max_new_tokens=3)
    except Exception as error:
        print(f"email classification failed: {type(error).__name__}", file=sys.stderr)
        return False
    return "EMAIL" in reply.strip().upper()


def detect_email(text: str, engine: LlmEngine | None = None) -> bool:
    score = email_score(text)
    if score >= EMAIL_SCORE_HIGH:
        return True
    if score <= EMAIL_SCORE_LOW or engine is None:
        return False
    return classify_email(text, engine)


_PLACEHOLDER = re.compile(r"\[[^\]\n]{1,40}\]|<[a-z ]{1,30}>", re.IGNORECASE)
_PREAMBLE = re.compile(
    r"^(?:sure|certainly|of course|here(?:'s| is)|okay|ok)\b[^\n:]{0,60}:\s*\n?",
    re.IGNORECASE,
)
_SUBJECT_LINE = re.compile(
    r"^[ \t]*[*#>]*[ \t]*subject[ \t]*:.*(?:\r?\n|$)", re.IGNORECASE
)
MIN_RETAINED_RATIO = 0.6

EMAIL_SYSTEM_PROMPT = (
    "You format dictated speech into an email. Keep every fact, request and "
    "commitment exactly as dictated. Never invent recipients, dates, "
    "numbers, deadlines, or promises that were not spoken. Never write a "
    "subject line — start at the greeting. Fix grammar and punctuation, "
    "remove filler words, and break the body into short paragraphs. Keep the "
    "speaker's own wording and tone wherever it already reads well. Reply "
    "with the email itself and nothing else — no commentary, no explanation, "
    "and never a bracketed placeholder of any kind."
)


def build_email_prompt(text: str) -> str:
    closing = (
        "End with a short closing line such as 'Thanks,' and do not sign "
        "a name after it."
    )
    return f"{closing}\n\nDictated text:\n{text.strip()}"


def postprocess_email(output: str, original: str) -> str | None:
    """Guardrails for the email pass.

    Unlike transcript cleanup there is no growth clamp — reformatting into
    paragraphs legitimately expands the text. The failure modes are the
    opposite ones: dropped content and placeholder tokens. Returns None when
    the reply cannot be trusted.

    The subject line is stripped rather than merely forbidden in the prompt,
    because a 1.5B model writes one anyway often enough to matter.
    """
    cleaned = _PREAMBLE.sub("", output.strip()).strip()
    cleaned = _SUBJECT_LINE.sub("", cleaned, count=1).strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 1:
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return None
    if _PLACEHOLDER.search(cleaned):
        return None
    if len(cleaned) < MIN_RETAINED_RATIO * len(original.strip()):
        return None
    return cleaned


def format_email(text: str, engine: LlmEngine) -> str:
    """Rewrite as an email, falling back to the input if anything looks off."""
    original = text.strip()
    if not original:
        return ""
    max_new_tokens = int(64 + 2.5 * len(original.split()))
    try:
        output = engine.generate(
            EMAIL_SYSTEM_PROMPT, build_email_prompt(original), max_new_tokens
        )
    except Exception as error:
        print(f"email formatting failed: {type(error).__name__}", file=sys.stderr)
        return original
    return postprocess_email(output, original) or original


TEXT_MODE = "text"
EMAIL_MODE = "email"


def format_dictation(
    raw: str,
    engine: LlmEngine | None = None,
    force_email: bool = False,
    vocabulary: Sequence[tuple[str, str]] = (),
    spoken_punctuation: bool = False,
    detect: bool = True,
) -> tuple[str, str]:
    """Raw transcript in, text-to-insert and its mode out."""
    cleaned = apply_rules(raw, vocabulary, spoken_punctuation)
    if not cleaned:
        return "", TEXT_MODE
    wants_email = force_email or (
        detect and engine is not None and detect_email(cleaned, engine)
    )
    if not wants_email or engine is None:
        return cleaned, TEXT_MODE
    return format_email(cleaned, engine), EMAIL_MODE
