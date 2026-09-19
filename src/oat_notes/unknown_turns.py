"""The turns nobody was named for, kept so a later identification can go
back and name them. Embeddings only, in memory only, bounded."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import cosine, cosine_scores
from .speaker_store import SpeakerStore
from .types import Channel

MEMORY_LIMIT = 400
BACKFILL_THRESHOLD = 0.70
BACKFILL_MARGIN = 0.10


@dataclass(frozen=True)
class UnknownTurn:
    channel: Channel
    embedding: np.ndarray
    speech_seconds: float


def remember(
    turns: dict[int, UnknownTurn],
    line_id: int,
    channel: Channel,
    embedding: np.ndarray,
    speech_seconds: float,
) -> None:
    turns[line_id] = UnknownTurn(channel, np.array(embedding, copy=True), speech_seconds)
    while len(turns) > MEMORY_LIMIT:
        turns.pop(next(iter(turns)))


def backfill_score(
    turn: UnknownTurn, store: SpeakerStore, speaker_id: str, rival_ids: list[str]
) -> float | None:
    """How confidently ``speaker_id`` explains this turn, or None when it
    does not. Deliberately stricter than live attribution: this rewrites
    text the user has already read, so a near-miss stays Unknown."""
    own = store.profile_vector(speaker_id, turn.channel)
    score = None if own is None else cosine(turn.embedding, own.embedding)
    if score is None or score < BACKFILL_THRESHOLD:
        return None
    rivals = store.match_vectors(rival_ids, turn.channel)
    rival_scores = cosine_scores(turn.embedding, (vector.embedding for vector in rivals))
    if rival_scores and score - max(rival_scores) < BACKFILL_MARGIN:
        return None
    return score
