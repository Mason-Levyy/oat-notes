"""Unit-normalised speaker embeddings and how they are compared."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def normalize(embedding: np.ndarray) -> np.ndarray:
    """A contiguous float32 unit vector; raises ValueError when there is no
    direction to keep (zero, NaN or infinite norm)."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("speaker embedding must have a finite non-zero norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def cosine(embedding: np.ndarray, other: np.ndarray) -> float | None:
    """Dot product of two unit vectors, or None when their shapes differ
    (a profile built by a different model)."""
    if embedding.shape != other.shape:
        return None
    return float(np.dot(embedding, other))


def cosine_scores(embedding: np.ndarray, others: Iterable[np.ndarray]) -> list[float]:
    scores = (cosine(embedding, other) for other in others)
    return [score for score in scores if score is not None]
