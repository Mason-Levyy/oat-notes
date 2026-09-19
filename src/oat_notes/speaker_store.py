"""Local persistent speaker directory and voice-embedding store.

Only compact float32 embeddings and metadata are persisted.  Captured audio
never enters this database.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import normalize
from .paths import app_data_dir
from .types import Channel, ProfileState

CURRENT_SCHEMA_VERSION = 2
DEFAULT_MODEL_KEY = "3dspeaker-eres2net-en-voxceleb-v1"
READY_SPEECH_SECONDS = 5.0
MIN_SAMPLE_SPEECH_SECONDS = 1.0
MAX_SAMPLES_PER_SOURCE = 20


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    name: str
    enrollment_seconds: float
    state: ProfileState

    def to_dict(self) -> dict:
        return {
            "id": self.speaker_id,
            "name": self.name,
            "enrollment_seconds": round(self.enrollment_seconds, 2),
            "profile_state": self.state,
        }


@dataclass(frozen=True)
class MatchVector:
    speaker_id: str
    embedding: np.ndarray
    source_specific: bool


@dataclass(frozen=True)
class GroupMember:
    speaker_id: str
    name: str
    position: int

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "name": self.name,
            "position": self.position,
        }


@dataclass(frozen=True)
class SpeakerGroup:
    group_id: str
    name: str
    members: tuple[GroupMember, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.group_id,
            "name": self.name,
            "members": [member.to_dict() for member in self.members],
        }


def profile_state(speech_seconds: float) -> ProfileState:
    if speech_seconds <= 0:
        return "untrained"
    return "ready" if speech_seconds >= READY_SPEECH_SECONDS else "learning"


def _profile_from_row(row: sqlite3.Row) -> SpeakerProfile:
    seconds = float(row["seconds"])
    return SpeakerProfile(str(row["id"]), str(row["name"]), seconds, profile_state(seconds))


class SpeakerStore:
    """Small SQLite repository safe to call from UI and worker threads."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (app_data_dir() / "speakers.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > CURRENT_SCHEMA_VERSION:
                raise RuntimeError(
                    f"speaker database version {version} is newer than supported"
                )
            if version < 1:
                connection.executescript(
                    """
                    CREATE TABLE speakers (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE voice_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        speaker_id TEXT NOT NULL REFERENCES speakers(id)
                            ON DELETE CASCADE,
                        model_key TEXT NOT NULL,
                        source TEXT NOT NULL CHECK(source IN ('mic', 'loopback')),
                        speech_seconds REAL NOT NULL CHECK(speech_seconds > 0),
                        quality REAL NOT NULL,
                        dimension INTEGER NOT NULL,
                        embedding BLOB NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX voice_samples_lookup
                        ON voice_samples(speaker_id, model_key, source, id);
                    CREATE TABLE groups (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE group_members (
                        group_id TEXT NOT NULL REFERENCES groups(id)
                            ON DELETE CASCADE,
                        speaker_id TEXT NOT NULL REFERENCES speakers(id)
                            ON DELETE CASCADE,
                        position INTEGER NOT NULL CHECK(position >= 0),
                        PRIMARY KEY(group_id, speaker_id),
                        UNIQUE(group_id, position)
                    );
                    PRAGMA user_version = 2;
                    """
                )
            elif version < 2:
                connection.executescript(
                    """
                    CREATE TABLE group_members_v2 (
                        group_id TEXT NOT NULL REFERENCES groups(id)
                            ON DELETE CASCADE,
                        speaker_id TEXT NOT NULL REFERENCES speakers(id)
                            ON DELETE CASCADE,
                        position INTEGER NOT NULL CHECK(position >= 0),
                        PRIMARY KEY(group_id, speaker_id),
                        UNIQUE(group_id, position)
                    );
                    INSERT INTO group_members_v2(group_id, speaker_id, position)
                        SELECT group_id, speaker_id, position FROM group_members;
                    DROP TABLE group_members;
                    ALTER TABLE group_members_v2 RENAME TO group_members;
                    PRAGMA user_version = 2;
                    """
                )

    @staticmethod
    def _clean_name(name: str, kind: str = "speaker") -> str:
        cleaned = str(name).strip()
        if not cleaned:
            raise ValueError(f"{kind} name cannot be empty")
        if len(cleaned) > 120:
            raise ValueError(f"{kind} name must be 120 characters or fewer")
        return cleaned

    def create_speaker(self, name: str) -> SpeakerProfile:
        cleaned = self._clean_name(name)
        speaker_id = str(uuid.uuid4())
        timestamp = _now()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO speakers(id, name, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    (speaker_id, cleaned, timestamp, timestamp),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"speaker {cleaned!r} already exists") from error
        return self.profile(speaker_id)

    def find_by_name(self, name: str) -> SpeakerProfile | None:
        """The saved person with this name, ignoring case (the column's collation)."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM speakers WHERE name = ?", (name.strip(),)
            ).fetchone()
        return None if row is None else self.find(str(row["id"]))

    def rename_speaker(self, speaker_id: str, name: str) -> SpeakerProfile:
        cleaned = self._clean_name(name)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE speakers SET name = ?, updated_at = ? WHERE id = ?",
                    (cleaned, _now(), speaker_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError("speaker not found")
        except sqlite3.IntegrityError as error:
            raise ValueError(f"speaker {cleaned!r} already exists") from error
        return self.profile(speaker_id)

    def delete_speaker(self, speaker_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
            if cursor.rowcount != 1:
                raise KeyError("speaker not found")

    def reset_profile(self, speaker_id: str) -> SpeakerProfile:
        with self._connect() as connection:
            if not connection.execute(
                "SELECT 1 FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone():
                raise KeyError("speaker not found")
            connection.execute(
                "DELETE FROM voice_samples WHERE speaker_id = ?", (speaker_id,)
            )
        return self.profile(speaker_id)

    def profile(
        self, speaker_id: str, model_key: str = DEFAULT_MODEL_KEY
    ) -> SpeakerProfile:
        profile = self.find(speaker_id, model_key)
        if profile is None:
            raise KeyError("speaker not found")
        return profile

    def find(
        self, speaker_id: str, model_key: str = DEFAULT_MODEL_KEY
    ) -> SpeakerProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT s.id, s.name,
                       COALESCE(SUM(v.speech_seconds), 0.0) AS seconds
                FROM speakers s
                LEFT JOIN voice_samples v
                  ON v.speaker_id = s.id AND v.model_key = ?
                WHERE s.id = ?
                GROUP BY s.id, s.name
                """,
                (model_key, speaker_id),
            ).fetchone()
        return None if row is None else _profile_from_row(row)

    def list_speakers(
        self, model_key: str = DEFAULT_MODEL_KEY
    ) -> tuple[SpeakerProfile, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.id, s.name,
                       COALESCE(SUM(v.speech_seconds), 0.0) AS seconds
                FROM speakers s
                LEFT JOIN voice_samples v
                  ON v.speaker_id = s.id AND v.model_key = ?
                GROUP BY s.id, s.name
                ORDER BY s.name COLLATE NOCASE
                """,
                (model_key,),
            ).fetchall()
        return tuple(_profile_from_row(row) for row in rows)

    def add_sample(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        source: Channel,
        speech_seconds: float,
        quality: float,
        model_key: str = DEFAULT_MODEL_KEY,
    ) -> SpeakerProfile:
        if not np.isfinite(speech_seconds) or speech_seconds < MIN_SAMPLE_SPEECH_SECONDS:
            raise ValueError("voice samples require at least one second of speech")
        if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be between zero and one")
        vector = normalize(embedding)
        with self._connect() as connection:
            if not connection.execute(
                "SELECT 1 FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone():
                raise KeyError("speaker not found")
            connection.execute(
                """
                INSERT INTO voice_samples(
                    speaker_id, model_key, source, speech_seconds, quality,
                    dimension, embedding, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    speaker_id,
                    model_key,
                    source.value,
                    float(speech_seconds),
                    float(quality),
                    vector.size,
                    vector.tobytes(),
                    _now(),
                ),
            )
            connection.execute(
                """
                DELETE FROM voice_samples
                WHERE id IN (
                    SELECT id FROM voice_samples
                    WHERE speaker_id = ? AND source = ?
                    ORDER BY id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (speaker_id, source.value, MAX_SAMPLES_PER_SOURCE),
            )
        return self.profile(speaker_id, model_key)

    @staticmethod
    def _centroid(rows: Iterable[sqlite3.Row]) -> np.ndarray | None:
        vectors = []
        dimension = None
        for row in rows:
            row_dimension = int(row["dimension"])
            if dimension is None:
                dimension = row_dimension
            if row_dimension != dimension:
                continue
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size == row_dimension:
                vectors.append(vector)
        if not vectors:
            return None
        return normalize(np.mean(np.stack(vectors), axis=0))

    def match_vectors(
        self,
        speaker_ids: Iterable[str],
        source: Channel,
        model_key: str = DEFAULT_MODEL_KEY,
    ) -> tuple[MatchVector, ...]:
        candidates = []
        for speaker_id in dict.fromkeys(speaker_ids):
            profile = self.profile(speaker_id, model_key)
            if profile.state != "ready":
                continue
            vector = self.profile_vector(speaker_id, source, model_key)
            if vector is not None:
                candidates.append(vector)
        return tuple(candidates)

    def profile_vector(
        self,
        speaker_id: str,
        source: Channel,
        model_key: str = DEFAULT_MODEL_KEY,
    ) -> MatchVector | None:
        """Return a person's centroid even while the profile is learning."""
        with self._connect() as connection:
            source_rows = connection.execute(
                """
                SELECT dimension, embedding FROM voice_samples
                WHERE speaker_id = ? AND model_key = ? AND source = ?
                ORDER BY id
                """,
                (speaker_id, model_key, source.value),
            ).fetchall()
            centroid = self._centroid(source_rows)
            source_specific = centroid is not None
            if centroid is None:
                rows = connection.execute(
                    """
                    SELECT dimension, embedding FROM voice_samples
                    WHERE speaker_id = ? AND model_key = ?
                    ORDER BY id
                    """,
                    (speaker_id, model_key),
                ).fetchall()
                centroid = self._centroid(rows)
        if centroid is None:
            return None
        return MatchVector(speaker_id, centroid, source_specific)

    def create_group(self, name: str, members: Iterable[Mapping[str, Any]]) -> SpeakerGroup:
        group_id = str(uuid.uuid4())
        cleaned = self._clean_name(name, "group")
        timestamp = _now()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO groups(id, name, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    (group_id, cleaned, timestamp, timestamp),
                )
                self._replace_members(connection, group_id, members)
        except sqlite3.IntegrityError as error:
            raise ValueError(f"group {cleaned!r} already exists or has invalid members") from error
        return self.group(group_id)

    def update_group(
        self, group_id: str, name: str, members: Iterable[Mapping[str, Any]]
    ) -> SpeakerGroup:
        cleaned = self._clean_name(name, "group")
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE groups SET name = ?, updated_at = ? WHERE id = ?",
                    (cleaned, _now(), group_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError("group not found")
                connection.execute(
                    "DELETE FROM group_members WHERE group_id = ?", (group_id,)
                )
                self._replace_members(connection, group_id, members)
        except sqlite3.IntegrityError as error:
            raise ValueError(f"group {cleaned!r} already exists or has invalid members") from error
        return self.group(group_id)

    @staticmethod
    def _replace_members(
        connection: sqlite3.Connection, group_id: str, members: Iterable[Mapping[str, Any]]
    ) -> None:
        seen: set[str] = set()
        for position, member in enumerate(members):
            speaker_id = str(member.get("speaker_id", "")).strip()
            if not speaker_id or speaker_id in seen:
                raise ValueError("group members must be unique saved speakers")
            seen.add(speaker_id)
            connection.execute(
                """
                INSERT INTO group_members(group_id, speaker_id, position)
                VALUES (?, ?, ?)
                """,
                (group_id, speaker_id, position),
            )

    def delete_group(self, group_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM groups WHERE id = ?", (group_id,))
            if cursor.rowcount != 1:
                raise KeyError("group not found")

    def group(self, group_id: str) -> SpeakerGroup:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name FROM groups WHERE id = ?", (group_id,)
            ).fetchone()
            if row is None:
                raise KeyError("group not found")
            member_rows = connection.execute(
                """
                SELECT gm.speaker_id, s.name, gm.position
                FROM group_members gm
                JOIN speakers s ON s.id = gm.speaker_id
                WHERE gm.group_id = ?
                ORDER BY gm.position
                """,
                (group_id,),
            ).fetchall()
        return SpeakerGroup(
            str(row["id"]),
            str(row["name"]),
            tuple(
                GroupMember(
                    str(member["speaker_id"]),
                    str(member["name"]),
                    int(member["position"]),
                )
                for member in member_rows
            ),
        )

    def list_groups(self) -> tuple[SpeakerGroup, ...]:
        with self._connect() as connection:
            ids = [
                str(row["id"])
                for row in connection.execute(
                    "SELECT id FROM groups ORDER BY name COLLATE NOCASE"
                ).fetchall()
            ]
        return tuple(self.group(group_id) for group_id in ids)

    def library(self, model_key: str = DEFAULT_MODEL_KEY) -> dict:
        return {
            "speakers": [speaker.to_dict() for speaker in self.list_speakers(model_key)],
            "groups": [group.to_dict() for group in self.list_groups()],
            "ready_seconds": READY_SPEECH_SECONDS,
        }
