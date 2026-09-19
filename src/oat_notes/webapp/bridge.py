"""Turns Session callbacks into browser events and LineFeed updates.

Every method here runs on a pipeline, cleanup or hotkey thread, some from
inside a call that already holds ``state.lock`` — so nothing here takes it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..output import format_timestamp
from ..session import SessionEvents
from ..types import AttributionSource, Channel, TranscriptSegment

if TYPE_CHECKING:
    from ..pipeline import ProfileLearningUpdate
    from .state import AppState


class SessionBridge:
    def __init__(self, state: AppState) -> None:
        self._state = state

    def events(self) -> SessionEvents:
        return SessionEvents(
            on_segment=self.on_segment,
            on_line_cleaned=self.on_line_cleaned,
            on_line_relabelled=self.on_line_relabelled,
            on_speaker=self.on_speaker,
            on_attribution=self.on_attribution,
            on_hotkey_bank=self.on_hotkey_bank,
            on_profile_learning=self.on_profile_learning,
            on_note=self.on_note,
        )

    def on_segment(
        self, segment: TranscriptSegment, label: str, latency: float, line_id: int
    ) -> None:
        session = self._state.session
        line = {
            "id": line_id,
            "time": format_timestamp(segment.start),
            "label": label,
            "text": segment.text,
            "channel": segment.channel.value,
            "speaker_id": segment.speaker_id,
            "speaker_index": segment.speaker_index,
            "active_speaker": (
                session.current_speaker if session is not None else segment.speaker_index
            ),
            "attribution": segment.attribution,
            "confidence": segment.confidence,
        }
        self._state.lines.append(line)
        self._state.hub.publish({"type": "line", **line})

    def on_line_cleaned(self, line_id: int, text: str | None) -> None:
        if text is None:
            if self._state.lines.drop(line_id):
                self._state.hub.publish({"type": "line_drop", "id": line_id})
            return
        change = self._state.lines.update(line_id, text=text)
        if change is not None:
            before, updated = change
            self._state.hub.publish(
                {"type": "line_update", **updated, "original": before["text"]}
            )

    def on_line_relabelled(self, line_id: int, name: str, index: int, score: float) -> None:
        self.relabel(line_id, name, index, "auto", score)

    def relabel(
        self,
        line_id: int,
        name: str,
        index: int | None,
        source: AttributionSource,
        confidence: float | None,
    ) -> None:
        """Re-render one transcript line under a new speaker, riding the same
        line_update event the cleanup pass already uses."""
        change = self._state.lines.update(
            line_id, label=name, speaker_index=index, attribution=source, confidence=confidence
        )
        if change is not None:
            self._state.hub.publish({"type": "line_update", **change[1]})

    def on_speaker(self, index: int) -> None:
        session = self._state.session
        if session is None or index >= len(session.roster):
            return
        if len(session.roster) != len(self._state.roster):
            self._state.sync_roster(session)
            self._state.publish_status()
        self._state.hub.publish({"type": "speaker", "index": index, "source": "manual"})

    def on_attribution(
        self,
        index: int | None,
        channel: Channel,
        source: AttributionSource,
        confidence: float | None,
    ) -> None:
        session = self._state.session
        self._state.hub.publish(
            {
                "type": "speaker",
                "index": index,
                "active_index": session.current_speaker if session is not None else index,
                "channel": channel.value,
                "source": source,
                "confidence": confidence,
            }
        )
        self._state.publish_status()

    def on_hotkey_bank(self, bank: int) -> None:
        self._state.hub.publish({"type": "hotkey_bank", "bank": bank})

    def on_profile_learning(self, update: ProfileLearningUpdate) -> None:
        self._state.hub.publish({"type": "profile_learning", **update.to_dict()})

    def on_note(self, note: dict) -> None:
        self._state.lines.add_note(note)
        self._state.hub.publish({"type": "note", **note})
