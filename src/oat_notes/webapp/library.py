"""The saved-speaker library: people, groups, and recording a voice sample
for one of them outside a meeting."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..speaker_store import SpeakerStore
from .requests import Body, conflict, not_found, parse_id, parse_members, parse_text
from .state import AppState

if TYPE_CHECKING:
    from ..enrollment import EnrollmentProgress

TERMINAL_PHASES = ("done", "stopped", "error")


def _mutate(state: AppState, operation: Callable[[SpeakerStore], object]) -> dict:
    with state.lock:
        if state.session is not None:
            raise conflict("end the meeting before changing the speaker library")
        if state.speaker_store is None:
            raise conflict("speaker library is unavailable")
        try:
            operation(state.speaker_store)
        except KeyError as error:
            raise not_found(str(error).strip("'")) from error
        except ValueError as error:
            raise conflict(str(error)) from error
    state.publish_status()
    return state.status()


def create_speaker(state: AppState, body: Body) -> dict:
    name = parse_text(body, "name", required_message="a name is required")
    return _mutate(state, lambda store: store.create_speaker(name))


def update_speaker(state: AppState, body: Body) -> dict:
    speaker_id = parse_id(body)
    name = parse_text(body, "name", required_message="a name is required")
    return _mutate(state, lambda store: store.rename_speaker(speaker_id, name))


def delete_speaker(state: AppState, body: Body) -> dict:
    speaker_id = parse_id(body)
    return _mutate(state, lambda store: store.delete_speaker(speaker_id))


def reset_speaker(state: AppState, body: Body) -> dict:
    speaker_id = parse_id(body)
    return _mutate(state, lambda store: store.reset_profile(speaker_id))


def create_group(state: AppState, body: Body) -> dict:
    name = parse_text(body, "name")
    members = parse_members(body)
    return _mutate(state, lambda store: store.create_group(name, members))


def update_group(state: AppState, body: Body) -> dict:
    group_id = parse_id(body)
    name = parse_text(body, "name")
    members = parse_members(body)
    return _mutate(state, lambda store: store.update_group(group_id, name, members))


def delete_group(state: AppState, body: Body) -> dict:
    group_id = parse_id(body)
    return _mutate(state, lambda store: store.delete_group(group_id))


def start_voice_recording(state: AppState, body: Body) -> dict:
    speaker_id = parse_id(body)
    with state.lock:
        if state.session is not None:
            raise conflict("end the meeting before recording a voice sample")
        if state.speaker_store is None or state.embedding_engine is None:
            raise conflict("speaker recognition is unavailable")
        if state.enrollment is not None:
            raise conflict("a voice recording is already in progress")
        profile = state.speaker_store.find(speaker_id)
        if profile is None:
            raise not_found("speaker not found")

        from ..enrollment import VoiceEnrollmentRecorder

        def on_progress(progress: EnrollmentProgress) -> None:
            terminal = progress.phase in TERMINAL_PHASES
            event = {
                "type": "enrollment",
                "speaker_id": speaker_id,
                "name": profile.name,
                "phase": progress.phase,
                "captured_seconds": round(progress.captured_seconds, 2),
                "target_seconds": progress.target_seconds,
                "reason": progress.reason,
            }
            with state.lock:
                state.enrollment_event = None if terminal else event
                if terminal:
                    state.enrollment = None
            state.hub.publish(event)

        state.enrollment = VoiceEnrollmentRecorder(
            state.speaker_store, state.embedding_engine, speaker_id, state.config, on_progress
        )
        state.enrollment.start()
    return {"ok": True, "speaker_id": speaker_id}


def stop_voice_recording(state: AppState, body: Body) -> dict:
    with state.lock:
        if state.enrollment is None:
            raise conflict("no voice recording in progress")
        state.enrollment.stop()
    return {"ok": True}
