"""The meeting endpoints: start, stop, finalize, the live roster, and the
transcript lines and notes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..attribution import Speaker
from ..session import Session, SessionOptions
from .bridge import SessionBridge
from .requests import (
    Body,
    bad_request,
    conflict,
    not_found,
    parse_direction,
    parse_flag,
    parse_int,
    parse_object,
    parse_optional_id,
    parse_renames,
    parse_roster,
    parse_text,
)
from .state import AppState, guests_with_lines

if TYPE_CHECKING:
    from ..speaker_store import SpeakerStore


def _library_name(store: SpeakerStore | None, speaker_id: str | None) -> str | None:
    """The saved name behind a library id; None when no id was given."""
    if speaker_id is None or store is None:
        return None
    profile = store.find(speaker_id)
    if profile is None:
        raise not_found("speaker not found")
    return profile.name


def _roster_from(state: AppState, body: Body) -> tuple[Speaker, ...]:
    roster = []
    for entry in parse_roster(body):
        name = entry.name
        if entry.speaker_id and state.speaker_store is not None:
            profile = state.speaker_store.find(entry.speaker_id)
            if profile is None:
                raise not_found("saved speaker not found")
            name = profile.name
        if name:
            roster.append(Speaker(name=name, speaker_id=entry.speaker_id, hotkey_slot=entry.hotkey_slot))
    return tuple(roster)


def start_session(state: AppState, body: Body) -> dict:
    with state.lock:
        if state.session is not None:
            raise conflict("already recording")
        if state.transcriber is None:
            if state.model_status == "error":
                raise conflict(state.model_error or "model failed to load")
            raise conflict("transcription model is still loading")
        state.roster = _roster_from(state, body)
        state.meeting_name = parse_text(body, "name") or "meeting"
        state.lines.reset()
        state.last_saved = None
        state.awaiting_backfill = None
        options = SessionOptions(
            speakers=state.roster,
            meeting_name=state.meeting_name,
            use_loopback=parse_flag(body, "loopback", True),
            out_dir=state.out_dir,
            hotkey_modifiers=state.settings.hotkey_modifiers,
            hotkey_listener=state.hotkey_listener,
            speaker_store=state.speaker_store,
            embedding_engine=state.embedding_engine,
            tracking_embedding_engine=state.tracking_embedding_engine,
            cleaner=state.cleaner if parse_flag(body, "cleanup", True) else None,
        )
        state.session = Session(
            state.config, options, state.transcriber, SessionBridge(state).events()
        )
        state.session.start()
    state.publish_status()
    return state.status()


def stop_session(state: AppState, body: Body) -> dict:
    discard = parse_flag(body, "discard", False)
    with state.lock:
        session = state.require_session()
        state.session = None
    session.stop()
    needs_backfill = not discard and bool(guests_with_lines(session)) and not session.log.is_empty
    with state.lock:
        if needs_backfill:
            state.awaiting_backfill = session
        elif discard:
            session.discard()
            state.last_saved = None
        else:
            saved = session.save()
            state.last_saved = str(saved) if saved else None
    state.publish_status()
    return state.status()


def finalize(state: AppState, body: Body) -> dict:
    renames = parse_renames(body)
    guest_profiles = parse_object(body, "guest_profiles")
    with state.lock:
        session = state.awaiting_backfill
        if session is None:
            raise conflict("nothing awaiting backfill")
        if state.speaker_store is not None:
            try:
                for guest_name, selection in guest_profiles.items():
                    if not isinstance(selection, dict):
                        continue
                    speaker_id = parse_optional_id(selection)
                    if speaker_id:
                        profile = state.speaker_store.profile(speaker_id)
                    else:
                        new_name = parse_text(selection, "name")
                        if not new_name:
                            continue
                        profile = state.speaker_store.create_speaker(new_name)
                    session.persist_guest(str(guest_name), profile.speaker_id)
                    renames[str(guest_name)] = profile.name
            except (KeyError, ValueError) as error:
                raise conflict(str(error).strip("'")) from error
        state.awaiting_backfill = None
        saved = session.save(renames)
        state.last_saved = str(saved) if saved else None
        state.lines.relabel(renames)
    state.publish_status()
    return state.status()


def add_guest(state: AppState, body: Body) -> dict:
    with state.lock:
        session = state.require_session()
        session.add_guest()
        state.sync_roster(session)
    state.publish_status()
    return state.status()


def add_roster_speaker(state: AppState, body: Body) -> dict:
    with state.lock:
        session = state.require_session()
        speaker_id = parse_optional_id(body)
        name = _library_name(state.speaker_store, speaker_id) or parse_text(body, "name")
        if not name:
            raise bad_request("a name is required")
        session.add_speaker(name, speaker_id)
        state.sync_roster(session)
    state.publish_status()
    return state.status()


def remove_roster_speaker(state: AppState, body: Body) -> dict:
    index = parse_int(body, "index")
    with state.lock:
        session = state.require_session()
        error = session.remove_speaker(index)
        if error:
            raise conflict(error)
        state.sync_roster(session)
    state.publish_status()
    return state.status()


def rename_roster_speaker(state: AppState, body: Body) -> dict:
    index = parse_int(body, "index")
    with state.lock:
        session = state.require_session()
        old = session.roster[index].name if 0 <= index < len(session.roster) else None
        speaker_id = parse_optional_id(body)
        _library_name(state.speaker_store, speaker_id)
        error = session.rename_speaker(index, str(body.get("name") or ""), speaker_id)
        if error:
            raise conflict(error)
        new_name = session.roster[index].name
        if old is not None and old != new_name:
            state.lines.relabel({old: new_name})
        state.sync_roster(session)
    state.publish_status()
    return state.status()


def assign_line(state: AppState, body: Body) -> dict:
    """Put a transcript line on a person by hand: a roster ``index``, or a
    ``name`` (with an optional library ``speaker_id``) for someone who is not
    on the roster yet. Recording only: once the meeting stops the session —
    and with it the log this writes through to — is gone."""
    line_id = parse_int(body, "id")
    with state.lock:
        session = state.require_session()
        if "name" in body:
            speaker_id = parse_optional_id(body)
            name = _library_name(state.speaker_store, speaker_id) or str(body.get("name") or "")
            index, error = session.assign_line_to_name(line_id, name, speaker_id)
            if error:
                raise conflict(error)
            state.sync_roster(session)
            state.publish_status()
        else:
            index = parse_int(body, "index", optional=True)
            error = session.assign_line(line_id, index)
            if error:
                raise conflict(error)
        name = "Unknown" if index is None else session.roster[index].name
        SessionBridge(state).relabel(line_id, name, index, "manual", None)
    return state.status()


def reset_roster_profile(state: AppState, body: Body) -> dict:
    """The in-meeting counterpart to the library reset, which is refused
    while a session is live."""
    index = parse_int(body, "index")
    with state.lock:
        session = state.require_session()
        error = session.reset_speaker_profile(index)
        if error:
            raise conflict(error)
    state.publish_status()
    return state.status()


def cancel_profile_learning(state: AppState, body: Body) -> dict:
    with state.lock:
        session = state.require_session()
    session.cancel_profile_learning()
    return state.status()


def add_note(state: AppState, body: Body) -> dict:
    text = parse_text(body, "text", required_message="note text is required")
    with state.lock:
        session = state.require_session()
        session.add_note(text)
    return state.status()


def switch_speaker(state: AppState, body: Body) -> dict:
    index = parse_int(body, "index")
    with state.lock:
        session = state.session
    if session is not None:
        session.switch_speaker(index)
    return state.status()


def page_hotkeys(state: AppState, body: Body) -> dict:
    direction = parse_direction(body)
    with state.lock:
        session = state.session
    if session is not None:
        session.page_hotkeys(direction)
    return state.status()
