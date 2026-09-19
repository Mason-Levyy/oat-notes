"""Saving settings, and the dictation endpoints that ride on them."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

from ..dictation.controller import HISTORY_LIMIT, DictationController
from ..log import error_kind
from .requests import Body, bad_request, conflict, not_found, parse_int
from .state import AppState

log = logging.getLogger(__name__)


def update_settings(state: AppState, body: Body) -> dict:
    try:
        settings = state.settings.merged(body)
    except (TypeError, ValueError) as error:
        raise bad_request(str(error)) from error
    with state.lock:
        changes_a_chord = settings.hotkey_modifiers != state.settings.hotkey_modifiers
        if state.session is not None and changes_a_chord:
            raise conflict("end the meeting before changing hotkeys")
        changes_the_model = settings.whisper_model != state.settings.whisper_model
        if state.session is not None and changes_the_model:
            raise conflict("end the meeting before changing the transcription model")
        try:
            if state.save_settings is not None:
                state.save_settings(settings)
        except OSError as error:
            raise conflict(f"could not save settings: {error}") from error
        state.settings = settings
        if changes_the_model:
            state.config = replace(state.config, model_name=settings.whisper_model)
            state.transcriber = None
            state.model_status = "loading"
            state.model_error = None
            state.model_load_seconds = None
    if changes_the_model:
        reload_transcriber(state)
    state.apply_dictation_settings()
    state.publish_status()
    return state.status()


def reload_transcriber(state: AppState) -> None:
    """Rebuild the transcription model off the request thread — loading it
    takes seconds, and a settings save must not block on that."""
    from .bootstrap import load_transcriber

    threading.Thread(
        target=load_transcriber, args=(state, state.config), name="model-reloader", daemon=True
    ).start()


def _require_dictation(state: AppState) -> DictationController:
    if state.dictation is None:
        raise conflict("dictation is unavailable")
    return state.dictation


def toggle_dictation(state: AppState, body: Body) -> dict:
    """Pause or resume the dictation chord without restarting the app."""
    _require_dictation(state)
    wanted = body.get("enabled")
    if wanted is None:
        wanted = not state.settings.dictation_enabled
    if not isinstance(wanted, bool):
        raise bad_request("enabled must be true or false")
    return update_settings(state, {"dictation_enabled": wanted})


def dictation_history(state: AppState) -> dict:
    if state.dictation is None:
        return {"entries": [], "limit": 0}
    return {"entries": state.dictation.history(), "limit": HISTORY_LIMIT}


def clear_dictation_history(state: AppState, body: Body) -> dict:
    dictation = _require_dictation(state)
    dictation.clear_history()
    state.hub.publish({"type": "dictation", "phase": dictation.phase})
    return dictation_history(state)


def copy_dictation(state: AppState, body: Body) -> dict:
    """Put a past dictation on the clipboard. Deliberately not a re-insert:
    the click comes from the browser, so the browser holds focus and the text
    would land there. Ctrl+Alt+Z is how you re-insert somewhere useful."""
    dictation = _require_dictation(state)
    entry_id = parse_int(body, "id")
    entry = next((item for item in dictation.history() if item["id"] == entry_id), None)
    if entry is None:
        raise not_found("that dictation is no longer in the history")
    from ..inject import write_clipboard_text

    try:
        write_clipboard_text(entry["text"])
    except Exception as error:
        raise conflict(f"could not copy: {error_kind(error)}") from error
    return dictation_history(state)
