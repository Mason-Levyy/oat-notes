"""Process start-up for ``--ui``: the single-instance guard, the HTTP server,
the overlay, and loading the inference models off the UI thread."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser
from dataclasses import replace
from http.server import ThreadingHTTPServer

from ..config import Config
from ..dictation.controller import DictationController
from ..hotkeys import HotkeyListener
from ..log import error_kind
from ..output import recover_journals
from ..overlay import Overlay
from ..settings import SettingsStore
from ..single_instance import SingleInstance
from ..speaker_store import SpeakerStore
from . import meeting
from .handler import Handler
from .hub import EventHub
from .state import AppState, dictation_options

log = logging.getLogger(__name__)

MODEL_LOAD_DELAY_SECONDS = 0.75
EXIT_BACKSTOP_SECONDS = 5.0
PORT_IN_USE_WINDOWS = 10048
PORT_IN_USE_POSIX = 98


def load_transcriber(state: AppState, config: Config) -> None:
    """Build and warm the transcription model, publishing the outcome. Called
    once at startup and again whenever Settings picks a different model."""
    started = time.perf_counter()
    try:
        from ..cli import warm_up
        from ..transcriber import create_transcriber

        print(f"Loading transcription model ({config.model_name})…", flush=True)
        transcriber = create_transcriber(config)
        warm_up(config, transcriber)
    except Exception as error:
        elapsed = time.perf_counter() - started
        log.error("model initialization failed after %.1fs: %s", elapsed, error)
        state.model_failed(error, elapsed)
    else:
        elapsed = time.perf_counter() - started
        print(f"Transcription model ready in {elapsed:.1f}s", flush=True)
        state.model_ready(transcriber, elapsed)


def _load_speaker_model(state: AppState) -> None:
    try:
        from ..speaker_id import SherpaOnnxEmbeddingEngine

        engine = SherpaOnnxEmbeddingEngine()
    except Exception as error:
        log.error("speaker recognition unavailable: %s: %s", error_kind(error), error)
        state.speaker_model_failed(error)
        return
    tracking_engine = engine
    try:
        tracking_engine = SherpaOnnxEmbeddingEngine()
    except Exception as error:
        log.warning(
            "live speaker tracking will share the main speaker model"
            " (second instance failed: %s: %s)",
            error_kind(error),
            error,
        )
    print("Local speaker recognition ready", flush=True)
    state.speaker_model_ready(engine, tracking_engine)


def _load_cleaner(state: AppState, config: Config) -> None:
    if not config.cleanup_enabled:
        state.cleaner_unavailable("disabled")
        return
    try:
        from ..cleanup import TranscriptCleaner
        from ..llm import LlmEngine, model_is_cached

        if not config.offline and not model_is_cached(config):
            print(f"Downloading language model ({config.cleanup_model})…", flush=True)
            state.cleaner_downloading()
        else:
            print(f"Loading language model ({config.cleanup_model})…", flush=True)
        engine = LlmEngine.from_config(config)
        cleaner = TranscriptCleaner(engine)
        cleaner.warm_up()
    except ImportError:
        log.warning(
            "transcript cleanup unavailable: install the openvino extra"
            " (uv sync --extra openvino)"
        )
        state.cleaner_unavailable("openvino extra not installed")
    except Exception as error:
        log.error("transcript cleanup unavailable: %s: %s", error_kind(error), error)
        state.cleaner_failed(error)
    else:
        print(f"Language model ready on {engine.device}", flush=True)
        if state.dictation is not None:
            state.dictation.set_engine(engine)
        state.cleaner_ready(cleaner)


def initialize_models(state: AppState, config: Config) -> None:
    """Load every local inference engine off the UI thread."""
    load_transcriber(state, config)
    _load_speaker_model(state)
    _load_cleaner(state, config)


def exit_backstop(seconds: float = EXIT_BACKSTOP_SECONDS) -> threading.Timer:
    """Guarantee the process actually dies after an orderly shutdown.

    Teardown stops the captures, drains the pipeline and closes the server, but
    the native audio and inference libraries underneath can leave a thread that
    outlives the interpreter. A process that lingers keeps its low-level
    keyboard hook installed, so the next launch runs two — and the stale one
    still answers on its port until it finally releases it.

    Nothing is masked: this fires only if the normal exit has not happened,
    and stderr is flushed first because a windowed build's stderr is the log.
    """

    def bail() -> None:
        sys.stderr.flush()
        os._exit(0)

    timer = threading.Timer(seconds, bail)
    timer.name = "exit-backstop"
    timer.daemon = True
    timer.start()
    return timer


def await_shutdown(state: AppState, overlay: Overlay, show_overlay: bool = True) -> None:
    """Block the main thread until the app is told to quit.

    Tk insists on owning the thread it was created on, so when the overlay is
    enabled the main thread becomes its event loop and polls ``shutdown``
    from inside. Without it — headless runs, or a build with no tkinter —
    this stays the plain wait it always was.
    """
    if show_overlay:
        try:
            overlay.run(state.shutdown)
            return
        except Exception as error:
            log.warning(
                "overlay unavailable, continuing without it: %s: %s", error_kind(error), error
            )
    while not state.shutdown.wait(timeout=0.3):
        pass


class DictationPhaseRelay:
    """Every phase reaches the overlay; the browser only hears changes."""

    def __init__(self, overlay: Overlay, hub: EventHub) -> None:
        self._overlay = overlay
        self._hub = hub
        self._last_phase = ""

    def __call__(self, phase: str, details: dict) -> None:
        self._overlay.post_phase(phase, details)
        if self._last_phase != phase:
            self._last_phase = phase
            self._hub.publish({"type": "dictation", "phase": phase, **details})


def _bind_port(port: int) -> ThreadingHTTPServer | None:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as error:
        if getattr(error, "winerror", None) == PORT_IN_USE_WINDOWS or error.errno == PORT_IN_USE_POSIX:
            return None
        raise
    server.daemon_threads = True
    return server


def serve(config: Config, args: argparse.Namespace, open_browser: bool = True) -> None:
    started = time.perf_counter()
    # Before anything installs a keyboard hook or opens the microphone: a
    # second instance arms on the same chord and pastes the same dictation
    # again.
    instance = SingleInstance()
    url = f"http://127.0.0.1:{args.port}"
    running_url = instance.acquire(args.port)
    if running_url is not None:
        _already_running(running_url, open_browser)
        return
    settings_store = SettingsStore()
    settings = settings_store.load()
    if getattr(args, "model", None) is None:
        config = replace(config, model_name=settings.whisper_model)
    hotkey_listener = HotkeyListener()
    shutdown = threading.Event()
    hub = EventHub()
    overlay = Overlay(on_quit=shutdown.set, on_open_ui=lambda: webbrowser.open(url))
    dictation = DictationController(
        config,
        options=dictation_options(settings),
        on_phase=DictationPhaseRelay(overlay, hub),
        level_sink=overlay.post_level,
    )
    state = AppState(
        config,
        transcriber=None,
        out_dir=args.out_dir,
        settings=settings,
        save_settings=settings_store.save,
        speaker_store=SpeakerStore(),
        hotkey_listener=hotkey_listener,
        dictation=dictation,
        shutdown=shutdown,
        hub=hub,
    )
    state.recovered = [str(path) for path in recover_journals(args.out_dir)]
    for path in state.recovered:
        print(f"Recovered an unsaved transcript from a previous run: {path}", flush=True)
    Handler.state = state
    Handler.port = args.port
    server = _bind_port(args.port)
    if server is None:
        _already_running(url, open_browser)
        return

    dictation.bind(hotkey_listener)
    dictation.start()
    hotkey_listener.start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"oat-notes UI ready in {time.perf_counter() - started:.2f}s: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    # OpenVINO construction can hold the interpreter lock briefly. Give the
    # browser enough time to fetch and paint the tiny UI before it begins.
    model_thread = threading.Timer(MODEL_LOAD_DELAY_SECONDS, initialize_models, args=(state, config))
    model_thread.name = "model-loader"
    model_thread.daemon = True
    model_thread.start()
    try:
        await_shutdown(state, overlay, show_overlay=settings.overlay_enabled and not args.no_overlay)
    except KeyboardInterrupt:
        log.info("interrupted")
    print("Shutting down…", flush=True)
    exit_backstop()
    instance.release()
    dictation.stop()
    hotkey_listener.stop()
    if state.session is not None:
        meeting.stop_session(state, {})
    server.shutdown()


def _already_running(url: str, open_browser: bool) -> None:
    print(f"oat-notes is already running at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
