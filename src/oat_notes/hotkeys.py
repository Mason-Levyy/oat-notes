"""Global hotkey chords for meeting speaker switching and dictation.

One process-wide pynput hook drives every binding. Chords match on an exact
modifier set. A chord with no key arms while its modifiers are held and
commits when they are released; pressing any other key cancels it, which is
what stops Windows' own Ctrl+Win+D and Ctrl+Win+arrow shortcuts from starting
a recording.

Callbacks run on the hook thread, which sits in the path of every keystroke
on the machine — keep them quick.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from . import inject
from .inject import VK_NONAME
from .log import error_kind
from .settings import MODIFIER_ORDER as MODIFIERS

log = logging.getLogger(__name__)

DIGIT = "digit"
BRACKET = "bracket"

_NUMPAD_ONE_VK = 97
_NUMPAD_NINE_VK = 105
_VK_OEM_4_LEFT_BRACKET = 219
_VK_OEM_6_RIGHT_BRACKET = 221
_PRESS_MESSAGES = (0x0100, 0x0104)
_RELEASE_MESSAGES = (0x0101, 0x0105)
_WINDOWS_KEYS = (inject.VK_LWIN, inject.VK_RWIN)
_MODIFIER_VKS = {
    0x10: "shift", 0xA0: "shift", 0xA1: "shift",
    0x11: "ctrl", 0xA2: "ctrl", 0xA3: "ctrl",
    0x12: "alt", 0xA4: "alt", 0xA5: "alt",
    inject.VK_LWIN: "win", inject.VK_RWIN: "win",
}


@dataclass(frozen=True)
class Chord:
    """A modifier set plus an optional key: ``DIGIT``, ``BRACKET``, or a
    pynput key name such as ``"esc"``. No key means a modifier-only chord."""

    modifiers: frozenset[str]
    key: str | None = None

    @classmethod
    def of(cls, modifiers, key: str | None = None) -> Chord:
        requested = frozenset(modifiers)
        unknown = requested.difference(MODIFIERS)
        if unknown:
            raise ValueError(f"unknown hotkey modifier: {min(unknown)}")
        if not requested and key is None:
            raise ValueError("a chord needs at least one modifier or a key")
        return cls(requested, key)

    @property
    def is_modifier_only(self) -> bool:
        return self.key is None


@dataclass
class _Binding:
    chord: Chord
    on_press: Callable[[], object]
    on_release: Callable[[float], object] | None = None
    on_cancel: Callable[[], object] | None = None
    armed_at: float | None = None
    cancelled: bool = False


class HotkeyListener:
    """The process-wide hook. Constructing it with ``on_switch`` registers the
    meeting speaker-switch bindings directly."""

    def __init__(
        self,
        on_switch: Callable[[int], None] | None = None,
        modifiers: tuple[str, ...] = ("ctrl", "alt"),
        on_page: Callable[[int], None] = lambda delta: None,
    ) -> None:
        self._listener = None
        self._held: set[str] = set()
        self._physical_modifiers: set[int] = set()
        self._consumed_windows_keys: set[int] = set()
        self._bindings: dict[str, _Binding] = {}
        self._lock = threading.Lock()
        if on_switch is not None:
            if not modifiers or not set(modifiers).issubset(set(MODIFIERS)):
                raise ValueError(
                    "modifiers must contain ctrl, alt, shift, and/or win"
                )
            self.bind_speaker_switch(modifiers, on_switch, on_page)

    def start(self) -> None:
        if self._listener is not None:
            return
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._handle_press, on_release=self._handle_release,
            win32_event_filter=self._filter_windows_event,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._held.clear()
        self._physical_modifiers.clear()
        self._consumed_windows_keys.clear()

    def bind(
        self,
        name: str,
        chord: Chord,
        on_press: Callable[[], object],
        on_release: Callable[[float], object] | None = None,
        on_cancel: Callable[[], object] | None = None,
    ) -> None:
        """Register or replace a binding. ``on_release`` receives how long the
        chord was held, so the caller can tell a tap from a hold. Timed with
        perf_counter because monotonic() only ticks every ~15.6 ms on Windows,
        which is coarse against a 400 ms tap threshold."""
        with self._lock:
            self._bindings[name] = _Binding(chord, on_press, on_release, on_cancel)

    def unbind(self, name: str) -> None:
        with self._lock:
            self._bindings.pop(name, None)

    def bind_speaker_switch(
        self,
        modifiers,
        on_switch: Callable[[int], None],
        on_page: Callable[[int], None] = lambda delta: None,
    ) -> None:
        self.bind("speaker-switch", Chord.of(modifiers, DIGIT), on_switch)
        self.bind("speaker-page", Chord.of(modifiers, BRACKET), on_page)

    def unbind_speaker_switch(self) -> None:
        self.unbind("speaker-switch")
        self.unbind("speaker-page")

    def _snapshot(self) -> tuple[_Binding, ...]:
        """bind and unbind run on the HTTP thread while the hook thread is
        dispatching, so dispatch iterates a copy."""
        with self._lock:
            return tuple(self._bindings.values())

    def _filter_windows_event(self, message, data) -> bool:
        """Mask a dictation's Win keyup before Windows can open Start.

        Raw modifier state is tracked here because pynput posts the ordinary
        callbacks after the hook returns. Even a quick tap must be recognised
        before its release reaches Windows. Our other injected input passes
        to Windows without arming or cancelling physical hotkey chords.
        """
        if data.dwExtraInfo == inject.INPUT_TAG:
            return False
        if data.dwExtraInfo == inject.WINDOWS_RELEASE_TAG:
            return True
        vk = data.vkCode
        if message in _PRESS_MESSAGES:
            if vk == VK_NONAME:
                return True
            if vk not in _MODIFIER_VKS:
                self._consumed_windows_keys.clear()
                return True
            self._physical_modifiers.add(vk)
            held = {_MODIFIER_VKS[key] for key in self._physical_modifiers}
            if any(
                binding.chord.is_modifier_only
                and "win" in binding.chord.modifiers
                and held == binding.chord.modifiers
                for binding in self._snapshot()
            ):
                self._consumed_windows_keys.update(
                    self._physical_modifiers.intersection(_WINDOWS_KEYS)
                )
        elif message in _RELEASE_MESSAGES:
            self._physical_modifiers.discard(vk)
            if vk in self._consumed_windows_keys:
                self._consumed_windows_keys.discard(vk)
                listener = self._listener
                if listener is None:
                    return True
                try:
                    inject.release_windows_key(vk)
                except Exception as error:
                    log.warning("could not mask the Windows key release: %s", error_kind(error))
                    return True
                listener.suppress_event()
        return True

    def _handle_press(self, key) -> None:
        if getattr(key, "vk", None) == VK_NONAME:
            return
        modifier = self._modifier_name(key)
        if modifier is not None:
            if modifier in self._held:
                return
            self._held.add(modifier)
            self._arm_modifier_chords()
            return
        self._cancel_armed()
        self._fire_key_chords(key)

    def _handle_release(self, key) -> None:
        modifier = self._modifier_name(key)
        if modifier is None:
            return
        self._held.discard(modifier)
        for binding in self._snapshot():
            if binding.armed_at is None:
                continue
            if self._held == binding.chord.modifiers:
                continue
            held_seconds = time.perf_counter() - binding.armed_at
            binding.armed_at = None
            if binding.cancelled:
                binding.cancelled = False
                continue
            if binding.on_release is not None:
                self._safely(binding.on_release, held_seconds)

    def _arm_modifier_chords(self) -> None:
        for binding in self._snapshot():
            if not binding.chord.is_modifier_only or binding.armed_at is not None:
                continue
            if self._held != binding.chord.modifiers:
                continue
            binding.armed_at = time.perf_counter()
            binding.cancelled = False
            self._safely(binding.on_press)

    def _cancel_armed(self) -> None:
        for binding in self._snapshot():
            if binding.armed_at is None or binding.cancelled:
                continue
            binding.cancelled = True
            if binding.on_cancel is not None:
                self._safely(binding.on_cancel)

    def _fire_key_chords(self, key) -> None:
        for binding in self._snapshot():
            chord = binding.chord
            if chord.is_modifier_only or self._held != chord.modifiers:
                continue
            matched, arguments = self._match_key(chord.key, key)
            if matched:
                self._safely(binding.on_press, *arguments)

    @classmethod
    def _match_key(cls, wanted: str | None, key) -> tuple[bool, tuple]:
        if wanted == DIGIT:
            index = cls._digit_index(key)
            return (False, ()) if index is None else (True, (index,))
        if wanted == BRACKET:
            direction = cls._page_direction(key)
            return (False, ()) if direction is None else (True, (direction,))
        if wanted and len(wanted) == 1 and wanted.isalpha():
            return (cls._is_letter(key, wanted), ())
        return (getattr(key, "name", "") == wanted, ())

    @staticmethod
    def _is_letter(key, letter: str) -> bool:
        char = getattr(key, "char", None)
        if char and char.lower() == letter.lower():
            return True
        # Ctrl+letter on Windows reports a control character (\x1a for Z), the
        # same problem _digit_index solves — fall back to the virtual key.
        virtual_key = getattr(key, "vk", None)
        return virtual_key is not None and virtual_key == ord(letter.upper())

    @staticmethod
    def _safely(callback: Callable, *arguments) -> None:
        """An exception escaping a pynput callback kills the hook, taking every
        other binding with it."""
        try:
            callback(*arguments)
        except Exception as error:
            log.error("hotkey callback failed: %s", error_kind(error))

    @staticmethod
    def _modifier_name(key) -> str | None:
        name = getattr(key, "name", "")
        if name.startswith("ctrl"):
            return "ctrl"
        if name.startswith("alt"):
            return "alt"
        if name.startswith("shift"):
            return "shift"
        if name.startswith(("cmd", "win")):
            return "win"
        return None

    @staticmethod
    def _digit_index(key) -> int | None:
        virtual_key = getattr(key, "vk", None)
        if virtual_key is not None and _NUMPAD_ONE_VK <= virtual_key <= _NUMPAD_NINE_VK:
            return virtual_key - _NUMPAD_ONE_VK
        char = getattr(key, "char", None)
        if char and char.isdigit() and char != "0":
            return int(char) - 1
        # Ctrl+Alt+digit on Windows often reports a control character
        # (\x11 for 1, etc.) or the raw vk instead of the digit char.
        if virtual_key is not None and ord("1") <= virtual_key <= ord("9"):
            return virtual_key - ord("1")
        return None

    @staticmethod
    def _page_direction(key) -> int | None:
        char = getattr(key, "char", None)
        if char == "[":
            return -1
        if char == "]":
            return 1
        virtual_key = getattr(key, "vk", None)
        if virtual_key == _VK_OEM_4_LEFT_BRACKET:
            return -1
        if virtual_key == _VK_OEM_6_RIGHT_BRACKET:
            return 1
        return None
