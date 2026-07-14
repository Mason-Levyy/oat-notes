"""Global modifier+digit speaker switching, active only during a session.

The chord works from any focused application and never collides with plain
number typing. Digits beyond the current roster are forwarded too — the
session turns them into auto-added guests.
"""

from __future__ import annotations

from typing import Callable

_NUMPAD_ONE_VK = 97
_NUMPAD_NINE_VK = 105


class HotkeyListener:
    def __init__(
        self,
        on_switch: Callable[[int], None],
        modifiers: tuple[str, ...] = ("ctrl", "alt"),
        on_page: Callable[[int], None] = lambda delta: None,
    ) -> None:
        allowed = {"ctrl", "alt", "shift", "win"}
        if not modifiers or not set(modifiers).issubset(allowed):
            raise ValueError("modifiers must contain ctrl, alt, shift, and/or win")
        self._on_switch = on_switch
        self._on_page = on_page
        self._required_modifiers = frozenset(modifiers)
        self._listener = None
        self._held_modifiers: set[str] = set()

    def start(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._handle_press, on_release=self._handle_release
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _handle_press(self, key) -> None:
        modifier = self._modifier_name(key)
        if modifier is not None:
            self._held_modifiers.add(modifier)
            return
        if self._held_modifiers != self._required_modifiers:
            return
        index = self._digit_index(key)
        if index is not None:
            self._on_switch(index)
            return
        direction = self._page_direction(key)
        if direction is not None:
            self._on_page(direction)

    def _handle_release(self, key) -> None:
        modifier = self._modifier_name(key)
        if modifier is not None:
            self._held_modifiers.discard(modifier)

    @staticmethod
    def _modifier_name(key) -> str | None:
        name = getattr(key, "name", "")
        if name.startswith("ctrl"):
            return "ctrl"
        if name.startswith("alt"):
            return "alt"
        if name.startswith("shift"):
            return "shift"
        if name.startswith("cmd") or name.startswith("win"):
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
        if virtual_key == 219:  # OEM_4: [ on a US Windows keyboard
            return -1
        if virtual_key == 221:  # OEM_6: ] on a US Windows keyboard
            return 1
        return None
