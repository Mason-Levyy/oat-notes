"""Global Ctrl+Alt+digit speaker-switch hotkeys, active only during a session.

The chord works from any focused application and never collides with plain
number typing. Digits beyond the current roster are forwarded too — the
session turns them into auto-added guests.
"""

from __future__ import annotations

from typing import Callable

_NUMPAD_ONE_VK = 97
_NUMPAD_NINE_VK = 105


class HotkeyListener:
    def __init__(self, on_switch: Callable[[int], None]) -> None:
        self._on_switch = on_switch
        self._listener = None
        self._ctrl_held = False
        self._alt_held = False

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
        if self._is_ctrl(key):
            self._ctrl_held = True
            return
        if self._is_alt(key):
            self._alt_held = True
            return
        if not (self._ctrl_held and self._alt_held):
            return
        index = self._digit_index(key)
        if index is not None:
            self._on_switch(index)

    def _handle_release(self, key) -> None:
        if self._is_ctrl(key):
            self._ctrl_held = False
        elif self._is_alt(key):
            self._alt_held = False

    @staticmethod
    def _is_ctrl(key) -> bool:
        name = getattr(key, "name", "")
        return name.startswith("ctrl")

    @staticmethod
    def _is_alt(key) -> bool:
        name = getattr(key, "name", "")
        return name.startswith("alt")

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
