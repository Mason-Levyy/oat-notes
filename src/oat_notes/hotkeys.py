"""Global speaker-switch hotkeys, active only while a session runs.

Plain number keys 1..N (top row or numpad) switch the active in-person
speaker — pressing them in any application counts, which is by design so a
switch never requires focusing the console. The trade-off: typing digits
elsewhere during a meeting also switches. If that bites in practice, swap
in a modifier chord here.
"""

from __future__ import annotations

from typing import Callable

_NUMPAD_ONE_VK = 97


class HotkeyListener:
    def __init__(self, speaker_count: int, on_switch: Callable[[int], None]) -> None:
        self._count = speaker_count
        self._on_switch = on_switch
        self._listener = None

    def set_count(self, speaker_count: int) -> None:
        """Widen the key range when a guest is added mid-session."""
        self._count = speaker_count

    def start(self) -> None:
        from pynput import keyboard

        def handle_press(key) -> None:
            index = self._speaker_index(key)
            if index is not None:
                self._on_switch(index)

        self._listener = keyboard.Listener(on_press=handle_press)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _speaker_index(self, key) -> int | None:
        char = getattr(key, "char", None)
        if char and char.isdigit() and 1 <= int(char) <= self._count:
            return int(char) - 1
        virtual_key = getattr(key, "vk", None)
        if (
            virtual_key is not None
            and _NUMPAD_ONE_VK <= virtual_key < _NUMPAD_ONE_VK + self._count
        ):
            return virtual_key - _NUMPAD_ONE_VK
        return None
