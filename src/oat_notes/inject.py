"""Insert dictated text into whatever window currently has focus.

Either paste through the clipboard or synthesize the characters directly,
both via Win32 ``SendInput``.

Ordering matters: at commit time the dictation chord's own modifiers may
still be held, and Ctrl+V with the Windows key down opens Clipboard History
instead of pasting. So held modifiers are cleared before anything is sent.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import wintypes

PASTE = "paste"
TYPE = "type"

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_RETURN = 0x0D
VK_V = 0x56
VK_NONAME = 0xFC

_CHORD_MODIFIER_KEYS = (VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN)

_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004

_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002

_MODIFIER_WAIT_SECONDS = 0.25
_MODIFIER_POLL_SECONDS = 0.01
_CLIPBOARD_RESTORE_SECONDS = 0.35
_CLIPBOARD_OPEN_ATTEMPTS = 10
_CLIPBOARD_RETRY_SECONDS = 0.02

_IS_WINDOWS = sys.platform == "win32"
_user32 = ctypes.WinDLL("user32", use_last_error=True) if _IS_WINDOWS else None
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if _IS_WINDOWS else None

_clipboard_lock = threading.Lock()


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput), ("hi", _HardwareInput)]


class INPUT(ctypes.Structure):
    """The union must carry all three members: a keyboard-only definition is
    16 bytes short on x64 and SendInput rejects it."""

    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]


class InjectionError(RuntimeError):
    pass


def _declare_prototypes() -> None:
    """ctypes defaults every return type to 32-bit int, silently truncating
    the 64-bit handles and pointers these functions return. The corrupt
    pointer then faults the interpreter on first dereference."""
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetForegroundWindow.argtypes = []
    _user32.SetForegroundWindow.restype = wintypes.BOOL
    _user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    _user32.GetAsyncKeyState.restype = ctypes.c_short
    _user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    _user32.SendInput.restype = wintypes.UINT
    _user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]

    _user32.OpenClipboard.restype = wintypes.BOOL
    _user32.OpenClipboard.argtypes = [wintypes.HWND]
    _user32.CloseClipboard.restype = wintypes.BOOL
    _user32.CloseClipboard.argtypes = []
    _user32.EmptyClipboard.restype = wintypes.BOOL
    _user32.EmptyClipboard.argtypes = []
    _user32.GetClipboardData.restype = wintypes.HANDLE
    _user32.GetClipboardData.argtypes = [wintypes.UINT]
    _user32.SetClipboardData.restype = wintypes.HANDLE
    _user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    _user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    _user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    _user32.RegisterClipboardFormatW.restype = wintypes.UINT
    _user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]

    _kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    _kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    _kernel32.GlobalLock.restype = wintypes.LPVOID
    _kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    _kernel32.GlobalUnlock.restype = wintypes.BOOL
    _kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    _kernel32.GlobalFree.restype = wintypes.HGLOBAL
    _kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]


if _IS_WINDOWS:
    _declare_prototypes()


def _require_windows() -> None:
    if not _IS_WINDOWS:
        raise InjectionError("text injection is only implemented for Windows")


def foreground_window() -> int:
    """The window to insert into, captured when the chord arms. Zero when
    nothing is focused: an HWND restype yields None rather than 0."""
    _require_windows()
    return int(_user32.GetForegroundWindow() or 0)


def restore_foreground(hwnd: int) -> bool:
    _require_windows()
    if not hwnd or foreground_window() == hwnd:
        return True
    return bool(_user32.SetForegroundWindow(wintypes.HWND(hwnd)))


def _key_input(vk: int, up: bool = False) -> INPUT:
    event = INPUT(type=_INPUT_KEYBOARD)
    event.ki = _KeyboardInput(
        wVk=vk, wScan=0, dwFlags=_KEYEVENTF_KEYUP if up else 0, time=0, dwExtraInfo=None
    )
    return event


def _unicode_input(code_unit: int, up: bool = False) -> INPUT:
    flags = _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if up else 0)
    event = INPUT(type=_INPUT_KEYBOARD)
    event.ki = _KeyboardInput(
        wVk=0, wScan=code_unit, dwFlags=flags, time=0, dwExtraInfo=None
    )
    return event


def _send(events: list[INPUT]) -> None:
    if not events:
        return
    array = (INPUT * len(events))(*events)
    sent = _user32.SendInput(len(events), array, ctypes.sizeof(INPUT))
    if sent != len(events):
        raise InjectionError(
            f"SendInput delivered {sent}/{len(events)} events"
            f" (error {ctypes.get_last_error()})"
        )


def held_modifiers() -> list[int]:
    _require_windows()
    return [
        vk for vk in _CHORD_MODIFIER_KEYS if _user32.GetAsyncKeyState(vk) & 0x8000
    ]


def release_modifiers(timeout: float = _MODIFIER_WAIT_SECONDS) -> None:
    """Wait for the user to lift the chord, then force up what is left. Only
    genuinely-held keys are released: a spurious Win keyup pops the Start menu."""
    _require_windows()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if not held_modifiers():
            return
        time.sleep(_MODIFIER_POLL_SECONDS)
    stuck = held_modifiers()
    if stuck:
        _send([_key_input(vk, up=True) for vk in stuck])


def defuse_start_menu() -> None:
    """Stop a held Windows key from opening Start when it is released.

    Windows arms "open Start on keyup" when the Windows key goes down on its
    own, and a modifier pressed afterwards does not disarm it — so Win-then-
    Ctrl opens Start while Ctrl-then-Win does not. Pressing any key during
    the hold does disarm it, so one unassigned virtual key is enough.

    Suppressing the keyup instead would leave Windows believing the key is
    still down, turning every later keystroke into a Win chord.
    """
    _require_windows()
    _send([_key_input(VK_NONAME), _key_input(VK_NONAME, up=True)])


def _text_events(text: str) -> list[INPUT]:
    """Text is walked as UTF-16 code units so astral characters arrive as the
    surrogate pairs KEYEVENTF_UNICODE expects. Newlines go as VK_RETURN —
    sending "\\n" as a unicode code unit is unreliable across applications."""
    events: list[INPUT] = []
    for index, line in enumerate(text.split("\n")):
        if index:
            events.append(_key_input(VK_RETURN))
            events.append(_key_input(VK_RETURN, up=True))
        encoded = line.encode("utf-16-le")
        for position in range(0, len(encoded), 2):
            unit = int.from_bytes(encoded[position:position + 2], "little")
            events.append(_unicode_input(unit))
            events.append(_unicode_input(unit, up=True))
    return events


def type_text(text: str) -> None:
    """Synthesize the text as one batched SendInput call."""
    _require_windows()
    if not text:
        return
    release_modifiers()
    _send(_text_events(text))


def _open_clipboard() -> None:
    for attempt in range(_CLIPBOARD_OPEN_ATTEMPTS):
        if _user32.OpenClipboard(None):
            return
        time.sleep(_CLIPBOARD_RETRY_SECONDS * (attempt + 1))
    raise InjectionError("could not open the clipboard")


def _global_text(text: str) -> int:
    encoded = text.encode("utf-16-le") + b"\x00\x00"
    handle = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(encoded))
    if not handle:
        raise InjectionError("could not allocate clipboard memory")
    pointer = _kernel32.GlobalLock(handle)
    if not pointer:
        _kernel32.GlobalFree(handle)
        raise InjectionError("could not lock clipboard memory")
    ctypes.memmove(pointer, encoded, len(encoded))
    _kernel32.GlobalUnlock(handle)
    return handle


def _hand_ownership_to_clipboard(clipboard_format: int, handle: int) -> None:
    _user32.SetClipboardData(clipboard_format, handle)


def read_clipboard_text() -> str | None:
    """Current CF_UNICODETEXT contents, or None when the clipboard holds
    something this module cannot preserve."""
    _require_windows()
    _open_clipboard()
    try:
        if not _user32.IsClipboardFormatAvailable(_CF_UNICODETEXT):
            return None
        handle = _user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return None
        pointer = _kernel32.GlobalLock(handle)
        if not pointer:
            return None
        try:
            return ctypes.c_wchar_p(pointer).value
        finally:
            _kernel32.GlobalUnlock(handle)
    finally:
        _user32.CloseClipboard()


def _mark_private() -> None:
    """Keep dictated text out of Win+V history and cloud clipboard sync."""
    fmt = _user32.RegisterClipboardFormatW(
        "ExcludeClipboardContentFromMonitorProcessing"
    )
    if fmt:
        _hand_ownership_to_clipboard(
            fmt, _kernel32.GlobalAlloc(_GMEM_MOVEABLE, 1)
        )
    fmt = _user32.RegisterClipboardFormatW("CanIncludeInClipboardHistory")
    if fmt:
        handle = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, 4)
        pointer = _kernel32.GlobalLock(handle)
        if pointer:
            ctypes.memmove(pointer, (0).to_bytes(4, "little"), 4)
            _kernel32.GlobalUnlock(handle)
            _hand_ownership_to_clipboard(fmt, handle)


def write_clipboard_text(text: str, private: bool = True) -> None:
    _require_windows()
    _open_clipboard()
    try:
        _user32.EmptyClipboard()
        _hand_ownership_to_clipboard(_CF_UNICODETEXT, _global_text(text))
        if private:
            _mark_private()
    finally:
        _user32.CloseClipboard()


def paste_text(text: str, restore_clipboard: bool = True) -> None:
    """Put the text on the clipboard and send Ctrl+V.

    The previous clipboard text is put back shortly afterwards. The delay is
    unavoidable — the target app reads the clipboard asynchronously — so the
    restore is best-effort, and only text contents can be preserved.
    """
    _require_windows()
    if not text:
        return
    with _clipboard_lock:
        previous = read_clipboard_text() if restore_clipboard else None
        write_clipboard_text(text)
        release_modifiers()
        _send(
            [
                _key_input(VK_CONTROL),
                _key_input(VK_V),
                _key_input(VK_V, up=True),
                _key_input(VK_CONTROL, up=True),
            ]
        )
    if previous is not None:
        timer = threading.Timer(
            _CLIPBOARD_RESTORE_SECONDS, _restore_clipboard, args=(previous,)
        )
        timer.name = "clipboard-restore"
        timer.daemon = True
        timer.start()


def _restore_clipboard(previous: str) -> None:
    try:
        with _clipboard_lock:
            write_clipboard_text(previous, private=False)
    except Exception as error:
        print(
            f"could not restore the clipboard: {type(error).__name__}", file=sys.stderr
        )


def inject(
    text: str,
    method: str = PASTE,
    hwnd: int | None = None,
    restore_clipboard: bool = True,
) -> None:
    """Insert ``text`` into ``hwnd``, or whatever has focus now.

    A non-elevated process cannot send input to an elevated window, so
    dictating into an admin console silently does nothing. That is Windows
    UIPI, not something this can work around.
    """
    _require_windows()
    if not text:
        return
    if hwnd:
        restore_foreground(hwnd)
    if method == TYPE:
        type_text(text)
    else:
        paste_text(text, restore_clipboard=restore_clipboard)
