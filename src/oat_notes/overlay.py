"""The dictation HUD: a small always-on-top pill above the taskbar.

``WS_EX_NOACTIVATE`` is the load-bearing detail. Without it, showing this
window pulls focus away from whatever the user is dictating into and the text
lands in the wrong place.

Tk widgets belong to the thread that created them, so the window is built on
the main thread and every other thread posts state through a queue that Tk's
own ``after`` loop drains.
"""

from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
from dataclasses import dataclass

from .dictation.phases import (
    CANCELLED,
    ERROR,
    FORMATTING,
    IDLE,
    INSERTED,
    LATCHED,
    LISTENING,
    LOADING,
    TRANSCRIBING,
)
from .log import error_kind

log = logging.getLogger(__name__)

INK = "#1A1A1A"
PAPER = "#EDE8DD"
TRANSPARENT = "#FF00FF"

_VISIBLE = {LISTENING, LATCHED, TRANSCRIBING, FORMATTING, INSERTED, ERROR, LOADING}
_SELF_DISMISSING = {INSERTED, CANCELLED, ERROR}
_DISMISS_AFTER_MS = 1100

_PIXEL = 3
_WIDTH = 174
_HEIGHT = 30
_BORDER = _PIXEL
_CORNER_STEPS = 2
_BOTTOM_MARGIN = 72
_METER_BARS = 9
_DOT = _PIXEL * 3
_LABEL_X = _PIXEL * 3 + _DOT + 6
_METER_SPAN = _PIXEL * 2
_METER_LEFT = _WIDTH - _PIXEL * 3 - _METER_BARS * _METER_SPAN
LABEL_BUDGET = int((_METER_LEFT - _LABEL_X) / 6.6)

_GWL_EXSTYLE = -20
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_TOPMOST = 0x00000008
_SW_SHOWNOACTIVATE = 4
_DPI_PER_MONITOR_AWARE_V2 = -4
_DPI_PER_MONITOR_AWARE_WINDOWS_10 = 2
_MONITOR_DEFAULTTONEAREST = 2


@dataclass
class OverlayState:
    phase: str = IDLE
    level: float = 0.0
    email: bool = False
    label: str = ""


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", _Rect),
        ("rcWork", _Rect),
        ("dwFlags", ctypes.c_ulong),
    ]


def set_dpi_awareness() -> None:
    """Must run before the first Tk window exists, or the HUD is blurry and
    mispositioned on any display that isn't at 100% scaling."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(_DPI_PER_MONITOR_AWARE_V2)
        )
        return
    except Exception as error:
        log.debug("per-monitor DPI awareness v2 unavailable: %s", error_kind(error))
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(
            _DPI_PER_MONITOR_AWARE_WINDOWS_10
        )
    except Exception as error:
        log.warning("DPI awareness not set; the overlay may be blurry: %s", error_kind(error))


def cursor_work_area() -> tuple[int, int, int, int]:
    """Work area of the monitor under the pointer — already excludes the
    taskbar. Falls back to a sane primary-display guess."""
    if sys.platform != "win32":
        return (0, 0, 1920, 1080)
    try:
        user32 = ctypes.windll.user32
        point = _Point()
        user32.GetCursorPos(ctypes.byref(point))
        monitor = user32.MonitorFromPoint(point, _MONITOR_DEFAULTTONEAREST)
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        user32.GetMonitorInfoW(monitor, ctypes.byref(info))
        work = info.rcWork
        return (work.left, work.top, work.right, work.bottom)
    except Exception:
        return (0, 0, 1920, 1080)


def make_click_through_and_unfocusable(hwnd: int) -> None:
    """The whole reason this module can exist alongside text injection."""
    if sys.platform != "win32" or not hwnd:
        return
    try:
        user32 = ctypes.windll.user32
        get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        set_style.restype = ctypes.c_longlong
        get_style.restype = ctypes.c_longlong
        style = get_style(ctypes.c_void_p(hwnd), _GWL_EXSTYLE)
        set_style(
            ctypes.c_void_p(hwnd),
            _GWL_EXSTYLE,
            ctypes.c_longlong(
                int(style) | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW | _WS_EX_TOPMOST
            ),
        )
        user32.ShowWindow(ctypes.c_void_p(hwnd), _SW_SHOWNOACTIVATE)
    except Exception as error:
        log.warning("overlay focus guard failed: %s", error_kind(error))


def phase_label(state: OverlayState) -> str:
    if state.label:
        return state.label
    label = {
        LISTENING: "REC",
        LATCHED: "LOCKED",
        TRANSCRIBING: "THINKING",
        FORMATTING: "EMAIL" if state.email else "CLEANING",
        INSERTED: "INSERTED",
        CANCELLED: "CANCELLED",
        ERROR: "ERROR",
        LOADING: "LOADING",
    }.get(state.phase, "")
    if state.email and state.phase in (LISTENING, LATCHED):
        return f"{label} EMAIL"
    return label


def pixel_rounded_rows(
    width: int, height: int, step: int = _PIXEL, corner_steps: int = _CORNER_STEPS
) -> list[tuple[int, int, int, int]]:
    """Horizontal bands making up a rectangle whose corners step inward, so
    they read as rounded while staying on the pixel grid."""
    rows = []
    for top in range(0, height, step):
        bottom = min(height, top + step)
        from_edge = min(top // step, (height - bottom) // step)
        inset = max(0, corner_steps - from_edge) * step
        rows.append((inset, top, width - inset, bottom))
    return rows


def meter_heights(level: float, bars: int = _METER_BARS) -> list[float]:
    """Bar fractions for the level meter, scaled so ordinary speech fills
    most of the meter rather than hugging the floor."""
    scaled = min(1.0, (max(0.0, level) ** 0.45) * 2.6)
    heights = []
    for index in range(bars):
        weight = 0.45 + 0.55 * (1.0 - abs((index / max(1, bars - 1)) - 0.5) * 2)
        heights.append(max(0.08, min(1.0, scaled * weight)))
    return heights


class Overlay:
    """Created and run on the main thread; fed from anywhere via ``post``."""

    def __init__(self, on_quit=None, on_open_ui=None) -> None:
        self._queue: queue.Queue[OverlayState] = queue.Queue(maxsize=64)
        self._state = OverlayState()
        self._on_quit = on_quit
        self._on_open_ui = on_open_ui
        self._root = None
        self._canvas = None
        self._visible = False
        self._hide_job = None

    def post(self, state: OverlayState) -> None:
        """Thread-safe. Drops rather than blocks, like every other producer
        in this application."""
        try:
            self._queue.put_nowait(state)
        except queue.Full:
            pass

    def post_phase(self, phase: str, details: dict | None = None) -> None:
        details = details or {}
        self.post(
            OverlayState(
                phase=phase,
                level=self._state.level,
                email=bool(details.get("email") or details.get("mode") == "email"),
            )
        )

    def post_level(self, level: float) -> None:
        if self._state.phase in (LISTENING, LATCHED):
            self.post(
                OverlayState(
                    phase=self._state.phase, level=level, email=self._state.email
                )
            )

    def run(self, should_stop: threading.Event, poll_ms: int = 100) -> None:
        """Owns the Tk main loop until ``should_stop`` is set."""
        import tkinter as tk

        set_dpi_awareness()
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.configure(bg=TRANSPARENT)
        try:
            self._root.attributes("-transparentcolor", TRANSPARENT)
        except tk.TclError:
            self._root.configure(bg=PAPER)
        self._canvas = tk.Canvas(
            self._root,
            width=_WIDTH,
            height=_HEIGHT,
            bg=self._root.cget("bg"),
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack()
        self._build_menu(tk)
        self._position()
        self._realize_then_apply_focus_guard()
        self._root.withdraw()

        def tick() -> None:
            if should_stop.is_set():
                self._root.quit()
                return
            self._drain()
            self._root.after(poll_ms, tick)

        self._root.after(poll_ms, tick)
        try:
            self._root.mainloop()
        finally:
            try:
                self._root.destroy()
            except Exception as error:
                log.debug("overlay window did not close cleanly: %s", error_kind(error))

    def _hwnd(self) -> int:
        try:
            return int(self._root.wm_frame(), 16)
        except Exception:
            return 0

    def _realize_then_apply_focus_guard(self) -> None:
        """A window has no HWND until Tk has realized it, and both creating
        and deiconifying one clear the guard, so every path that puts this on
        screen re-applies it."""
        self._root.update_idletasks()
        make_click_through_and_unfocusable(self._hwnd())

    def _build_menu(self, tk) -> None:
        """Right-click menu stands in for a tray icon — no extra dependency."""
        menu = tk.Menu(self._root, tearoff=0)
        if self._on_open_ui is not None:
            menu.add_command(label="Open Oat Notes", command=self._on_open_ui)
        if self._on_quit is not None:
            menu.add_separator()
            menu.add_command(label="Quit", command=self._on_quit)
        self._menu = menu
        self._canvas.bind(
            "<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root)
        )

    def _position(self) -> None:
        left, _top, right, bottom = cursor_work_area()
        x = left + (right - left - _WIDTH) // 2
        y = bottom - _HEIGHT - _BOTTOM_MARGIN
        self._root.geometry(f"{_WIDTH}x{_HEIGHT}+{x}+{y}")

    def _drain(self) -> None:
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return
        self._state = latest
        self._render()

    def _render(self) -> None:
        if self._state.phase in _VISIBLE:
            self._show()
            self._draw()
            if self._state.phase in _SELF_DISMISSING:
                self._schedule_hide()
        else:
            self._hide()

    def _show(self) -> None:
        if self._hide_job is not None:
            self._root.after_cancel(self._hide_job)
            self._hide_job = None
        if self._visible:
            return
        self._position()
        self._root.deiconify()
        self._realize_then_apply_focus_guard()
        self._visible = True

    def _hide(self) -> None:
        if not self._visible:
            return
        self._root.withdraw()
        self._visible = False

    def _schedule_hide(self) -> None:
        if self._hide_job is not None:
            self._root.after_cancel(self._hide_job)
        self._hide_job = self._root.after(_DISMISS_AFTER_MS, self._hide)

    def _draw(self) -> None:
        canvas = self._canvas
        canvas.delete("all")
        self._draw_body(canvas)

        state = self._state
        recording = state.phase in (LISTENING, LATCHED)
        middle = _HEIGHT / 2

        canvas.create_rectangle(
            _PIXEL * 3,
            middle - _DOT / 2,
            _PIXEL * 3 + _DOT,
            middle + _DOT / 2,
            fill=INK if recording else PAPER,
            outline=INK,
            width=1,
        )
        canvas.create_text(
            _LABEL_X,
            middle,
            text=phase_label(state),
            anchor="w",
            fill=INK,
            font=("Consolas", 8, "bold"),
        )
        if recording:
            self._draw_meter(canvas)

    def _draw_body(self, canvas) -> None:
        for left, top, right, bottom in pixel_rounded_rows(_WIDTH, _HEIGHT):
            canvas.create_rectangle(left, top, right, bottom, fill=INK, outline="")
        inner = pixel_rounded_rows(_WIDTH - 2 * _BORDER, _HEIGHT - 2 * _BORDER)
        for left, top, right, bottom in inner:
            canvas.create_rectangle(
                left + _BORDER,
                top + _BORDER,
                right + _BORDER,
                bottom + _BORDER,
                fill=PAPER,
                outline="",
            )

    def _draw_meter(self, canvas) -> None:
        middle = _HEIGHT / 2
        tallest = _HEIGHT - _PIXEL * 6
        for index, height in enumerate(meter_heights(self._state.level)):
            x = _METER_LEFT + index * _METER_SPAN
            half = max(1.0, height * tallest / 2)
            canvas.create_rectangle(
                x,
                middle - half,
                x + _METER_SPAN - 2,
                middle + half,
                fill=INK,
                outline="",
            )
