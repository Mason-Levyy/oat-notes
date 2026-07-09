"""Session-wide monotonic clock.

Every timestamp in the pipeline is seconds from one shared epoch, stamped at
capture time. When the loopback stream lands (Phase 3), both streams share a
single SessionClock so their timelines merge without drift.
"""

import time


class SessionClock:
    """Uses perf_counter: monotonic, and sub-microsecond on Windows, where
    time.monotonic() only ticks every ~15.6 ms (GetTickCount64)."""

    def __init__(self) -> None:
        self._epoch = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self._epoch
