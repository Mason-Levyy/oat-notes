"""Session-wide monotonic clock.

Every timestamp in the pipeline is seconds from one shared epoch, stamped at
capture time. When the loopback stream lands (Phase 3), both streams share a
single SessionClock so their timelines merge without drift.
"""

import time


class SessionClock:
    def __init__(self) -> None:
        self._epoch = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._epoch
