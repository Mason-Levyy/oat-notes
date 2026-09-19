"""Fans JSON events out to every connected SSE client."""

from __future__ import annotations

import logging
import queue
import threading

log = logging.getLogger(__name__)

SUBSCRIBER_BACKLOG = 512


class EventHub:
    def __init__(self) -> None:
        self._subscribers: list[queue.Queue] = []
        self._lagging: set[int] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_BACKLOG)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
            self._lagging.discard(id(subscriber))

    def publish(self, event: dict) -> int:
        """Returns how many subscribers missed the event because their
        queue was full — a browser tab that has stopped draining."""
        with self._lock:
            subscribers = list(self._subscribers)
        missed = 0
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                missed += 1
                if id(subscriber) not in self._lagging:
                    self._lagging.add(id(subscriber))
                    log.warning("a browser tab stopped reading events; dropping them")
        return missed
