import time

from oat_notes.clock import SessionClock


def test_starts_near_zero_and_increases():
    clock = SessionClock()
    first = clock.now()
    assert 0.0 <= first < 0.1
    time.sleep(0.01)
    assert clock.now() > first


def test_clocks_have_independent_epochs():
    earlier = SessionClock()
    time.sleep(0.02)
    later = SessionClock()
    assert later.now() < earlier.now()
