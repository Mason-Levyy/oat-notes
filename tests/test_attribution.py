import numpy as np

from oat_notes.attribution import Attributor, SwitchLog
from oat_notes.types import AudioChunk, Channel


def chunk(start, end, channel=Channel.MIC):
    return AudioChunk(
        samples=np.zeros(16, dtype=np.float32), channel=channel, start=start, end=end
    )


def test_speaker_zero_active_before_any_switch():
    log = SwitchLog()
    assert log.active_at(0.0) == 0
    assert log.attribute(0.0, 10.0) == 0


def test_switch_before_chunk():
    log = SwitchLog()
    log.record(5.0, 1)
    assert log.attribute(6.0, 9.0) == 1


def test_majority_overlap_mid_chunk_switch():
    log = SwitchLog()
    log.record(4.0, 1)
    # Chunk spans 0..10: speaker 0 for 4s, speaker 1 for 6s.
    assert log.attribute(0.0, 10.0) == 1
    # Chunk spans 0..6: speaker 0 for 4s, speaker 1 for 2s.
    assert log.attribute(0.0, 6.0) == 0


def test_multiple_switches_within_chunk():
    log = SwitchLog()
    log.record(2.0, 1)
    log.record(3.0, 0)
    log.record(9.0, 2)
    # Spans in 0..10: speaker 0 gets 2s + 6s, speaker 1 gets 1s, speaker 2 gets 1s.
    assert log.attribute(0.0, 10.0) == 0


def test_switch_back_and_forth_respects_latest_before_chunk():
    log = SwitchLog()
    log.record(1.0, 1)
    log.record(2.0, 0)
    log.record(3.0, 1)
    assert log.active_at(2.5) == 0
    assert log.active_at(3.5) == 1


def test_attributor_loopback_gets_remote_name():
    attributor = Attributor(["Mason", "Sarah"], SwitchLog(), remote_name="Priya")
    assert attributor.for_chunk(chunk(0, 5, Channel.LOOPBACK)) == "Priya"


def test_attributor_loopback_without_name_returns_none():
    attributor = Attributor(["Mason", "Sarah"], SwitchLog())
    assert attributor.for_chunk(chunk(0, 5, Channel.LOOPBACK)) is None


def test_attributor_single_mic_speaker_needs_no_log():
    attributor = Attributor(["Mason"], SwitchLog())
    assert attributor.for_chunk(chunk(0, 5)) == "Mason"


def test_attributor_multi_speaker_uses_switch_log():
    log = SwitchLog()
    attributor = Attributor(["Mason", "Sarah"], log)
    assert attributor.for_chunk(chunk(0.0, 4.0)) == "Mason"
    log.record(5.0, 1)
    assert attributor.for_chunk(chunk(6.0, 9.0)) == "Sarah"


def test_attributor_no_mic_speakers_returns_none():
    attributor = Attributor([], SwitchLog(), remote_name="Priya")
    assert attributor.for_chunk(chunk(0, 5)) is None
