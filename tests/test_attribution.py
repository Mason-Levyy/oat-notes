import numpy as np

from oat_notes.attribution import Attributor, Speaker, SwitchLog, parse_speakers
from oat_notes.types import AudioChunk, Channel


def chunk(start, end, channel=Channel.MIC):
    return AudioChunk(
        samples=np.zeros(16, dtype=np.float32), channel=channel, start=start, end=end
    )


def make_attributor(roster, mic_log=None, loopback_log=None):
    return Attributor(
        roster,
        mic_log=mic_log or SwitchLog(),
        loopback_log=loopback_log or SwitchLog(),
    )


def test_parse_speakers_returns_location_neutral_roster():
    roster = parse_speakers("Mason, Sarah, Priya, Dev")
    assert roster == (
        Speaker("Mason"),
        Speaker("Sarah"),
        Speaker("Priya"),
        Speaker("Dev"),
    )


def test_parse_speakers_empty():
    assert parse_speakers("") == ()
    assert parse_speakers(" , ,") == ()


def test_initial_speaker_active_before_any_switch():
    log = SwitchLog(initial=2)
    assert log.active_at(0.0) == 2
    assert log.attribute(0.0, 10.0) == 2


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


def test_rename_updates_future_attribution():
    attributor = make_attributor((Speaker("Guest 1"), Speaker("Dev")))
    attributor.rename(0, "Sarah")
    assert attributor.for_chunk(chunk(0, 5)) == "Sarah"
    attributor.rename(9, "ignored")  # out of range is a no-op, not an error


def test_single_speaker_is_attributed_on_either_source():
    roster = parse_speakers("Priya")
    attributor = make_attributor(roster)
    assert attributor.for_chunk(chunk(0, 5, Channel.LOOPBACK)) == "Priya"
    assert attributor.for_chunk(chunk(0, 5, Channel.MIC)) == "Priya"


def test_each_source_keeps_its_own_switch_timeline():
    roster = parse_speakers("Mason, Priya, Dev")
    loopback_log = SwitchLog(initial=1)
    attributor = make_attributor(roster, loopback_log=loopback_log)
    assert attributor.for_chunk(chunk(0.0, 4.0, Channel.LOOPBACK)) == "Priya"
    loopback_log.record(5.0, 2)
    assert attributor.for_chunk(chunk(6.0, 9.0, Channel.LOOPBACK)) == "Dev"
    # A switch observed on one capture source does not rewrite the other's timeline.
    assert attributor.for_chunk(chunk(6.0, 9.0)) == "Mason"


def test_speakers_switch_on_mic_log():
    roster = parse_speakers("Mason, Sarah, Priya")
    mic_log = SwitchLog(initial=0)
    attributor = make_attributor(roster, mic_log=mic_log)
    assert attributor.for_chunk(chunk(0.0, 4.0)) == "Mason"
    mic_log.record(5.0, 1)
    assert attributor.for_chunk(chunk(6.0, 9.0)) == "Sarah"


def test_empty_roster_falls_back_to_none():
    attributor = make_attributor(())
    assert attributor.for_chunk(chunk(0, 5)) is None
    assert attributor.for_chunk(chunk(0, 5, Channel.LOOPBACK)) is None


def test_source_log_helpers():
    roster = parse_speakers("Mason, Priya")
    attributor = make_attributor(roster)
    assert attributor.first_member(Channel.MIC) == 0
    assert attributor.first_member(Channel.LOOPBACK) == 0


def test_add_guest_mid_session():
    roster = parse_speakers("Mason, Priya")
    mic_log = SwitchLog()
    loopback_log = SwitchLog(initial=1)
    attributor = make_attributor(roster, mic_log=mic_log, loopback_log=loopback_log)

    guest_one = attributor.add(Speaker("Guest 1"))
    guest_two = attributor.add(Speaker("Guest 2"))
    assert (guest_one, guest_two) == (2, 3)

    mic_log.record(5.0, guest_one)
    loopback_log.record(5.0, guest_two)
    assert attributor.for_chunk(chunk(6.0, 9.0)) == "Guest 1"
    assert attributor.for_chunk(chunk(6.0, 9.0, Channel.LOOPBACK)) == "Guest 2"


def test_out_of_group_log_index_falls_back_to_first_member():
    roster = parse_speakers("Mason, Sarah")
    mic_log = SwitchLog(initial=20)
    attributor = make_attributor(roster, mic_log=mic_log)
    assert attributor.for_chunk(chunk(0.0, 4.0)) == "Mason"
