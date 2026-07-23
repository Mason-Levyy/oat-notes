from datetime import datetime

from oat_notes.output import MeetingLog, format_timestamp, line_for
from oat_notes.types import Channel, TranscriptSegment


def segment(text, start, channel=Channel.MIC, speaker=None):
    return TranscriptSegment(
        text=text, channel=channel, start=start, end=start + 2.0, speaker=speaker
    )


def test_format_timestamp():
    assert format_timestamp(0.4) == "00:00:00"
    assert format_timestamp(192.7) == "00:03:12"
    assert format_timestamp(3723.0) == "01:02:03"


def test_line_labels():
    mic = segment("hello", 5.0)
    system = segment("hi there", 6.0, channel=Channel.LOOPBACK)
    named = segment("agreed", 7.0, speaker="Sarah")

    assert line_for(mic, label_channels=False) == "[00:00:05] hello"
    assert line_for(mic, label_channels=True) == "[00:00:05] Microphone: hello"
    assert line_for(system, label_channels=True) == "[00:00:06] System audio: hi there"
    # A named speaker (Phase 2 attribution) wins over the channel label.
    assert line_for(named, label_channels=True) == "[00:00:07] Sarah: agreed"


def test_slugify():
    from oat_notes.output import slugify

    assert slugify("Stand-up Meeting!") == "stand-up-meeting"
    assert slugify("  Q3 budget: review  ") == "q3-budget-review"
    assert slugify("///") == "meeting"
    assert slugify("") == "meeting"


def test_meeting_log_named_file(tmp_path):
    from datetime import datetime

    log = MeetingLog(label_channels=False)
    log.add(segment("hello", 1.0))
    path = log.save(tmp_path, datetime(2026, 7, 9, 14, 30), "Stand-up Meeting")
    assert path.name == "stand-up-meeting_2026-07-09_1430.txt"


def test_meeting_log_sorts_and_writes(tmp_path):
    log = MeetingLog(label_channels=True)
    log.add(segment("second", 10.0, channel=Channel.LOOPBACK))
    log.add(segment("first", 3.0))
    log.add(segment("third", 20.0))

    path = log.save(tmp_path, datetime(2026, 7, 9, 14, 30))

    assert path.name == "meeting_2026-07-09_1430.txt"
    assert path.read_text(encoding="utf-8") == (
        "[00:00:03] Microphone: first\n"
        "[00:00:10] System audio: second\n"
        "[00:00:20] Microphone: third\n"
    )


def test_meeting_log_rename_backfills_guests(tmp_path):
    from datetime import datetime

    log = MeetingLog(label_channels=False)
    log.add(segment("hello", 1.0, speaker="Guest 1"))
    log.add(segment("hi", 2.0, speaker="Mason"))
    log.add(segment("more", 3.0, speaker="Guest 1"))
    log.rename({"Guest 1": "Tom"})

    path = log.save(tmp_path, datetime(2026, 7, 9, 14, 30))
    content = path.read_text(encoding="utf-8")
    assert "Tom: hello" in content
    assert "Tom: more" in content
    assert "Mason: hi" in content
    assert "Guest" not in content


def test_speakers_with_lines():
    log = MeetingLog(label_channels=False)
    log.add(segment("hello", 1.0, speaker="Mason"))
    log.add(segment("hi", 2.0, speaker="Guest 1"))
    log.add(segment("unlabeled", 3.0))
    assert log.speakers_with_lines() == {"Mason", "Guest 1"}


def test_apply_cleanup_rewrites_and_drops_lines(tmp_path):
    log = MeetingLog(label_channels=False)
    first = log.add(segment("um, hello there", 1.0, speaker="Mason"))
    second = log.add(segment("asdfjkl noise", 2.0))
    third = log.add(segment("unchanged", 3.0))
    assert (first, second, third) == (0, 1, 2)

    log.apply_cleanup({first: "hello there", second: None})

    path = log.save(tmp_path, datetime(2026, 7, 9, 14, 30))
    content = path.read_text(encoding="utf-8")
    assert "Mason: hello there\n" in content
    assert "asdfjkl" not in content
    assert "unchanged" in content


def test_meeting_log_empty(tmp_path):
    log = MeetingLog(label_channels=False)
    assert log.is_empty
    log.add(segment("something", 1.0))
    assert not log.is_empty


def test_meeting_log_empty_considers_notes_too():
    log = MeetingLog(label_channels=False)
    assert log.is_empty
    log.add_note(1.0, "action item")
    assert not log.is_empty


def test_meeting_log_merges_notes_with_transcript_by_timestamp(tmp_path):
    log = MeetingLog(label_channels=False)
    log.add(segment("first", 3.0))
    log.add_note(10.0, "follow up with finance")
    log.add(segment("third", 20.0))
    log.add_note(1.0, "kickoff")

    path = log.save(tmp_path, datetime(2026, 7, 9, 14, 30))

    assert path.read_text(encoding="utf-8") == (
        "[00:00:01] NOTE: kickoff\n"
        "[00:00:03] first\n"
        "[00:00:10] NOTE: follow up with finance\n"
        "[00:00:20] third\n"
    )
