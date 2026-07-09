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
    remote = segment("hi there", 6.0, channel=Channel.LOOPBACK)
    named = segment("agreed", 7.0, speaker="Sarah")

    assert line_for(mic, label_channels=False) == "[00:00:05] hello"
    assert line_for(mic, label_channels=True) == "[00:00:05] Me: hello"
    assert line_for(remote, label_channels=True) == "[00:00:06] Remote: hi there"
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
        "[00:00:03] Me: first\n"
        "[00:00:10] Remote: second\n"
        "[00:00:20] Me: third\n"
    )


def test_meeting_log_empty(tmp_path):
    log = MeetingLog(label_channels=False)
    assert log.is_empty
    log.add(segment("something", 1.0))
    assert not log.is_empty
