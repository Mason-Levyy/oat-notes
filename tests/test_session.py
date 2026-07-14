from types import SimpleNamespace

from oat_notes.attribution import Speaker
from oat_notes.session import Session
from oat_notes.types import Channel, TranscriptSegment


def bare_session():
    session = Session.__new__(Session)
    session.hotkey_bank = 0
    session.roster = []
    session._events = SimpleNamespace(on_hotkey_bank=lambda bank: None)
    return session


def test_hotkey_banks_can_page_forward_without_a_roster():
    session = bare_session()
    seen = []
    session._events.on_hotkey_bank = seen.append

    for _ in range(25):
        session.page_hotkeys(1)
    assert session.hotkey_bank == 25
    assert seen[-1] == 25

    for _ in range(30):
        session.page_hotkeys(-1)
    assert session.hotkey_bank == 0


def test_empty_hotkey_slot_creates_guest_at_exact_slot():
    session = bare_session()
    session.hotkey_bank = 3
    created = []
    switched = []

    def create_guest(hotkey_slot):
        created.append(hotkey_slot)
        session.roster.append(Speaker("Guest 1", hotkey_slot=hotkey_slot))
        return 0

    session._create_guest = create_guest
    session.switch_speaker = switched.append
    session.switch_hotkey(4)

    assert created == [31]
    assert switched == [0]


def test_selecting_person_applies_to_both_audio_sources():
    session = bare_session()
    session.roster = [Speaker("Alex"), Speaker("Taylor")]
    session.channels = (Channel.MIC, Channel.LOOPBACK)
    session.active = {Channel.MIC: 0, Channel.LOOPBACK: 0}
    session.clock = SimpleNamespace(now=lambda: 4.0)
    recorded = []
    session._attributor = SimpleNamespace(
        log_for=lambda channel: SimpleNamespace(
            record=lambda timestamp, index: recorded.append((channel, timestamp, index))
        )
    )
    split = []
    session._pipeline = SimpleNamespace(split_channel=split.append)
    session._speaker_resolver = None
    selected = []
    session._events = SimpleNamespace(on_speaker=selected.append)

    session.switch_speaker(1)

    assert recorded == [
        (Channel.MIC, 4.0, 1),
        (Channel.LOOPBACK, 4.0, 1),
    ]
    assert split == [Channel.MIC, Channel.LOOPBACK]
    assert session.active == {Channel.MIC: 1, Channel.LOOPBACK: 1}
    assert selected == [1]


def test_delayed_old_attribution_cannot_override_manual_selection():
    session = bare_session()
    session.active = {Channel.MIC: 1, Channel.LOOPBACK: None}
    session._manual_override_pending = {
        Channel.MIC: True,
        Channel.LOOPBACK: False,
    }
    attributed = []
    emitted = []
    session._events = SimpleNamespace(
        on_attribution=lambda *args: attributed.append(args),
        on_segment=lambda *args: emitted.append(args),
    )
    session.log = SimpleNamespace(add=lambda segment: None)
    session._label_channels = False

    old_turn = TranscriptSegment(
        "old words",
        Channel.MIC,
        0.0,
        1.0,
        attribution="unknown",
    )
    session._handle_segment(old_turn, 0.1)
    assert session.active[Channel.MIC] == 1
    assert attributed == []

    selected_turn = TranscriptSegment(
        "new words",
        Channel.MIC,
        1.0,
        2.0,
        speaker="Bob",
        speaker_index=1,
        attribution="manual",
    )
    session._handle_segment(selected_turn, 0.1)
    assert attributed[-1][:3] == (1, Channel.MIC, "manual")
    assert session._manual_override_pending[Channel.MIC] is False
