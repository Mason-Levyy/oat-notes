from types import SimpleNamespace

from oat_notes.attribution import Speaker
from oat_notes.session import Session
from oat_notes.types import Channel, TranscriptSegment


def bare_session():
    session = Session.__new__(Session)
    session.hotkey_bank = 0
    session.roster = []
    session._cleanup = None
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


def test_switch_hotkey_creates_a_new_guest_when_the_slot_holder_was_removed():
    session = bare_session()
    session.hotkey_bank = 0
    session.roster = [Speaker("Guest 1", hotkey_slot=4, active=False)]
    created = []
    switched = []

    def create_guest(hotkey_slot):
        created.append(hotkey_slot)
        session.roster.append(Speaker("Guest 2", hotkey_slot=hotkey_slot))
        return len(session.roster) - 1

    session._create_guest = create_guest
    session.switch_speaker = switched.append
    session.switch_hotkey(4)

    assert created == [4]
    assert switched == [1]


def test_switch_speaker_ignores_a_removed_person():
    session = bare_session()
    session.roster = [Speaker("Alex", hotkey_slot=0, active=False)]
    session.current_speaker = None
    session.switch_speaker(0)
    assert session.current_speaker is None


def test_add_speaker_appends_with_next_free_hotkey_slot_without_switching():
    session = bare_session()
    session.roster = [Speaker("Alex", hotkey_slot=0)]
    session._attributor = SimpleNamespace(add=lambda speaker: len(session.roster))

    index = session.add_speaker("Jordan", speaker_id="sp-1")

    assert index == 1
    assert session.roster[1] == Speaker("Jordan", speaker_id="sp-1", hotkey_slot=1)


def test_remove_speaker_rejects_out_of_range_or_spoken():
    session = bare_session()
    session.roster = [Speaker("Alex", hotkey_slot=0), Speaker("Sam", hotkey_slot=1)]
    session.current_speaker = 0
    session.active = {Channel.MIC: 0, Channel.LOOPBACK: None}
    session.profile_learning = None
    session.log = SimpleNamespace(speakers_with_lines=lambda: {"Sam"})

    assert session.remove_speaker(5) == "speaker not found"
    assert session.remove_speaker(1) == "speaker has already spoken"


def test_remove_speaker_allows_removing_a_mistakenly_created_active_guest():
    # A freshly created Guest is switched to immediately, but they haven't
    # actually spoken — removing them (undoing the mis-click) must still
    # work, and must clear the now-stale active/current-speaker pointers.
    session = bare_session()
    session.roster = [Speaker("Alex", hotkey_slot=0), Speaker("Guest 1", hotkey_slot=1)]
    session.current_speaker = 1
    session._current_speaker_channel = Channel.MIC
    session.active = {Channel.MIC: 1, Channel.LOOPBACK: 1}
    session.profile_learning = None
    session.log = SimpleNamespace(speakers_with_lines=lambda: set())

    assert session.remove_speaker(1) is None
    assert session.roster[1].active is False
    assert session.current_speaker is None
    assert session._current_speaker_channel is None
    assert session.active == {Channel.MIC: None, Channel.LOOPBACK: None}


def test_remove_speaker_blocks_while_sample_capture_in_progress():
    session = bare_session()
    session.roster = [Speaker("Alex", hotkey_slot=0), Speaker("Sam", hotkey_slot=1)]
    session.current_speaker = 0
    session.active = {Channel.MIC: 0, Channel.LOOPBACK: None}
    session.profile_learning = {"speaker_index": 1, "phase": "collecting"}
    session.log = SimpleNamespace(speakers_with_lines=lambda: set())

    assert session.remove_speaker(1) == "a voice sample is being captured for this speaker"


def test_remove_speaker_succeeds_once_then_reports_already_removed():
    session = bare_session()
    session.roster = [Speaker("Alex", hotkey_slot=0), Speaker("Sam", hotkey_slot=1)]
    session.current_speaker = 0
    session.active = {Channel.MIC: 0, Channel.LOOPBACK: None}
    session.profile_learning = None
    session.log = SimpleNamespace(speakers_with_lines=lambda: set())

    assert session.remove_speaker(1) is None
    assert session.roster[1].active is False
    assert session.remove_speaker(1) == "speaker already removed"


def test_cancel_profile_learning_delegates_to_pipeline():
    session = bare_session()
    calls = []
    session._pipeline = SimpleNamespace(cancel_profile_learning=lambda: calls.append(True))

    session.cancel_profile_learning()

    assert calls == [True]


def _rename_session(roster):
    session = bare_session()
    session.roster = list(roster)
    session.renamed = []
    session.log_renames = []
    session._attributor = SimpleNamespace(
        rename=lambda i, n: session.renamed.append((i, n))
    )
    session.log = SimpleNamespace(rename=lambda mapping: session.log_renames.append(mapping))
    session._options = SimpleNamespace(speaker_store=None)
    return session


def test_rename_speaker_renames_roster_attributor_and_past_lines():
    session = _rename_session([Speaker("Guest 1", hotkey_slot=0)])

    assert session.rename_speaker(0, "  Sarah  ") is None
    assert session.roster[0].name == "Sarah"
    assert session.renamed == [(0, "Sarah")]
    assert session.log_renames == [{"Guest 1": "Sarah"}]


def test_rename_speaker_validates_index_and_name():
    session = _rename_session([Speaker("Guest 1", hotkey_slot=0)])
    assert session.rename_speaker(5, "Sarah") == "speaker not found"
    assert session.rename_speaker(0, "   ") == "a name is required"
    assert session.rename_speaker(0, "Guest 1") is None  # unchanged is a no-op
    assert session.log_renames == []


def test_rename_speaker_syncs_the_library_profile_for_an_enrolled_speaker():
    session = _rename_session([Speaker("Bob", speaker_id="sp-bob", hotkey_slot=0)])
    store_calls = []
    session._options = SimpleNamespace(
        speaker_store=SimpleNamespace(
            rename_speaker=lambda sid, name: store_calls.append((sid, name))
        )
    )

    assert session.rename_speaker(0, "Bobby") is None
    assert store_calls == [("sp-bob", "Bobby")]


def test_add_note_timestamps_and_fires_event():
    session = bare_session()
    session.clock = SimpleNamespace(now=lambda: 75.0)
    notes = []
    session.log = SimpleNamespace(add_note=lambda ts, text: notes.append((ts, text)))
    fired = []
    session._events = SimpleNamespace(on_note=fired.append)

    note = session.add_note("check the budget numbers")

    assert notes == [(75.0, "check the budget numbers")]
    assert note == {"time": "00:01:15", "text": "check the budget numbers"}
    assert fired == [note]


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
    session.current_speaker = None
    session._current_speaker_channel = None
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
    assert session.current_speaker == 1
    assert selected == [1]


def test_delayed_old_attribution_cannot_override_manual_selection():
    session = bare_session()
    session.active = {Channel.MIC: 1, Channel.LOOPBACK: None}
    session.current_speaker = 1
    session._current_speaker_channel = None
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
    assert session.current_speaker == 1
    assert session._manual_override_pending[Channel.MIC] is False


def test_rolling_attribution_updates_active_but_defers_to_manual_selection():
    session = bare_session()
    session.active = {Channel.MIC: 0, Channel.LOOPBACK: None}
    session.current_speaker = 0
    session._current_speaker_channel = Channel.MIC
    session._manual_override_pending = {
        Channel.MIC: False,
        Channel.LOOPBACK: False,
    }
    attributed = []
    session._events = SimpleNamespace(on_attribution=lambda *args: attributed.append(args))

    session._handle_tracking_attribution(1, Channel.MIC, "auto", 0.91)
    assert session.active[Channel.MIC] == 1
    assert session.current_speaker == 1
    assert attributed == [(1, Channel.MIC, "auto", 0.91)]

    session._manual_override_pending[Channel.MIC] = True
    session._handle_tracking_attribution(0, Channel.MIC, "auto", 0.95)
    assert session.active[Channel.MIC] == 1
    assert session.current_speaker == 1
    assert len(attributed) == 1


def test_unknown_on_other_channel_does_not_clear_current_speaker():
    session = bare_session()
    session.active = {Channel.MIC: 1, Channel.LOOPBACK: 0}
    session.current_speaker = 1
    session._current_speaker_channel = Channel.MIC
    session._manual_override_pending = {
        Channel.MIC: False,
        Channel.LOOPBACK: False,
    }
    attributed = []
    session._events = SimpleNamespace(
        on_attribution=lambda *args: attributed.append(args),
        on_segment=lambda *args: None,
    )
    session.log = SimpleNamespace(add=lambda segment: None)
    session._label_channels = False

    session._handle_segment(
        TranscriptSegment(
            "other channel ended",
            Channel.LOOPBACK,
            0.0,
            1.0,
            attribution="unknown",
        ),
        0.1,
    )

    assert session.active[Channel.LOOPBACK] is None
    assert session.current_speaker == 1
    assert attributed[-1][:3] == (None, Channel.LOOPBACK, "unknown")
