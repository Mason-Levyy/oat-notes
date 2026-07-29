import math
from types import SimpleNamespace

import numpy as np

from oat_notes.attribution import Speaker
from oat_notes.session import UNKNOWN_TURN_MEMORY, Session
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


def _rename_session(roster, guest_indices=()):
    session = bare_session()
    session.roster = list(roster)
    session.guest_indices = list(guest_indices)
    session._speaker_resolver = None
    session.renamed = []
    session.log_renames = []
    session._attributor = SimpleNamespace(
        rename=lambda i, n: session.renamed.append((i, n))
    )
    session.log = SimpleNamespace(rename=lambda mapping: session.log_renames.append(mapping))
    session._options = SimpleNamespace(speaker_store=None)
    return session


class FakeStore:
    """Stands in for SpeakerStore — the speaker directory, no SQLite."""

    def __init__(self, speakers=()):
        self.speakers = list(speakers)
        self.created = []

    def list_speakers(self):
        return tuple(self.speakers)

    def create_speaker(self, name):
        self.created.append(name)
        profile = SimpleNamespace(speaker_id=f"sp-{name.lower()}", name=name)
        self.speakers.append(profile)
        return profile


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


def test_naming_a_guest_identifies_them_and_clears_the_backfill_queue():
    session = _rename_session([Speaker("Guest 1", hotkey_slot=0)], guest_indices=[0])
    store = FakeStore()
    session._options = SimpleNamespace(speaker_store=store)
    persisted = []
    session._speaker_resolver = SimpleNamespace(
        persist_guest=lambda index, sid: persisted.append((index, sid))
    )

    assert session.rename_speaker(0, "Sarah") is None
    assert session.roster[0].name == "Sarah"
    assert store.created == ["Sarah"]
    assert session.roster[0].speaker_id == "sp-sarah"
    assert persisted == [(0, "sp-sarah")]
    assert session.guest_indices == []


def test_naming_a_guest_an_existing_person_links_rather_than_duplicating():
    session = _rename_session([Speaker("Guest 1", hotkey_slot=0)], guest_indices=[0])
    store = FakeStore([SimpleNamespace(speaker_id="sp-priya", name="Priya")])
    session._options = SimpleNamespace(speaker_store=store)
    session._speaker_resolver = SimpleNamespace(persist_guest=lambda index, sid: None)

    assert session.rename_speaker(0, "priya") is None
    assert store.created == []
    assert session.roster[0].speaker_id == "sp-priya"
    assert session.guest_indices == []


def test_an_explicit_identity_wins_over_the_typed_name():
    session = _rename_session([Speaker("Guest 1", hotkey_slot=0)], guest_indices=[0])
    store = FakeStore()
    session._options = SimpleNamespace(speaker_store=store)
    session._speaker_resolver = SimpleNamespace(persist_guest=lambda index, sid: None)

    assert session.rename_speaker(0, "Sarah", speaker_id="sp-chosen") is None
    assert store.created == []
    assert session.roster[0].speaker_id == "sp-chosen"


def test_an_unusable_voice_sample_does_not_cost_the_guest_their_name():
    session = _rename_session([Speaker("Guest 1", hotkey_slot=0)], guest_indices=[0])
    session._options = SimpleNamespace(speaker_store=FakeStore())

    def explode(index, sid):
        raise ValueError("voice samples require at least one second of speech")

    session._speaker_resolver = SimpleNamespace(persist_guest=explode)

    assert session.rename_speaker(0, "Sarah") is None
    assert session.roster[0].name == "Sarah"
    assert session.guest_indices == []


def test_guest_numbering_survives_naming_an_earlier_guest():
    session = bare_session()
    session.roster = []
    session.guest_indices = []
    session._guests_created = 0
    session._attributor = SimpleNamespace(add=lambda s: len(session.roster))

    session._create_guest(hotkey_slot=0)
    session.guest_indices.remove(0)
    session._create_guest(hotkey_slot=1)

    assert [speaker.name for speaker in session.roster] == ["Guest 1", "Guest 2"]


def _reset_session(roster, resolver=None):
    session = bare_session()
    session.roster = list(roster)
    session.profile_learning = None
    session.current_speaker = 0
    session._current_speaker_channel = Channel.MIC
    session.active = {Channel.MIC: 0, Channel.LOOPBACK: 0}
    session._speaker_resolver = resolver
    session._pipeline = SimpleNamespace(
        reset_speaker_tracking=lambda index=None: session.tracking_resets.append(index)
    )
    session.tracking_resets = []
    return session


def test_resetting_a_profile_clears_it_and_the_live_tracking_state():
    reset = []
    session = _reset_session(
        [Speaker("Sarah", speaker_id="sp-1", hotkey_slot=0)],
        resolver=SimpleNamespace(reset_profile=reset.append),
    )

    assert session.reset_speaker_profile(0) is None

    assert reset == [0]
    assert session.tracking_resets == [None]
    assert session.current_speaker is None
    assert session._current_speaker_channel is None
    assert session.active == {Channel.MIC: None, Channel.LOOPBACK: None}


def test_resetting_a_profile_validates_the_target():
    session = _reset_session(
        [Speaker("Sarah", hotkey_slot=0), Speaker("Gone", hotkey_slot=1, active=False)],
        resolver=SimpleNamespace(reset_profile=lambda index: None),
    )

    assert session.reset_speaker_profile(9) == "speaker not found"
    assert session.reset_speaker_profile(1) == "speaker already removed"


def test_resetting_a_profile_waits_for_an_in_flight_sample():
    session = _reset_session(
        [Speaker("Sarah", hotkey_slot=0)],
        resolver=SimpleNamespace(reset_profile=lambda index: None),
    )
    session.profile_learning = {"speaker_index": 0, "phase": "collecting"}

    assert session.reset_speaker_profile(0) == (
        "a voice sample is being captured for this speaker"
    )


def test_resetting_a_profile_needs_speaker_recognition():
    session = _reset_session([Speaker("Sarah", hotkey_slot=0)], resolver=None)
    assert session.reset_speaker_profile(0) == "speaker recognition is unavailable"


def _backfill_session(roster, store, unknowns):
    session = bare_session()
    session.roster = list(roster)
    session._options = SimpleNamespace(speaker_store=store)
    session._speaker_resolver = object()
    session._unknown_turns = dict(unknowns)
    session.relabelled = []
    session.log = SimpleNamespace(
        relabel=lambda line_id, name: (
            session.relabelled.append((line_id, name)) or True
        )
    )
    session._events = SimpleNamespace(
        on_line_relabelled=lambda line_id, name, index, score: None
    )
    return session


class VectorStore:
    """Stands in for SpeakerStore's matching surface."""

    def __init__(self, vectors):
        self.vectors = vectors

    def profile_vector(self, speaker_id, source):
        embedding = self.vectors.get(speaker_id)
        if embedding is None:
            return None
        return SimpleNamespace(speaker_id=speaker_id, embedding=embedding)

    def match_vectors(self, speaker_ids, source):
        return tuple(
            self.profile_vector(speaker_id, source)
            for speaker_id in speaker_ids
            if speaker_id in self.vectors
        )


def test_an_unknown_line_is_named_once_the_profile_explains_it():
    store = VectorStore({"sp-sarah": np.array([1.0, 0.0], dtype=np.float32)})
    session = _backfill_session(
        [Speaker("Sarah", speaker_id="sp-sarah", hotkey_slot=0)],
        store,
        {7: (Channel.MIC, np.array([1.0, 0.0], dtype=np.float32))},
    )

    session._rescore_unknown_turns(0)

    assert session.relabelled == [(7, "Sarah")]
    assert session._unknown_turns == {}


def test_a_near_miss_is_left_unknown_rather_than_guessed():
    weak = np.array([0.65, math.sqrt(1 - 0.65**2)], dtype=np.float32)
    store = VectorStore({"sp-sarah": np.array([1.0, 0.0], dtype=np.float32)})
    session = _backfill_session(
        [Speaker("Sarah", speaker_id="sp-sarah", hotkey_slot=0)], store, {7: (Channel.MIC, weak)}
    )

    session._rescore_unknown_turns(0)

    assert session.relabelled == []
    assert 7 in session._unknown_turns


def test_a_line_two_people_both_match_stays_unknown():
    store = VectorStore(
        {
            "sp-sarah": np.array([1.0, 0.0], dtype=np.float32),
            "sp-alex": np.array([0.99, 0.141], dtype=np.float32),
        }
    )
    session = _backfill_session(
        [
            Speaker("Sarah", speaker_id="sp-sarah", hotkey_slot=0),
            Speaker("Alex", speaker_id="sp-alex", hotkey_slot=1),
        ],
        store,
        {7: (Channel.MIC, np.array([1.0, 0.0], dtype=np.float32))},
    )

    session._rescore_unknown_turns(0)

    assert session.relabelled == []


def test_only_saved_samples_trigger_a_rescore():
    session = bare_session()
    rescored = []
    session._rescore_unknown_turns = rescored.append
    session.profile_learning = None
    session._events = SimpleNamespace(on_profile_learning=lambda update: None)

    session._handle_profile_learning(
        SimpleNamespace(phase="collecting", speaker_index=2, to_dict=lambda: {})
    )
    assert rescored == []

    session._handle_profile_learning(
        SimpleNamespace(phase="saved", speaker_index=2, to_dict=lambda: {})
    )
    assert rescored == [2]


def test_the_unknown_turn_memory_is_bounded():
    session = bare_session()
    session._unknown_turns = {}
    for line_id in range(UNKNOWN_TURN_MEMORY + 25):
        session._remember_unknown(
            line_id, Channel.MIC, np.array([1.0, 0.0], dtype=np.float32)
        )

    assert len(session._unknown_turns) == UNKNOWN_TURN_MEMORY
    assert 0 not in session._unknown_turns
    assert UNKNOWN_TURN_MEMORY + 24 in session._unknown_turns


def test_assigning_a_line_by_hand_validates_and_settles_it():
    session = bare_session()
    session.roster = [Speaker("Sarah", hotkey_slot=0), Speaker("Gone", hotkey_slot=1, active=False)]
    session._unknown_turns = {4: (Channel.MIC, np.array([1.0, 0.0], dtype=np.float32))}
    relabelled = []
    session.log = SimpleNamespace(
        relabel=lambda line_id, name: (relabelled.append((line_id, name)) or True)
    )

    assert session.assign_line(4, 9) == "speaker not found"
    assert session.assign_line(4, 1) == "speaker already removed"
    assert session.assign_line(4, 0) is None

    assert relabelled == [(4, "Sarah")]
    assert session._unknown_turns == {}


def test_a_line_can_be_put_back_to_unknown():
    session = bare_session()
    session.roster = [Speaker("Sarah", hotkey_slot=0)]
    session._unknown_turns = {}
    relabelled = []
    session.log = SimpleNamespace(
        relabel=lambda line_id, name: (relabelled.append((line_id, name)) or True)
    )

    assert session.assign_line(4, None) is None
    assert relabelled == [(4, "Unknown")]
