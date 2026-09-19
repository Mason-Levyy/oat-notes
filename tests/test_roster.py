from dataclasses import replace

from oat_notes.attribution import Speaker
from oat_notes.roster import (
    assign_hotkey_slots,
    check_index,
    index_for,
    index_for_slot,
    next_free_slot,
)


def test_requested_slots_are_kept_and_the_rest_follow_the_highest():
    roster = assign_hotkey_slots(
        (Speaker("A", hotkey_slot=3), Speaker("B"), Speaker("C", hotkey_slot=3), Speaker("D", hotkey_slot=-1))
    )
    assert [speaker.hotkey_slot for speaker in roster] == [3, 4, 5, 6]
    plain = assign_hotkey_slots((Speaker("A"), Speaker("B", hotkey_slot=0), Speaker("C")))
    assert [speaker.hotkey_slot for speaker in plain] == [0, 1, 2]


def test_next_free_slot_prefers_the_current_bank_then_the_end():
    roster = [Speaker("A", hotkey_slot=0), Speaker("B", hotkey_slot=1)]
    assert next_free_slot(roster, bank=0) == 2
    assert next_free_slot(roster, bank=1) == 9
    full = [Speaker(str(slot), hotkey_slot=slot) for slot in range(9)]
    assert next_free_slot(full, bank=0) == 9


def test_lookups_skip_removed_members():
    roster = [
        replace(Speaker("Sarah", speaker_id="sp-1", hotkey_slot=0), active=False),
        Speaker("Dev", hotkey_slot=1),
    ]
    assert index_for_slot(roster, 0) is None
    assert index_for_slot(roster, 1) == 1
    assert index_for(roster, "sarah") is None
    assert index_for(roster, "DEV") == 1
    assert index_for(roster, "someone", speaker_id="sp-1") is None


def test_a_saved_identity_is_found_under_any_spelling():
    roster = [Speaker("Dev", speaker_id="sp-1"), Speaker("Sarah K", speaker_id="sp-2")]
    assert index_for(roster, "Sarah", speaker_id="sp-2") == 1
    assert index_for(roster, "Sarah") is None


def test_check_index_explains_why_not():
    roster = [Speaker("Sarah"), replace(Speaker("Dev"), active=False)]
    assert check_index(roster, 0) is None
    assert check_index(roster, 1) == "speaker already removed"
    assert check_index(roster, 2) == "speaker not found"
    assert check_index(roster, -1) == "speaker not found"
