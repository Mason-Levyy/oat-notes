import pytest

from oat_notes.webapp.requests import (
    ApiError,
    RosterEntry,
    parse_direction,
    parse_flag,
    parse_id,
    parse_int,
    parse_members,
    parse_object,
    parse_optional_id,
    parse_renames,
    parse_roster,
    parse_text,
)


def rejects(call, message):
    with pytest.raises(ApiError, match=message) as raised:
        call()
    assert raised.value.status == 400


def test_text_is_stripped_and_optionally_required():
    assert parse_text({"name": "  Sarah "}, "name") == "Sarah"
    assert parse_text({}, "name") == ""
    rejects(lambda: parse_text({"name": " "}, "name", required_message="a name is required"),
            "a name is required")


def test_ids():
    assert parse_id({"id": " sp-1 "}) == "sp-1"
    rejects(lambda: parse_id({}), "id is required")
    assert parse_optional_id({"speaker_id": ""}) is None
    assert parse_optional_id({"speaker_id": "sp-2"}) == "sp-2"


def test_ints_reject_bools_and_strings():
    assert parse_int({"index": 3}, "index") == 3
    rejects(lambda: parse_int({"index": "3"}, "index"), "index must be an integer")
    rejects(lambda: parse_int({"index": True}, "index"), "index must be an integer")
    rejects(lambda: parse_int({}, "index"), "index must be an integer")
    assert parse_int({}, "index", optional=True) is None
    rejects(lambda: parse_int({"index": 1.5}, "index", optional=True), "or null")


def test_flags_and_direction():
    assert parse_flag({}, "discard", False) is False
    assert parse_flag({"discard": True}, "discard", False) is True
    rejects(lambda: parse_flag({"discard": 1}, "discard", False), "must be true or false")
    assert parse_direction({"direction": -1}) == -1
    rejects(lambda: parse_direction({"direction": True}), "direction must be -1 or 1")
    rejects(lambda: parse_direction({"direction": 2}), "direction must be -1 or 1")


def test_members_renames_and_objects():
    assert parse_members({"members": [{"speaker_id": "a"}]}) == [{"speaker_id": "a"}]
    rejects(lambda: parse_members({"members": ["a"]}), "group members must be a list")
    assert parse_renames({"renames": {"Guest 1": " Sarah ", "Guest 2": " "}}) == {"Guest 1": "Sarah"}
    rejects(lambda: parse_renames({"renames": []}), "renames must be an object")
    assert parse_object({}, "guest_profiles") == {}
    rejects(lambda: parse_object({"guest_profiles": 1}, "guest_profiles"), "must be an object")


def test_roster_entries_keep_names_ids_and_slots():
    roster = parse_roster(
        {
            "speakers": [
                {"name": "Sarah", "hotkey_slot": 4},
                {"speaker_id": "sp-1"},
                {"name": " ", "hotkey_slot": True},
                "not an object",
                {"name": "Dev", "hotkey_slot": "x"},
            ]
        }
    )
    assert roster == [
        RosterEntry("Sarah", None, 4),
        RosterEntry("", "sp-1", 1),
        RosterEntry("Dev", None, 4),
    ]
    rejects(lambda: parse_roster({"speakers": "Sarah"}), "speakers must be a list")
