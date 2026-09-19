"""Where a JSON body becomes typed values, and where an API failure gets
its HTTP status. Every handler parses through here before touching state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

Body = Mapping[str, Any]


class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.CONFLICT) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def bad_request(message: str) -> ApiError:
    return ApiError(message, HTTPStatus.BAD_REQUEST)


def not_found(message: str) -> ApiError:
    return ApiError(message, HTTPStatus.NOT_FOUND)


def conflict(message: str) -> ApiError:
    return ApiError(message, HTTPStatus.CONFLICT)


@dataclass(frozen=True)
class RosterEntry:
    name: str
    speaker_id: str | None
    hotkey_slot: int


def parse_text(body: Body, field: str, *, required_message: str | None = None) -> str:
    """A stripped string; empty is an error when ``required_message`` is given."""
    value = str(body.get(field) or "").strip()
    if not value and required_message is not None:
        raise bad_request(required_message)
    return value


def parse_optional_id(body: Body, field: str = "speaker_id") -> str | None:
    return parse_text(body, field) or None


def parse_id(body: Body, field: str = "id") -> str:
    return parse_text(body, field, required_message=f"{field} is required")


def parse_int(body: Body, field: str) -> int:
    value = body.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise bad_request(f"{field} must be an integer")
    return value


def parse_optional_int(body: Body, field: str) -> int | None:
    value = body.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise bad_request(f"{field} must be an integer or null")
    return value


def parse_flag(body: Body, field: str, default: bool) -> bool:
    value = body.get(field, default)
    if not isinstance(value, bool):
        raise bad_request(f"{field} must be true or false")
    return value


def parse_direction(body: Body) -> int:
    value = body.get("direction")
    if isinstance(value, bool) or value not in (-1, 1):
        raise bad_request("direction must be -1 or 1")
    return value


def parse_members(body: Body) -> list[Body]:
    members = body.get("members", [])
    if not isinstance(members, list) or not all(isinstance(item, dict) for item in members):
        raise bad_request("group members must be a list of objects")
    return members


def parse_renames(body: Body) -> dict[str, str]:
    renames = parse_object(body, "renames")
    return {str(old): str(new).strip() for old, new in renames.items() if str(new).strip()}


def parse_object(body: Body, field: str) -> Body:
    value = body.get(field)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise bad_request(f"{field} must be an object")
    return value


def parse_roster(body: Body) -> list[RosterEntry]:
    """Setup-screen roster: entries without a name are skipped, a saved
    speaker's name is resolved later from the library."""
    entries = body.get("speakers", [])
    if not isinstance(entries, list):
        raise bad_request("speakers must be a list")
    roster: list[RosterEntry] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        speaker_id = parse_optional_id(entry)
        name = parse_text(entry, "name")
        if not name and not speaker_id:
            continue
        slot = entry.get("hotkey_slot", position)
        hotkey_slot = slot if isinstance(slot, int) and not isinstance(slot, bool) else position
        roster.append(RosterEntry(name, speaker_id, hotkey_slot))
    return roster
