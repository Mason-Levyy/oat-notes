import json

import pytest

from oat_notes.settings import AppSettings, SettingsStore, normalize_modifiers


def test_default_settings_use_ctrl_alt():
    settings = AppSettings()
    assert settings.hotkey_modifiers == ("ctrl", "alt")
    assert settings.hotkey_label == "Ctrl+Alt+1–9"


def test_modifiers_are_deduplicated_and_canonically_ordered():
    assert normalize_modifiers(["SHIFT", "ctrl", "shift"]) == ("ctrl", "shift")


@pytest.mark.parametrize(
    "value",
    [[], ["ctrl", "hyper"], "ctrl", [1]],
)
def test_invalid_modifiers_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_modifiers(value)


def test_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    settings = AppSettings(("alt", "shift"))

    store.save(settings)

    assert store.load() == settings
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["hotkey_modifiers"] == ["alt", "shift"]


def test_missing_or_corrupt_settings_use_defaults(tmp_path):
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    assert store.load() == AppSettings()

    path.write_text("not json", encoding="utf-8")
    assert store.load() == AppSettings()
