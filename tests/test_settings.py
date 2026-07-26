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


# -- dictation settings ----------------------------------------------------


def test_dictation_defaults():
    settings = AppSettings()
    assert settings.dictation_modifiers == ("ctrl", "win")
    assert settings.dictation_label == "Ctrl+Win"
    assert settings.dictation_email_label == "Ctrl+Shift+Win"
    assert settings.dictation_injection == "paste"
    assert settings.dictation_replay_label == "Ctrl+Alt+Z"


def test_every_default_survives_a_round_trip(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.save(AppSettings())
    assert store.load() == AppSettings()


def test_full_dictation_round_trip(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    settings = AppSettings(
        dictation_modifiers=("alt", "win"),
        dictation_email_modifiers=(),
        dictation_replay_modifiers=("alt", "shift"),
        dictation_tap_ms=250,
        dictation_injection="type",
        dictation_restore_clipboard=False,
        dictation_spoken_punctuation=True,
        dictation_vocabulary=(("levya", "Mason"), ("oat notes", "Oat Notes")),
        dictation_signature="Mason",
        overlay_enabled=False,
    )
    store.save(settings)
    assert store.load() == settings


# -- partial merge ---------------------------------------------------------


def test_merge_leaves_untouched_fields_alone():
    settings = AppSettings(
        hotkey_modifiers=("alt", "shift"), dictation_signature="Mason"
    )
    merged = settings.merged({"dictation_injection": "type"})
    assert merged.dictation_injection == "type"
    assert merged.hotkey_modifiers == ("alt", "shift")
    assert merged.dictation_signature == "Mason"


def test_merge_validates_the_result():
    with pytest.raises(ValueError):
        AppSettings().merged({"dictation_injection": "telepathy"})


def test_merge_rejects_a_non_object():
    with pytest.raises(ValueError):
        AppSettings().merged(["nope"])


# -- validation ------------------------------------------------------------


def test_colliding_chords_are_rejected():
    with pytest.raises(ValueError, match="choose different modifiers"):
        AppSettings.from_dict(
            {"hotkey_modifiers": ["ctrl", "win"], "dictation_modifiers": ["ctrl", "win"]}
        )


def test_dictation_and_email_chords_may_not_collide():
    with pytest.raises(ValueError, match="choose different modifiers"):
        AppSettings.from_dict(
            {
                "dictation_modifiers": ["ctrl", "win"],
                "dictation_email_modifiers": ["win", "ctrl"],
            }
        )


def test_an_empty_email_chord_disables_it():
    settings = AppSettings.from_dict({"dictation_email_modifiers": []})
    assert settings.dictation_email_modifiers == ()
    assert settings.dictation_email_label == "off"


@pytest.mark.parametrize(
    "payload",
    [
        {"dictation_injection": "telekinesis"},
        {"dictation_tap_ms": 5},
        {"dictation_tap_ms": 99999},
        {"dictation_tap_ms": "fast"},
        {"dictation_tap_ms": True},
        {"dictation_enabled": "yes"},
        {"dictation_signature": 42},
        {"dictation_vocabulary": "levya=Mason"},
        {"dictation_vocabulary": [["only-one-item"]]},
        {"dictation_vocabulary": [[1, 2]]},
    ],
)
def test_invalid_dictation_settings_are_rejected(payload):
    with pytest.raises(ValueError):
        AppSettings.from_dict(payload)


def test_vocabulary_accepts_the_ui_shape_and_drops_blanks():
    settings = AppSettings.from_dict(
        {"dictation_vocabulary": [["levya", "Mason"], ["  ", "ignored"], ["x", " y "]]}
    )
    assert settings.dictation_vocabulary == (("levya", "Mason"), ("x", "y"))


def test_vocabulary_accepts_objects():
    settings = AppSettings.from_dict(
        {"dictation_vocabulary": [{"heard": "levya", "written": "Mason"}]}
    )
    assert settings.dictation_vocabulary == (("levya", "Mason"),)


def test_unknown_keys_are_ignored():
    settings = AppSettings.from_dict(AppSettings().to_dict())
    assert settings == AppSettings()


def test_a_letter_chord_may_share_modifiers_with_a_digit_chord():
    settings = AppSettings.from_dict(
        {
            "hotkey_modifiers": ["ctrl", "alt"],
            "dictation_replay_modifiers": ["ctrl", "alt"],
        }
    )
    assert settings.dictation_replay_label == "Ctrl+Alt+Z"


def test_two_modifier_only_chords_still_collide():
    with pytest.raises(ValueError, match="choose different modifiers"):
        AppSettings.from_dict(
            {
                "dictation_modifiers": ["ctrl", "win"],
                "dictation_email_modifiers": ["ctrl", "win"],
            }
        )


def test_a_letter_chord_may_not_shadow_a_modifier_only_chord():
    with pytest.raises(ValueError, match="choose different modifiers"):
        AppSettings.from_dict(
            {
                "dictation_modifiers": ["ctrl", "alt"],
                "hotkey_modifiers": ["ctrl", "shift"],
                "dictation_replay_modifiers": ["ctrl", "alt"],
            }
        )


def test_an_empty_replay_chord_disables_it():
    settings = AppSettings.from_dict({"dictation_replay_modifiers": []})
    assert settings.dictation_replay_modifiers == ()
    assert settings.dictation_replay_label == "off"
