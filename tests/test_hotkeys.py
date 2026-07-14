from oat_notes.hotkeys import HotkeyListener


class FakeKey:
    def __init__(self, char=None, vk=None, name=""):
        if char is not None:
            self.char = char
        if vk is not None:
            self.vk = vk
        self.name = name


CTRL = FakeKey(name="ctrl_l")
ALT = FakeKey(name="alt_l")
SHIFT = FakeKey(name="shift_l")
WIN = FakeKey(name="cmd_l")


def make_listener(modifiers=("ctrl", "alt")):
    received = []
    listener = HotkeyListener(received.append, modifiers=modifiers)
    return listener, received


def press_chord(listener, key, modifier_keys=(CTRL, ALT)):
    for modifier in modifier_keys:
        listener._handle_press(modifier)
    listener._handle_press(key)
    for modifier in reversed(modifier_keys):
        listener._handle_release(modifier)


def test_chord_fires_switch():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="2"))
    assert received == [1]


def test_bare_digit_does_not_fire():
    listener, received = make_listener()
    listener._handle_press(FakeKey(char="2"))
    assert received == []


def test_digit_after_release_does_not_fire():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="1"))
    listener._handle_press(FakeKey(char="2"))
    assert received == [0]


def test_ctrl_only_does_not_fire():
    listener, received = make_listener()
    listener._handle_press(CTRL)
    listener._handle_press(FakeKey(char="3"))
    assert received == []


def test_numpad_vk_path():
    listener, received = make_listener()
    press_chord(listener, FakeKey(vk=99))
    assert received == [2]


def test_control_character_vk_fallback():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="\x11", vk=ord("1")))
    assert received == [0]


def test_zero_and_letters_ignored():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="0"))
    press_chord(listener, FakeKey(char="a"))
    assert received == []


def test_out_of_roster_digit_still_forwards():
    listener, received = make_listener()
    press_chord(listener, FakeKey(char="9"))
    assert received == [8]


def test_custom_ctrl_shift_chord_fires():
    listener, received = make_listener(("ctrl", "shift"))
    press_chord(listener, FakeKey(char="4"), (CTRL, SHIFT))
    assert received == [3]


def test_old_chord_does_not_fire_after_customization():
    listener, received = make_listener(("ctrl", "shift"))
    press_chord(listener, FakeKey(char="4"), (CTRL, ALT))
    assert received == []


def test_extra_modifier_does_not_match():
    listener, received = make_listener(("ctrl", "alt"))
    press_chord(listener, FakeKey(char="2"), (CTRL, ALT, SHIFT))
    assert received == []


def test_windows_modifier_is_supported():
    listener, received = make_listener(("win", "shift"))
    press_chord(listener, FakeKey(char="5"), (WIN, SHIFT))
    assert received == [4]


def test_brackets_page_speaker_banks():
    switched = []
    pages = []
    listener = HotkeyListener(switched.append, on_page=pages.append)
    press_chord(listener, FakeKey(char="["))
    press_chord(listener, FakeKey(char="]"))
    assert pages == [-1, 1]
    assert switched == []


def test_bracket_virtual_key_fallback():
    pages = []
    listener = HotkeyListener(lambda index: None, on_page=pages.append)
    press_chord(listener, FakeKey(vk=219))
    press_chord(listener, FakeKey(vk=221))
    assert pages == [-1, 1]
