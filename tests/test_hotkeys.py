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


def make_listener():
    received = []
    listener = HotkeyListener(received.append)
    return listener, received


def press_chord(listener, key):
    listener._handle_press(CTRL)
    listener._handle_press(ALT)
    listener._handle_press(key)
    listener._handle_release(ALT)
    listener._handle_release(CTRL)


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
