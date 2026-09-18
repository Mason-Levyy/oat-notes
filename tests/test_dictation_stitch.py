import pytest

from oat_notes.dictation.stitch import Piece, stitch_pieces

CLAUSE_GAP = 0.7
SENTENCE_GAP = 1.2


def stitch(*pieces):
    return stitch_pieces(list(pieces), CLAUSE_GAP, SENTENCE_GAP)


def test_a_breath_does_not_end_the_sentence():
    assert stitch(
        Piece("So I was thinking."),
        Piece("That we should ship on Thursday.", pause_before=0.3),
    ) == "So I was thinking that we should ship on Thursday."


def test_a_long_pause_does_end_the_sentence():
    assert stitch(
        Piece("Let's ship on Thursday"),
        Piece("QA needs the time.", pause_before=1.5),
    ) == "Let's ship on Thursday. QA needs the time."


def test_a_medium_pause_trusts_whisper():
    assert stitch(
        Piece("Let's ship on Thursday."),
        Piece("QA needs the time.", pause_before=0.9),
    ) == "Let's ship on Thursday. QA needs the time."
    assert stitch(
        Piece("Let's ship on Thursday"),
        Piece("if QA has the time.", pause_before=0.9),
    ) == "Let's ship on Thursday if QA has the time."


def test_a_forced_split_carries_no_pause_and_so_never_breaks():
    assert stitch(
        Piece("The number we saw yesterday."),
        Piece("Looked high to me.", pause_before=0.0),
    ) == "The number we saw yesterday looked high to me."


def test_the_pronoun_i_keeps_its_capital():
    assert stitch(
        Piece("When it lands."),
        Piece("I'll let you know.", pause_before=0.2),
    ) == "When it lands I'll let you know."


def test_an_acronym_keeps_its_capitals():
    assert stitch(
        Piece("Hand it to."),
        Piece("QA first.", pause_before=0.2),
    ) == "Hand it to QA first."


def test_a_question_or_exclamation_still_ends_the_sentence():
    assert stitch(Piece("Does that work?"), Piece("Great.", pause_before=0.2)) == (
        "Does that work? Great."
    )


def test_an_ellipsis_trails_off_into_the_next_words():
    assert stitch(Piece("Well..."), Piece("Maybe.", pause_before=0.2)) == "Well... maybe."


def test_no_full_stop_is_invented_for_a_short_fragment():
    assert stitch(Piece("Okay"), Piece("let's go with option B.", pause_before=2.0)) == (
        "Okay Let's go with option B."
    )


def test_empty_pieces_are_skipped():
    assert stitch(Piece(""), Piece("hello", pause_before=0.1), Piece("  ")) == "hello"
    assert stitch() == ""


@pytest.mark.parametrize("pause", [0.0, 0.3, 0.69])
def test_anything_under_the_clause_gap_continues(pause):
    assert stitch(Piece("One."), Piece("Two.", pause_before=pause)) == "One two."
