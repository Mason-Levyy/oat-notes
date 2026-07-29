from oat_notes.repetition import collapse_repeated_runs


def test_a_dictated_sentence_printed_five_times_collapses_to_one():
    sentence = "Thanks for the update."
    assert collapse_repeated_runs(" ".join([sentence] * 5)) == sentence


def test_a_long_meeting_loop_collapses():
    line = "The churn number looks high to me."
    assert collapse_repeated_runs(" ".join([line] * 30)) == line


def test_the_run_still_collapses_when_only_the_last_one_is_punctuated():
    assert (
        collapse_repeated_runs("let's ship it let's ship it let's ship it.")
        == "let's ship it"
    )


def test_capitalisation_does_not_hide_a_loop():
    assert collapse_repeated_runs("Send it over. send it over.") == "Send it over."


def test_a_phrase_said_twice_for_emphasis_survives():
    # Two words twice is ordinary speech, not a decoder loop.
    assert collapse_repeated_runs("thank you thank you") == "thank you thank you"
    assert collapse_repeated_runs("no no") == "no no"


def test_a_single_word_needs_three_before_it_reads_as_a_loop():
    assert collapse_repeated_runs("had had a point") == "had had a point"
    assert collapse_repeated_runs("yeah yeah yeah yeah") == "yeah"


def test_only_the_repeated_run_is_touched():
    assert (
        collapse_repeated_runs("First, the plan. The plan. The plan. Then we go.")
        == "First, the plan. Then we go."
    )


def test_text_without_repetition_is_returned_byte_for_byte():
    original = "We should  revisit the model assumptions on Tuesday."
    assert collapse_repeated_runs(original) == original


def test_lines_are_collapsed_independently():
    assert (
        collapse_repeated_runs("ship it. ship it. ship it.\nreally ship it")
        == "ship it.\nreally ship it"
    )


def test_empty_and_whitespace_are_left_alone():
    assert collapse_repeated_runs("") == ""
    assert collapse_repeated_runs("   ") == "   "
