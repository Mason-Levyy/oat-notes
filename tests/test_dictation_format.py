import pytest

from oat_notes.dictation.format import (
    EMAIL_MODE,
    EMAIL_SCORE_HIGH,
    EMAIL_SCORE_LOW,
    TEXT_MODE,
    apply_rules,
    apply_vocabulary,
    detect_email,
    email_score,
    format_dictation,
    format_email,
    postprocess_email,
)


class FakeEngine:
    """Stands in for LlmEngine — the seam every LLM test in this repo uses."""

    def __init__(self, reply="", explode=False):
        self.reply = reply
        self.explode = explode
        self.calls = []

    def generate(self, system_prompt, prompt, max_new_tokens):
        if self.explode:
            raise RuntimeError("model unavailable")
        self.calls.append((system_prompt, prompt, max_new_tokens))
        return self.reply


# -- rule cleanup ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("um so I think we should ship it", "So I think we should ship it"),
        ("uh, the build is green", "The build is green"),
        ("it's, um, mostly done", "It's mostly done"),
        ("the the deployment failed", "The deployment failed"),
        ("I I I need more coffee", "I need more coffee"),
        ("  lots   of   space  ", "Lots of space"),
        ("hello world", "Hello world"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_rule_cleanup(raw, expected):
    assert apply_rules(raw) == expected


def test_legitimate_double_words_survive():
    assert apply_rules("I had had enough") == "I had had enough"
    assert apply_rules("the thing that that team built") == (
        "The thing that that team built"
    )


def test_fillers_inside_words_are_left_alone():
    assert apply_rules("the summary is umbrella shaped") == (
        "The summary is umbrella shaped"
    )


def test_no_trailing_period_is_invented():
    assert apply_rules("weather in london tomorrow") == "Weather in london tomorrow"


def test_spoken_line_commands():
    assert apply_rules("first line new line second line") == (
        "First line\nsecond line"
    )
    assert apply_rules("one new paragraph two") == "One\n\ntwo"


def test_spoken_punctuation_is_off_by_default():
    assert apply_rules("the period was short") == "The period was short"
    assert apply_rules("that works period", spoken_punctuation=True) == "That works."


def test_spoken_punctuation_when_enabled():
    assert apply_rules(
        "are you free comma tomorrow question mark", spoken_punctuation=True
    ) == "Are you free, tomorrow?"


# -- vocabulary ------------------------------------------------------------


def test_vocabulary_substitution_is_whole_word_and_case_insensitive():
    vocabulary = (("oat notes", "Oat Notes"), ("sherpa", "sherpa-onnx"))
    assert apply_vocabulary("i use OAT NOTES daily", vocabulary) == (
        "i use Oat Notes daily"
    )
    assert apply_vocabulary("sherpanoise", vocabulary) == "sherpanoise"


def test_vocabulary_runs_inside_apply_rules():
    assert apply_rules("um ask levya about it", (("levya", "Mason"),)) == (
        "Ask Mason about it"
    )


def test_vocabulary_ignores_blank_entries():
    assert apply_vocabulary("hello", (("", "x"), ("  ", "y"))) == "hello"


def test_vocabulary_replacement_with_backslashes_is_literal():
    assert apply_vocabulary("path", (("path", r"C:\temp"),)) == r"C:\temp"


# -- email scoring ---------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Hey Sarah, following up on the deployment we talked about yesterday. "
        "I think we should push it to Thursday so QA has time to run the full "
        "suite. Let me know if that works for you.",
        "Hi team, quick update on the migration. We finished the first half "
        "this morning and the rest should land by Friday afternoon. Thanks for "
        "your patience with the downtime while we sorted the indexes out.",
        "write an email to Dave about the invoice",
    ],
)
def test_obvious_emails_score_high(text):
    assert email_score(text) >= EMAIL_SCORE_HIGH


@pytest.mark.parametrize(
    "text",
    [
        "hey can you check this",
        "weather in london tomorrow",
        "git rebase interactive head three",
        "remind me to buy milk",
        "",
    ],
)
def test_obvious_non_emails_score_low(text):
    assert email_score(text) <= EMAIL_SCORE_LOW


def test_salutation_needs_an_addressee_and_a_comma():
    assert email_score("Hey Sarah, thanks") > email_score("Hey can you check this")


def test_explicit_request_is_decisive():
    assert email_score("send an email to legal") == 1.0


# -- email detection tiers -------------------------------------------------


def test_high_score_skips_the_model_entirely():
    engine = FakeEngine(reply="TEXT")
    text = (
        "Hey Sarah, following up on the deployment. I think we should push it "
        "to Thursday so QA has time to run the suite. Let me know if that works."
    )
    assert detect_email(text, engine) is True
    assert engine.calls == []


def test_low_score_skips_the_model_entirely():
    engine = FakeEngine(reply="EMAIL")
    assert detect_email("weather in london", engine) is False
    assert engine.calls == []


def test_ambiguous_text_asks_the_model():
    engine = FakeEngine(reply="EMAIL")
    text = "Hi team, quick update: the build is green."
    assert EMAIL_SCORE_LOW < email_score(text) < EMAIL_SCORE_HIGH
    assert detect_email(text, engine) is True
    assert len(engine.calls) == 1


def test_ambiguous_text_without_a_model_stays_plain():
    text = "Hi team, quick update: the build is green."
    assert detect_email(text, None) is False


def test_classifier_failure_falls_back_to_plain_text():
    assert detect_email("Hi team, quick update: the build is green.", FakeEngine(explode=True)) is False


# -- email formatting guardrails -------------------------------------------


def test_placeholder_output_is_rejected():
    assert postprocess_email("Hi Dave,\n\nThanks.\n\n[Your Name]", "Hi Dave thanks") is None
    assert postprocess_email("Hi <recipient>, thanks", "Hi Dave thanks") is None


def test_truncated_output_is_rejected():
    original = "a" * 100
    assert postprocess_email("short", original) is None


def test_expanded_output_is_accepted():
    original = "hey dave the build is green"
    expanded = "Hi Dave,\n\nJust letting you know the build is green.\n\nThanks,"
    assert postprocess_email(expanded, original) == expanded


@pytest.mark.parametrize(
    "reply",
    [
        "Subject: Build status\n\nHi Dave,\n\nThe build is green and ready.",
        "subject: build status\nHi Dave,\n\nThe build is green and ready.",
        "**Subject:** Build status\n\nHi Dave,\n\nThe build is green and ready.",
        "Sure, here is the email:\nSubject: Build\n\nHi Dave,\n\nThe build is green and ready.",
    ],
)
def test_subject_lines_are_stripped(reply):
    cleaned = postprocess_email(reply, "hey dave the build is green and ready")
    assert cleaned is not None
    assert "ubject" not in cleaned
    assert cleaned.startswith("Hi Dave,")


def test_a_body_mentioning_a_subject_is_left_alone():
    body = "Hi Dave,\n\nThe subject: line in your draft looks wrong to me."
    assert postprocess_email(body, "hey dave the subject line looks wrong") == body


def test_assistant_preamble_is_stripped():
    assert postprocess_email(
        "Sure, here is the email:\nHi Dave,\n\nThe build is green.",
        "hey dave the build is green",
    ) == "Hi Dave,\n\nThe build is green."


def test_empty_output_is_rejected():
    assert postprocess_email("   ", "something") is None


def test_format_email_falls_back_on_a_bad_reply():
    original = "Hey Dave, the build is green and ready to go out today."
    assert format_email(original, FakeEngine(reply="[Your Name]")) == original


def test_format_email_falls_back_when_the_model_throws():
    original = "Hey Dave, the build is green."
    assert format_email(original, FakeEngine(explode=True)) == original


def test_the_model_is_told_not_to_sign_a_name():
    engine = FakeEngine(reply="Hi Dave,\n\nThe build is green and ready.\n\nThanks,")
    format_email("Hey Dave the build is green and ready", engine)
    assert "do not sign a name" in engine.calls[0][1]


# -- the whole pipeline ----------------------------------------------------


def test_plain_dictation_never_touches_the_model():
    engine = FakeEngine(reply="should not be used")
    text, mode = format_dictation("um, git status please", engine=engine)
    assert (text, mode) == ("Git status please", TEXT_MODE)
    assert engine.calls == []


def test_detected_email_is_formatted():
    formatted = (
        "Hi Sarah,\n\nFollowing up on the deployment — I think we should push "
        "it to Thursday so QA has time to run the full suite.\n\nLet me know "
        "if that works for you.\n\nThanks,"
    )
    engine = FakeEngine(reply=formatted)
    raw = (
        "Hey Sarah, following up on the deployment. I think we should push it "
        "to Thursday so QA has time to run the suite. Let me know if that works."
    )
    text, mode = format_dictation(raw, engine=engine)
    assert mode == EMAIL_MODE and text == formatted


def test_force_email_overrides_detection():
    engine = FakeEngine(reply="Hi,\n\nThe build is green and ready to ship.\n\nThanks,")
    text, mode = format_dictation(
        "the build is green and ready to ship", engine=engine, force_email=True
    )
    assert mode == EMAIL_MODE


def test_force_email_without_a_loaded_model_still_inserts_text():
    text, mode = format_dictation("the build is green", engine=None, force_email=True)
    assert (text, mode) == ("The build is green", TEXT_MODE)


def test_detection_can_be_disabled():
    engine = FakeEngine(reply="should not be used")
    raw = (
        "Hey Sarah, following up on the deployment. I think we should push it "
        "to Thursday so QA has time. Let me know if that works for you."
    )
    text, mode = format_dictation(raw, engine=engine, detect=False)
    assert mode == TEXT_MODE and engine.calls == []


def test_empty_dictation_produces_nothing():
    assert format_dictation("   ", engine=FakeEngine()) == ("", TEXT_MODE)
