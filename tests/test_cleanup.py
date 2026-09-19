import time

from oat_notes.cleanup import (
    DROP_TOKEN,
    CleanupWorker,
    build_prompt,
    postprocess_output,
)


class FakeCleaner:
    """Stands in for TranscriptCleaner — no model download or inference.
    Records the context each call receives so tests can assert on it."""

    def __init__(self, transform):
        self._transform = transform
        self.calls = []

    def clean(self, text, before=(), after=()):
        self.calls.append((text, tuple(before), tuple(after)))
        return self._transform(text)


def drain(worker, submissions):
    for line_id, text in submissions:
        worker.submit(line_id, text)
    worker.start()
    worker.finish()


def test_postprocess_keeps_a_normal_cleaned_reply():
    original = "So, um, the budget is fine."
    assert postprocess_output(" So the budget is fine. ", original) == (
        "So the budget is fine."
    )


def test_postprocess_unwraps_a_quoted_reply():
    assert postprocess_output('"Hello there."', "Hello, um, there.") == "Hello there."


def test_postprocess_strips_a_line_label_prefix():
    assert postprocess_output("LINE: the budget is fine", "um the budget is fine") == (
        "the budget is fine"
    )


def test_postprocess_drop_token_drops_the_line():
    assert postprocess_output(DROP_TOKEN, "asdfjkl") is None
    assert postprocess_output(f" {DROP_TOKEN}. ", "asdfjkl") is None


def test_postprocess_drop_token_inside_a_sentence_is_not_a_drop():
    reply = f"He said {DROP_TOKEN} during the standup meeting yesterday"
    assert postprocess_output(reply, "He said drop during the standup meeting yesterday")


def test_postprocess_empty_reply_keeps_the_original():
    assert postprocess_output("   ", "keep me") == "keep me"


def test_postprocess_invented_content_keeps_the_original():
    original = "short line"
    invented = (
        "this reply is far longer than the original line and is clearly"
        " invented content the model made up"
    )
    assert postprocess_output(invented, original) == original


def test_build_prompt_without_context_is_just_the_line():
    assert build_prompt("hello") == "hello"


def test_build_prompt_frames_the_target_between_neighbours():
    prompt = build_prompt("target", before=["p1", "p2"], after=["n1", "n2"])
    assert "PREV: p1" in prompt and "PREV: p2" in prompt
    assert "LINE: target" in prompt
    assert "NEXT: n1" in prompt and "NEXT: n2" in prompt


def test_worker_cleans_and_records_results():
    events = []
    cleaner = FakeCleaner(str.upper)
    worker = CleanupWorker(cleaner, lambda i, t: events.append((i, t)))
    drain(worker, [(0, "hello"), (1, "world")])

    assert worker.results() == {0: "HELLO", 1: "WORLD"}
    assert events == [(0, "HELLO"), (1, "WORLD")]


def test_worker_feeds_cleaned_prior_and_raw_future_context():
    cleaner = FakeCleaner(str.upper)
    worker = CleanupWorker(cleaner, lambda i, t: None, context_before=2, context_after=2)
    drain(worker, [(i, ch) for i, ch in enumerate(["a", "b", "c", "d", "e"])])

    assert cleaner.calls == [
        ("a", (), ("b", "c")),
        ("b", ("A",), ("c", "d")),
        ("c", ("A", "B"), ("d", "e")),
        ("d", ("B", "C"), ("e",)),
        ("e", ("C", "D"), ()),
    ]


def test_worker_excludes_dropped_lines_from_prior_context():
    cleaner = FakeCleaner(lambda t: None if t == "junk" else t.upper())
    worker = CleanupWorker(cleaner, lambda i, t: None, context_before=3, context_after=0)
    drain(worker, [(0, "hello"), (1, "junk"), (2, "there"), (3, "friend")])

    last = cleaner.calls[-1]
    assert last[0] == "friend"
    assert last[1] == ("HELLO", "THERE")


def test_worker_skips_unchanged_lines():
    events = []
    worker = CleanupWorker(FakeCleaner(lambda t: t), lambda i, t: events.append((i, t)))
    drain(worker, [(0, "already clean")])

    assert worker.results() == {}
    assert events == []


def test_worker_records_dropped_lines_as_none():
    worker = CleanupWorker(FakeCleaner(lambda t: None), lambda i, t: None)
    drain(worker, [(3, "asdfjkl")])

    assert worker.results() == {3: None}


def test_worker_error_leaves_line_raw_and_hides_captured_words(capsys):
    def boom(text):
        raise RuntimeError("captured words must stay private")

    events = []
    worker = CleanupWorker(FakeCleaner(boom), lambda i, t: events.append(t))
    drain(worker, [(0, "hello")])

    assert worker.results() == {}
    assert events == []
    error_log = capsys.readouterr().err
    assert "RuntimeError" in error_log
    assert "captured words must stay private" not in error_log


def test_worker_full_queue_skips_silently():
    worker = CleanupWorker(FakeCleaner(str.upper), lambda i, t: None)
    for index in range(140):
        worker.submit(index, f"line {index}")
    worker.start()
    worker.finish()

    assert len(worker.results()) == 128


def test_worker_age_fallback_cleans_a_line_when_no_future_arrives():
    results = []
    cleaner = FakeCleaner(str.upper)
    worker = CleanupWorker(
        cleaner, lambda i, t: results.append((i, t)), context_after=2, max_wait_seconds=0.0
    )
    worker.start()
    worker.submit(0, "hello")
    deadline = time.monotonic() + 3.0
    while not results and time.monotonic() < deadline:
        time.sleep(0.05)
    worker.finish()

    assert results == [(0, "HELLO")]
    assert cleaner.calls[0] == ("hello", (), ())
