import time

from oat_notes.cleanup import DROP_TOKEN, CleanupWorker, postprocess_output


class FakeCleaner:
    """Stands in for TranscriptCleaner — no model download or inference."""

    def __init__(self, transform):
        self._transform = transform

    def clean(self, text):
        return self._transform(text)


def test_postprocess_keeps_a_normal_cleaned_reply():
    original = "So, um, the budget is fine."
    assert postprocess_output(" So the budget is fine. ", original) == (
        "So the budget is fine."
    )


def test_postprocess_unwraps_a_quoted_reply():
    assert postprocess_output('"Hello there."', "Hello, um, there.") == "Hello there."


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


def test_worker_cleans_and_records_results():
    events = []
    worker = CleanupWorker(
        FakeCleaner(str.upper), 0.0, lambda i, t: events.append((i, t))
    )
    worker.start()
    worker.submit(0, "hello")
    worker.submit(1, "world")
    worker.finish()

    assert worker.results() == {0: "HELLO", 1: "WORLD"}
    assert events == [(0, "HELLO"), (1, "WORLD")]


def test_worker_skips_unchanged_lines():
    events = []
    worker = CleanupWorker(
        FakeCleaner(lambda t: t), 0.0, lambda i, t: events.append((i, t))
    )
    worker.start()
    worker.submit(0, "already clean")
    worker.finish()

    assert worker.results() == {}
    assert events == []


def test_worker_records_dropped_lines_as_none():
    worker = CleanupWorker(FakeCleaner(lambda t: None), 0.0, lambda i, t: None)
    worker.start()
    worker.submit(3, "asdfjkl")
    worker.finish()

    assert worker.results() == {3: None}


def test_worker_error_leaves_line_raw_and_hides_captured_words(capsys):
    def boom(text):
        raise RuntimeError("captured words must stay private")

    events = []
    worker = CleanupWorker(FakeCleaner(boom), 0.0, lambda i, t: events.append(t))
    worker.start()
    worker.submit(0, "hello")
    worker.finish()

    assert worker.results() == {}
    assert events == []
    error_log = capsys.readouterr().err
    assert "RuntimeError" in error_log
    assert "captured words must stay private" not in error_log


def test_worker_full_queue_skips_silently():
    worker = CleanupWorker(FakeCleaner(str.upper), 0.0, lambda i, t: None)
    for index in range(70):
        worker.submit(index, f"line {index}")
    worker.start()
    worker.finish()

    assert len(worker.results()) == 64


def test_finish_drains_without_waiting_out_the_delay():
    worker = CleanupWorker(FakeCleaner(str.upper), 60.0, lambda i, t: None)
    worker.start()
    worker.submit(0, "hello")
    started = time.monotonic()
    worker.finish()

    assert time.monotonic() - started < 5.0
    assert worker.results() == {0: "HELLO"}
