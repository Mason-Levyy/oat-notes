import threading

from oat_notes.line_feed import LineFeed


def test_snapshot_returns_copies():
    feed = LineFeed()
    feed.append({"id": 0, "label": "Alex", "text": "hi"})
    lines, _ = feed.snapshot()
    lines[0]["text"] = "changed"
    assert feed.snapshot()[0][0]["text"] == "hi"


def test_update_reports_before_and_after():
    feed = LineFeed()
    feed.append({"id": 3, "label": "Unknown", "text": "raw"})
    before, after = feed.update(3, text="clean")
    assert before["text"] == "raw" and after["text"] == "clean"
    assert feed.update(99, text="nope") is None


def test_drop_and_relabel():
    feed = LineFeed()
    feed.append({"id": 0, "label": "Guest 1", "text": "a"})
    feed.append({"id": 1, "label": "Alex", "text": "b"})
    assert feed.drop(0) is True
    assert feed.drop(0) is False
    feed.relabel({"Alex": "Alexandra"})
    assert feed.snapshot()[0] == [{"id": 1, "label": "Alexandra", "text": "b"}]


def test_reset_clears_notes_too():
    feed = LineFeed()
    feed.add_note({"time": "00:00", "text": "todo"})
    feed.reset()
    assert feed.snapshot() == ([], [])


def test_concurrent_appends_and_relabels_lose_nothing():
    feed = LineFeed()
    stop = threading.Event()

    def relabel_forever():
        while not stop.is_set():
            feed.relabel({"A": "B", "B": "A"})

    worker = threading.Thread(target=relabel_forever)
    worker.start()
    for line_id in range(500):
        feed.append({"id": line_id, "label": "A", "text": ""})
        feed.update(line_id, text=str(line_id))
    stop.set()
    worker.join()
    lines, _ = feed.snapshot()
    assert [line["id"] for line in lines] == list(range(500))
    assert all(line["text"] == str(line["id"]) for line in lines)
