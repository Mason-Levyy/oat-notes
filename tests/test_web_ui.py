from pathlib import Path


HTML = (
    Path(__file__).parents[1] / "src" / "oat_notes" / "web" / "index.html"
).read_text(encoding="utf-8")


def test_group_picker_is_a_roster_preset_before_individual_people():
    assert HTML.index('id="group-select"') < HTML.index('id="person-search"')
    assert 'id="add-group-btn"' not in HTML
    assert '$("group-select").onchange' in HTML
    assert "setupRoster = [];" in HTML
    assert "SELECT GROUP TO FILL ROSTER" in HTML


def test_live_roster_highlights_only_the_current_speaker():
    assert "if (index === state.active_speaker)" in HTML
    assert "index === state.actives.mic || index === state.actives.loopback" not in HTML
    assert "state.active_speaker = Object.prototype.hasOwnProperty.call" in HTML
    assert 'Object.prototype.hasOwnProperty.call(event, "active_speaker")' in HTML


def test_transcript_prepends_newest_lines_and_preserves_scrolled_reading():
    assert "const anchor = cursor ? cursor.nextSibling : screen.firstChild;" in HTML
    assert "screen.insertBefore(div, anchor);" in HTML
    assert "else screen.scrollTop += screen.scrollHeight - previousHeight;" in HTML
    assert "$(\"screen\").insertBefore(cursor, $(\"screen\").firstChild);" in HTML


def test_profile_learning_feedback_uses_the_new_three_and_five_second_rules():
    assert "profile_learning: null" in HTML
    assert "LEARN ${Number(learning.speech_seconds || 0).toFixed(1)}" in HTML
    assert "SAMPLE ADDED" in HTML
    assert "SAMPLE SKIPPED" in HTML
    assert "ready_seconds: 5.0" in HTML
