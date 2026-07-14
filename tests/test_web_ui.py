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
