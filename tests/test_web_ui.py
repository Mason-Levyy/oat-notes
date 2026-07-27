from pathlib import Path


HTML = (
    Path(__file__).parents[1] / "src" / "oat_notes" / "web" / "index.html"
).read_text(encoding="utf-8")


def test_group_picker_is_a_roster_preset_before_individual_people():
    assert HTML.index('id="group-select"') < HTML.index('id="person-search"')
    assert 'id="add-group-btn"' not in HTML
    assert "setupRoster = [];" in HTML
    # A button expanding an inline list, not a native <select>.
    assert '<button type="button" class="row-btn" id="group-select"' in HTML
    assert '$("group-select").onclick' in HTML
    assert "function renderGroupOptions()" in HTML
    assert "LOAD A SAVED GROUP" in HTML
    assert "NO SAVED GROUPS" in HTML


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


def test_cleanup_updates_lines_in_place_with_a_toggle():
    assert 'event.type === "line_update"' in HTML
    assert 'event.type === "line_drop"' in HTML
    assert "div.dataset.lineId = line.id" in HTML
    assert 'id="cleanup"' in HTML
    assert 'cleanup: $("cleanup").checked' in HTML
    assert "div.title = event.original" in HTML


def test_chips_offer_mid_meeting_rename():
    assert '"/api/roster/rename"' in HTML
    assert "Speaker name" in HTML  # the rename prompt


def test_recovered_transcripts_are_surfaced_in_the_status_line():
    assert "s.recovered" in HTML
    assert "RECOVERED" in HTML


def test_cleanup_download_state_is_shown_to_the_user():
    assert "downloading:" in HTML
    assert "Downloading cleanup model" in HTML


def test_nearest_guess_speakers_are_flagged_in_the_live_transcript():
    assert 'line.attribution === "nearest"' in HTML
    assert "who guess" in HTML
    assert ".line .who.guess" in HTML


def test_one_control_both_finds_a_saved_person_and_creates_a_new_one():
    assert 'id="new-person-name"' not in HTML
    assert 'id="create-person-btn"' not in HTML
    assert 'id="add-person-btn"' not in HTML  # the field alone commits now
    assert "function renderPersonResults()" in HTML
    assert "function commitChoice(choice)" in HTML
    assert '"/api/library/speakers/create"' in HTML
    # Matches first, "create" as the last resort — and only without an exact hit.
    assert 'if (query && !matchingSpeaker(query)) personResults.push({ type: "create", name: query });' in HTML
    assert '`+ CREATE "${choice.name.toUpperCase()}"`' in HTML


def test_person_results_exclude_people_already_on_the_roster():
    assert "const onRoster = new Set(setupRoster.map((entry) => entry.speaker_id));" in HTML
    assert "!onRoster.has(speaker.id)" in HTML


def test_person_results_are_keyboard_navigable():
    assert 'if (event.key === "ArrowDown" || event.key === "ArrowUp")' in HTML
    assert "paintPersonActive()" in HTML
    assert 'aria-activedescendant' in HTML
    assert 'if (event.key === "Escape") { closePersonResults(); return; }' in HTML


def test_enter_in_the_person_search_adds_a_person_instead_of_starting_the_meeting():
    assert 'if (event.key !== "Enter") return;' in HTML
    assert "const choice = personResults[personActive] || personResults[0];" in HTML
    assert "if (choice) commitChoice(choice);" in HTML


def test_roster_rows_reorder_by_drag_or_by_keyboard():
    # The setup roster's arrows are gone; the grip is the affordance. (Group
    # members in Settings keep their arrows — they aren't drag targets.)
    assert "button.title = `Move ${entry.name} ${word} to hotkey slot ${target + 1}`;" not in HTML
    assert 'grip.className = "grip"' in HTML
    assert "row.draggable = true;" in HTML
    assert "row.ondrop = (event) => {" in HTML
    assert "moveRosterEntry(from, index);" in HTML
    # Row order assigns the hotkey slots, so it must stay reachable without a mouse.
    assert "Press Alt+Up or Alt+Down to move." in HTML
    assert 'const delta = event.key === "ArrowUp" ? -1 : event.key === "ArrowDown" ? 1 : 0;' in HTML
    assert 'rosterFocus = { action: "grip", index: to };' in HTML
    assert 'remove.title = `Remove ${entry.name} from this meeting`;' in HTML


def test_the_grip_is_drawn_in_css_not_a_font_glyph():
    # Same failure mode as the ✎ button: the bundled pixel font lacks the glyph.
    assert ".grip::before" in HTML
    assert "box-shadow: 0 4px #6b6558" in HTML


def test_capture_options_share_one_row():
    assert '<div class="check-row"' in HTML
    assert ".check-row { display: flex;" in HTML
    assert "SYSTEM AUDIO" in HTML
    assert "CLEAN (AI)" in HTML


def test_a_pending_hotkey_choice_survives_background_state_updates():
    # `status` events fire on every library mutation and re-run applyState; without
    # the dirty flag they silently reverted the user's pending selection, so SAVE
    # persisted the old combination.
    assert "let hotkeyDirty = false;" in HTML
    assert "if (!hotkeyDirty) {" in HTML
    assert "input.onchange = () => { hotkeyDirty = true; updateHotkeyPreview(); };" in HTML
    assert "hotkeyDirty = false;\n    applyState(result);" in HTML
    assert "— UNSAVED" in HTML


def test_roster_instructions_are_a_disclosure_rather_than_permanent_copy():
    assert '<details class="help">' in HTML
    assert "HOW THE ROSTER WORKS" in HTML
    assert 'id="hotkey-hint"' in HTML  # still written to by renderSettings


def test_destructive_library_actions_sit_behind_an_overflow_toggle():
    assert 'actions.className = "row-actions"' in HTML
    assert "actions.append(rename, reset, remove);" in HTML
    assert "row.append(record, more, actions);" in HTML
    assert ".library-row.open .row-actions { display: flex; }" in HTML


def test_the_library_gains_a_filter_only_once_the_list_is_long():
    assert 'id="library-search"' in HTML
    assert "LIBRARY_FILTER_THRESHOLD" in HTML
    assert "speaker.name.toLowerCase().includes(needle)" in HTML


def test_groups_collapse_by_default_and_label_their_member_list():
    assert 'document.createElement("details")' in HTML
    assert "card.open = openGroups.has(group.id);" in HTML
    assert "MEMBERS — ORDER SETS HOTKEY SLOTS" in HTML
    assert '${count} MEMBER${count === 1 ? "" : "S"}' in HTML


def test_selected_states_are_marked_with_the_oat_accent():
    assert ".tab.active { background: var(--oat); }" in HTML
    assert "background: var(--oat); box-shadow: 3px 3px 0 var(--ink);" in HTML
    assert '.notes-tab[aria-expanded="true"] { background: var(--oat-dark); }' in HTML


def test_secondary_buttons_stay_unfilled():
    assert 'id="live-add-btn">+ ADD' in HTML
    assert 'class="mini primary" id="library-create"' not in HTML
    assert 'class="mini primary" id="group-create"' not in HTML
    # RECORD stays filled only where a profile is still missing.
    assert 'record.className = trained ? "mini" : "mini primary";' in HTML


def test_chip_controls_use_glyphs_the_bundled_pixel_font_actually_has():
    assert '"✎"' not in HTML
    assert 'rename.textContent = "EDIT";' in HTML
    assert '.chip .label { flex: 1; min-width: 0; }' in HTML


def test_profile_learning_feedback_uses_the_new_three_and_five_second_rules():
    assert "profile_learning: null" in HTML
    assert "LEARN ${Number(learning.speech_seconds || 0).toFixed(1)}" in HTML
    assert "SAMPLE ADDED" in HTML
    assert "SAMPLE SKIPPED" in HTML
    assert "ready_seconds: 5.0" in HTML


def test_dictation_card_offers_both_chords_and_the_vocabulary_editor():
    assert 'id="dictation-form"' in HTML
    assert 'id="dictation-modifier-grid"' in HTML
    assert 'id="dictation-email-grid"' in HTML
    assert 'id="vocabulary-list"' in HTML
    assert "function renderVocabulary()" in HTML
    assert "DICTATION" in HTML


def test_dictation_card_edits_are_not_clobbered_by_status_events():
    assert "if (!dictationDirty) {" in HTML
    assert "dictationDirty = true;" in HTML
    # Cleared before applyState so the card re-syncs from the saved values.
    assert HTML.index("dictationDirty = false;") < HTML.index("applyState(result);\n    const label")


def test_dictation_save_sends_every_field_it_owns():
    for field in (
        "dictation_enabled", "dictation_modifiers", "dictation_email_modifiers",
        "dictation_activation", "dictation_injection", "dictation_email_detection",
        "dictation_restore_clipboard", "dictation_spoken_punctuation",
        "dictation_signature", "dictation_vocabulary", "overlay_enabled",
    ):
        assert f"{field}:" in HTML


def test_dictation_status_mirrors_the_overlay():
    assert 'id="dictation-status"' in HTML
    assert 'event.type === "dictation"' in HTML
    assert "function renderDictationStatus()" in HTML
    assert "DICTATION_LABELS" in HTML


def test_empty_email_chord_reads_as_off():
    assert 'chordLabel(gridModifiers("dictation-email-grid"), "OFF")' in HTML
