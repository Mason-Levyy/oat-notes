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
    # `status` events fire on every library mutation and re-run applyState. The
    # open editor keeps its own draft, so a repaint no longer has anything to
    # revert — the draft is only seeded when EDIT is pressed.
    assert "let editingShortcut = null;" in HTML
    assert "let shortcutDraft = [];" in HTML
    assert "shortcutDraft = shortcutChord(shortcut);" in HTML
    assert "editingShortcut = null;\n    shortcutDraft = [];\n    applyState(result);" in HTML


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
    assert ".nav-btn.active { background: var(--oat); }" in HTML
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
    assert ".chip .label {\n  flex: 1; min-width: 0;" in HTML


def test_profile_learning_feedback_uses_the_new_three_and_five_second_rules():
    assert "profile_learning: null" in HTML
    assert "LEARN ${Number(learning.speech_seconds || 0).toFixed(1)}" in HTML
    assert "SAMPLE ADDED" in HTML
    assert "SAMPLE SKIPPED" in HTML
    assert "ready_seconds: 5.0" in HTML


def test_every_shortcut_lives_in_one_card():
    assert 'id="shortcut-list"' in HTML
    assert "function renderShortcuts()" in HTML
    for field in (
        "dictation_modifiers", "dictation_email_modifiers", "hotkey_modifiers"
    ):
        assert f'field: "{field}"' in HTML
    # No modifier grid is rendered until EDIT opens one.
    assert 'class="modifier-grid" id=' not in HTML


def test_shortcut_rows_collapse_to_a_single_line():
    assert 'class="shortcut-chord' in HTML
    assert 'data-edit="${shortcut.id}"' in HTML
    assert 'data-save="${shortcut.id}"' in HTML
    assert 'data-cancel="${shortcut.id}"' in HTML


def test_the_speaker_shortcut_is_locked_while_recording():
    assert "lockedWhileRecording: true" in HTML
    assert "shortcut.lockedWhileRecording && state.recording" in HTML


def test_dictation_card_keeps_behaviour_and_vocabulary():
    assert 'id="dictation-form"' in HTML
    assert 'id="vocabulary-list"' in HTML
    assert "function renderVocabulary()" in HTML
    assert "DICTATION" in HTML


def test_dictation_card_edits_are_not_clobbered_by_status_events():
    assert "if (!dictationDirty) {" in HTML
    assert "dictationDirty = true;" in HTML
    # Cleared before applyState so the card re-syncs from the saved values.
    submit = HTML[HTML.index('$("dictation-form").onsubmit'):]
    assert submit.index("dictationDirty = false;") < submit.index("applyState(result);")


def test_dictation_save_sends_every_field_it_owns():
    for field in (
        "dictation_email_detection", "dictation_restore_clipboard",
        "dictation_spoken_punctuation",
        "dictation_vocabulary", "overlay_enabled",
    ):
        assert f"{field}:" in HTML


def test_the_enable_switch_lives_in_the_card_header():
    head = HTML[HTML.index('<div class="card-head">'):]
    head = head[:head.index("</div>")]
    assert 'id="dictation-enabled"' in head


def test_the_enable_switch_applies_without_a_save():
    # It is the one dictation setting with an immediate effect, so it posts on
    # change; the rest of the card still waits for SAVE.
    handler = HTML[HTML.index('$("dictation-enabled").onchange'):]
    handler = handler[:handler.index("\n};")]
    assert '"/api/dictation/toggle"' in handler
    submit = HTML[HTML.index('$("dictation-form").onsubmit'):]
    assert "dictation_enabled:" not in submit


def test_turning_dictation_off_greys_out_the_rest_of_the_card():
    assert 'id="dictation-body"' in HTML
    assert '.card-body.off {' in HTML
    assert 'body.classList.toggle("off", !enabled);' in HTML
    assert "body.inert = !enabled;" in HTML


def test_the_dictation_card_no_longer_saves_chords():
    # The SHORTCUTS card owns them; sending them from here too would let a
    # stale copy overwrite a chord saved seconds earlier.
    submit = HTML[HTML.index('$("dictation-form").onsubmit'):]
    submit = submit[:submit.index("};")]
    assert "dictation_modifiers" not in submit
    assert "hotkey_modifiers" not in submit


def test_dictation_status_mirrors_the_overlay():
    assert 'id="dictation-status"' in HTML
    assert 'event.type === "dictation"' in HTML
    assert "function renderDictationStatus()" in HTML
    assert "DICTATION_LABELS" in HTML


def test_empty_email_chord_reads_as_off():
    assert 'allowEmpty: true' in HTML
    assert 'shortcut.allowEmpty ? "OFF" : "CHOOSE A MODIFIER"' in HTML


def test_navigation_is_a_vertical_rail_not_tabs():
    assert '<nav class="nav-rail" role="tablist" aria-orientation="vertical"' in HTML
    assert 'class="tabs"' not in HTML
    for section in ("nav-meeting", "nav-dictation", "nav-settings", "nav-info"):
        assert f'id="{section}"' in HTML
    assert "function openSection(name)" in HTML
    assert "function openTab(" not in HTML


def test_every_rail_button_maps_to_a_section():
    for view in ("meeting-view", "dictation-view", "settings-view", "info-view"):
        assert f'id="{view}"' in HTML


def test_rail_glyphs_are_drawn_in_css_not_a_font():
    # Same reason the grip is CSS: the bundled pixel font has no icon coverage.
    assert ".nav-glyph.info" in HTML
    assert ".nav-glyph.dictation" in HTML
    assert "clip-path: polygon" in HTML


def test_history_is_listed_with_copy_and_clear():
    assert 'id="history-list"' in HTML
    assert 'id="history-clear"' in HTML
    assert "function renderHistory()" in HTML
    assert '"/api/dictation/copy"' in HTML
    assert '"/api/dictation/history/clear"' in HTML


def test_history_offers_copy_rather_than_reinsert():
    # A click comes from the browser, so the browser holds focus and a
    # re-insert would land there. Ctrl+Alt+Z is the re-insert path.
    assert ">COPY<" in HTML
    assert '"/api/dictation/insert"' not in HTML


def test_transcript_text_never_rides_the_event_stream():
    assert 'if (event.phase === "inserted") refreshHistory();' in HTML


def test_history_state_does_not_shadow_the_window_global():
    assert "let dictationHistory = [];" in HTML
    assert "let history = [];" not in HTML


def test_info_summarises_status_and_shortcuts():
    assert "function renderInfo()" in HTML
    assert 'id="info-status"' in HTML
    assert 'id="info-shortcuts"' in HTML
    assert "dictation_replay_label" in HTML


def test_transient_dictation_phases_reset_in_the_browser():
    # The overlay dismisses INSERTED itself; the browser needs telling, or the
    # pill reads INSERTED until the next dictation.
    assert "DICTATION_TRANSIENT" in HTML
    assert 'phase: "idle"' in HTML
    assert "clearTimeout(dictationReset)" in HTML


def test_meeting_is_the_landing_section():
    assert '<button class="nav-btn active" id="nav-meeting"' in HTML
    assert 'id="nav-home"' not in HTML
    assert HTML.index('id="nav-meeting"') < HTML.index('id="nav-dictation"')


def test_info_is_pinned_to_the_bottom_of_the_rail():
    assert HTML.index('id="nav-info"') > HTML.index('id="nav-settings"')
    assert ".nav-btn.info {" in HTML
    assert "margin-top: auto;" in HTML


def test_roster_controls_stay_inside_the_rail():
    # The rail is a fixed 300px and .stage is a later sibling, so a chip row
    # that cannot shrink pushes EDIT under the transcript's background, where
    # only part of it is clickable.
    assert ".chip-row .chip { flex: 1; width: auto; min-width: 0; }" in HTML
    assert ".chip-row { display: flex; flex-wrap: wrap;" in HTML
    assert "overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" in HTML
    assert "position: relative; z-index: 1;" in HTML


def test_enter_files_a_note_and_shift_enter_writes_a_newline():
    assert '$("notes-input").addEventListener("keydown"' in HTML
    assert 'if (event.key !== "Enter" || event.shiftKey) return;' in HTML
    assert '$("notes-form").requestSubmit();' in HTML


def test_the_notes_tab_does_not_cover_the_transcript():
    assert "@media (max-width: 1100px) {" in HTML
    assert ".stage { padding-right: 42px; }" in HTML
