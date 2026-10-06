"""The channel picker's decisions, exercised through node, plus its call sites.

Same split as test_follow_scroll.py: everything the picker decides — the count
it asks for, which videos are selected, which URLs land in the textarea, which
panel shows — is a pure function in static/picker.js and genuinely executed
here. What those tests cannot reach is the DOM wiring that calls them, so the
second half pins each call site with assertions scoped to its enclosing
function. Neither half says anything about how the picker LOOKS: layout,
thumbnails rendering, focus rings and dark-mode contrast need a browser, and
there is none in this environment.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from .test_follow_scroll import _code_only, js_function_body

STATIC = Path(__file__).resolve().parents[1] / "src/panekmodel2/server/static"
PICKER_JS = STATIC / "picker.js"
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"
STYLES = STATIC / "styles.css"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def call(fn: str, *args):
    script = (
        f"const m = require({json.dumps(str(PICKER_JS))});"
        f"const out = m.{fn}(...{json.dumps(list(args))});"
        "process.stdout.write(JSON.stringify(out === undefined ? null : out));"
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(result.stdout)


def video(n: int) -> dict:
    vid = f"v{n:010d}"
    return {"id": vid, "title": f"Video {n}", "duration_seconds": 60 * n,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "thumbnail_url": f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"}


VIDEOS = [video(i) for i in range(1, 6)]
IDS = [v["id"] for v in VIDEOS]
URLS = [v["url"] for v in VIDEOS]


# ── count ───────────────────────────────────────────────────────────
@needs_node
@pytest.mark.parametrize("raw,expected", [
    (25, 25), ("40", 40), (1, 1), (100, 100), (0, 1), (-3, 1), (101, 100), ("250", 100),
    ("", 25), (None, 25), ("abc", 25), (12.6, 13),
])
def test_clamp_count(raw, expected):
    assert call("clampCount", raw) == expected


@needs_node
def test_limits_match_the_server():
    from panekmodel2.channel_resolver import DEFAULT_COUNT, MAX_COUNT

    assert call("clampCount", "") == DEFAULT_COUNT
    assert call("clampCount", 10**6) == MAX_COUNT


# ── selection state ─────────────────────────────────────────────────
@needs_node
def test_checking_adds_and_unchecking_removes():
    sel = call("setSelected", [], IDS[1], True)
    assert sel == [IDS[1]]
    sel = call("setSelected", sel, IDS[3], True)
    assert sorted(sel) == sorted([IDS[1], IDS[3]])
    assert call("setSelected", sel, IDS[1], False) == [IDS[3]]


@needs_node
def test_checking_twice_does_not_duplicate():
    assert call("setSelected", [IDS[0]], IDS[0], True) == [IDS[0]]


@needs_node
def test_unchecking_an_unselected_video_is_harmless():
    assert call("setSelected", [IDS[0]], IDS[4], False) == [IDS[0]]


@needs_node
def test_set_selected_does_not_mutate_its_input():
    script = (
        f"const m = require({json.dumps(str(PICKER_JS))});"
        "const before = ['a']; m.setSelected(before, 'b', true); m.setSelected(before, 'a', false);"
        "process.stdout.write(JSON.stringify(before));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == ["a"]


@needs_node
def test_select_all_and_none():
    assert call("selectAll", VIDEOS) == IDS
    assert call("selectNone") == []


# ── what reaches the textarea ───────────────────────────────────────
@needs_node
def test_selected_urls_follow_channel_order_not_click_order():
    assert call("selectedUrls", VIDEOS, [IDS[3], IDS[0]]) == [URLS[0], URLS[3]]


@needs_node
def test_stale_selections_are_ignored():
    """IDs left over from a previous channel are not in this list, so not added."""
    assert call("selectedUrls", VIDEOS, ["vSTALE00000", IDS[2]]) == [URLS[2]]


@needs_node
def test_nothing_selected_means_nothing_added():
    assert call("selectedUrls", VIDEOS, []) == []


@needs_node
def test_append_to_an_empty_corpus():
    out = call("appendToCorpus", "", [], URLS[:2])
    assert out == {"text": "\n".join(URLS[:2]), "added": 2, "already": 0}


@needs_node
def test_append_keeps_what_the_researcher_already_typed():
    existing = "https://youtu.be/aaaaaaaaaaa\n  https://youtu.be/bbbbbbbbbbb  "
    out = call("appendToCorpus", existing, ["https://youtu.be/aaaaaaaaaaa", "https://youtu.be/bbbbbbbbbbb"], [URLS[0]])
    assert out["text"] == existing + "\n" + URLS[0]
    assert out["text"].startswith(existing), "existing text is never rewritten"


@needs_node
def test_append_after_a_trailing_newline_does_not_leave_a_blank_line():
    out = call("appendToCorpus", "https://youtu.be/aaaaaaaaaaa\n", ["https://youtu.be/aaaaaaaaaaa"], [URLS[0]])
    assert out["text"] == "https://youtu.be/aaaaaaaaaaa\n" + URLS[0]


@needs_node
def test_append_skips_urls_already_in_the_corpus():
    out = call("appendToCorpus", URLS[0], [URLS[0]], [URLS[0], URLS[1]])
    assert out == {"text": URLS[0] + "\n" + URLS[1], "added": 1, "already": 1}


@needs_node
def test_append_of_nothing_new_leaves_the_text_untouched():
    out = call("appendToCorpus", URLS[0], [URLS[0]], [URLS[0]])
    assert out == {"text": URLS[0], "added": 0, "already": 1}


@needs_node
def test_append_dedupes_within_the_batch_itself():
    out = call("appendToCorpus", "", [], [URLS[0], URLS[0]])
    assert out["text"] == URLS[0] and out["added"] == 1


# ── which panel shows ───────────────────────────────────────────────
@needs_node
@pytest.mark.parametrize("state,panel", [
    ({"status": "idle"}, "idle"),
    ({"status": "loading"}, "loading"),
    ({"status": "ready", "videos": [video(1)]}, "list"),
    ({"status": "ready", "videos": []}, "empty"),
    ({"status": "error", "kind": "private_or_empty"}, "empty"),
    ({"status": "error", "kind": "not_found"}, "error"),
    ({"status": "error", "kind": "unreachable"}, "error"),
    ({"status": "error", "kind": ""}, "error"),
    ({}, "idle"),
])
def test_panel_state(state, panel):
    assert call("pickerPanelState", state) == panel


@needs_node
def test_panel_state_is_safe_on_missing_state():
    assert call("pickerPanelState", None) == "idle"


@needs_node
@pytest.mark.parametrize("event,broken", [
    ({"type": "error"}, True),
    ({"type": "load", "naturalWidth": 120}, True),   # i.ytimg's 404 placeholder, measured
    ({"type": "load", "naturalWidth": 90}, True),
    ({"type": "load", "naturalWidth": 320}, False),  # a real mqdefault, measured
    ({"type": "load", "naturalWidth": 121}, False),
    ({"type": "load", "naturalWidth": 0}, False),    # not decoded yet: no verdict
    ({"type": "load"}, False),
    (None, False),
])
def test_thumbnail_fallback_decision(event, broken):
    assert call("thumbIsBroken", event) is broken


@needs_node
@pytest.mark.parametrize("n,label", [
    (0, "Select videos to add"), (1, "Add 1 selected to run"), (7, "Add 7 selected to run"),
])
def test_add_button_label(n, label):
    assert call("addLabel", n) == label


@needs_node
@pytest.mark.parametrize("added,already,note", [
    (3, 0, "Added 3 videos to the list above."),
    (1, 0, "Added 1 video to the list above."),
    (2, 1, "Added 2 videos to the list above · 1 was already there."),
    (0, 4, "All 4 were already in the list — nothing added."),
])
def test_added_note(added, already, note):
    assert call("addedNote", added, already) == note


# ── call sites: what the decisions cannot reach ─────────────────────
def app():
    return _code_only(APP_JS)


def test_picker_js_loads_before_app_js():
    html = INDEX_HTML.read_text()
    assert html.index('src="./picker.js"') < html.index('src="./app.js"')


def test_picker_js_never_touches_the_dom():
    code = _code_only(PICKER_JS)
    assert "document." not in code and "window." not in code


def test_fetch_clamps_the_count_and_calls_the_endpoint():
    body = js_function_body(app(), "fetchChannel")
    assert "clampCount(" in body
    assert "/api/channel/videos?" in body
    assert "new URLSearchParams(" in body


def test_fetch_results_render_without_stealing_focus():
    """A listing that lands while the researcher types in the URL textarea must
    put them back where they were. Source-level only: the behaviour itself was
    checked once in headless Chrome, which this suite cannot repeat."""
    assert "renderKeepingFocus('#pick-status')" in js_function_body(app(), "fetchChannel")
    body = js_function_body(app(), "renderKeepingFocus")
    assert "document.getElementById(id)" in body
    assert "back.focus()" in body
    assert body.index("render()") < body.index("back.focus()")


def test_fetch_ignores_a_superseded_response():
    """Two quick fetches: the slower first one must not overwrite the second."""
    body = js_function_body(app(), "fetchChannel")
    assert body.count("seq !== channelSeq") == 2


def test_fetch_records_the_error_kind_for_the_panel():
    body = js_function_body(app(), "fetchChannel")
    assert "err.kind" in body


def test_api_passes_kind_through():
    body = js_function_body(app(), "api")
    assert "err.kind" in body and "body.kind" in body


def test_the_screen_renders_the_picker_and_the_panel_is_chosen_by_the_decision():
    assert "channelCard()" in js_function_body(app(), "screenRun")
    assert "pickerPanelState(" in js_function_body(app(), "channelPanel")


def test_add_goes_through_selected_urls_and_the_existing_parser():
    body = js_function_body(app(), "addSelectedToRun")
    assert "selectedUrls(" in body
    assert "appendToCorpus(S.urlText, parseUrls(S.urlText)" in body
    assert "S.urlText = " in body


def test_checkbox_changes_update_selection_without_a_rerender():
    """A full render() replaces the screen and drops keyboard focus mid-list."""
    body = js_function_body(app(), "onPickToggle")
    assert "setSelected(" in body
    assert "syncPickerControls()" in body
    assert "render()" not in body


def test_select_all_and_none_keep_focus_too():
    code = app()
    for action, fn in (("pick-all", "selectAll("), ("pick-none", "selectNone(")):
        branch = code[code.index(f"case '{action}'"):]
        branch = branch[: branch.index("case ", 5)]
        assert fn in branch
        assert "syncPickerControls()" in branch
        assert "render()" not in branch


def test_sync_applies_selection_to_checkboxes_and_the_add_button():
    body = js_function_body(app(), "syncPickerControls")
    assert ".checked = " in body
    assert "addLabel(" in body
    assert ".disabled = " in body


def test_the_channel_form_submits_on_enter_without_reloading():
    code = app()
    handler = code[code.index("document.addEventListener('submit'"):]
    handler = handler[: handler.index("});") + 3]
    assert "preventDefault()" in handler
    assert "fetchChannel()" in handler


def test_broken_thumbnails_fall_back():
    code = app()
    # error and load do not bubble, so only a capture-phase listener sees them.
    assert "document.addEventListener('error', onThumbEvent, true)" in code
    assert "document.addEventListener('load', onThumbEvent, true)" in code
    body = js_function_body(code, "onThumbEvent")
    assert "thumbIsBroken(" in body and "naturalWidth" in body
    assert "pick-thumb" in body and "classList.add('broken')" in body


def test_cards_use_a_real_checkbox_and_a_new_tab_link():
    body = js_function_body(app(), "pickCard")
    assert 'type="checkbox"' in body
    assert 'target="_blank"' in body and 'rel="noopener noreferrer"' in body
    assert 'referrerpolicy="no-referrer"' in body
    # Every interpolated payload field is escaped.
    for field in ("v.id", "v.title", "v.url", "v.thumbnail_url"):
        assert f"esc({field})" in body, field


def test_cards_never_carry_attributes_the_global_click_router_acts_on():
    """data-video on a card would navigate to the video page on every click."""
    body = js_function_body(app(), "pickCard")
    for attr in ("data-video", "data-go", "data-topic", "data-field", "data-seek"):
        assert attr not in body


def test_styles_use_tokens_not_literal_colours():
    css = STYLES.read_text()
    block = css[css.index("/* ── channel picker"):css.index("/* ── progress")]
    import re

    assert not re.search(r"#[0-9A-Fa-f]{3,8}\b", block), "picker styles must use the design tokens"
    assert "var(--" in block
