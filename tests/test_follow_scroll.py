"""Transcript auto-follow scroll maths, exercised through node.

The human hit a real defect: playback scrolled the whole browser page to the
active transcript line, and fought them for the viewport every tick when they
scrolled back to the video. The cause was ``row.scrollIntoView()``, which
scrolls every scrollable ancestor including the document.

There is no JS test harness in this project, but node is available, so the
arithmetic that replaced it is a pure function and genuinely covered here
rather than merely asserted in prose. The DOM wiring around it — the scroll
listener, the pause flag, the chip — is still only verified by hand.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

FOLLOW_JS = Path(__file__).resolve().parents[1] / "src/panekmodel2/server/static/follow.js"

# N-2: this used to be a module-level pytestmark, which also skipped the static
# source guards below — they need no node and must run everywhere.
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

APP_JS = Path(__file__).resolve().parents[1] / "src/panekmodel2/server/static/app.js"


def follow_scroll_top(**view):
    """Call followScrollTop() in node and return its result."""
    script = (
        f"const {{followScrollTop}} = require({json.dumps(str(FOLLOW_JS))});"
        f"const out = followScrollTop({json.dumps(view)});"
        "process.stdout.write(JSON.stringify(out === null ? null : out));"
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(result.stdout)


BAND = {"scrollTop": 100, "viewHeight": 400, "contentHeight": 2000, "rowHeight": 40}


@needs_node
def test_row_already_visible_does_not_move_the_reader():
    """The whole point: no scrolling unless the row actually left the view."""
    assert follow_scroll_top(rowTop=200, **BAND) is None


@needs_node
def test_row_above_the_view_scrolls_up_to_it():
    target = follow_scroll_top(rowTop=20, **BAND)
    assert target is not None and target < BAND["scrollTop"]


@needs_node
def test_row_below_the_view_scrolls_down_to_it():
    target = follow_scroll_top(rowTop=900, **BAND)
    assert target is not None and target > BAND["scrollTop"]
    # Its bottom lands inside the visible band, not past it.
    assert target <= 900


@needs_node
def test_target_never_goes_negative():
    assert follow_scroll_top(rowTop=0, scrollTop=50, viewHeight=400,
                             contentHeight=2000, rowHeight=40, margin=12) == 0


@needs_node
def test_target_is_clamped_to_the_scrollable_range():
    target = follow_scroll_top(rowTop=1980, scrollTop=0, viewHeight=400,
                               contentHeight=2000, rowHeight=40)
    assert target == 2000 - 400


@needs_node
def test_a_row_taller_than_the_container_shows_its_start():
    """Otherwise a long chunk would scroll past its own first line."""
    target = follow_scroll_top(rowTop=1000, rowHeight=900, scrollTop=0,
                               viewHeight=400, contentHeight=3000, margin=12)
    assert target == 1000 - 12


@needs_node
def test_zero_height_container_is_a_no_op():
    """Before layout, or on a hidden panel, there is nothing to scroll."""
    assert follow_scroll_top(rowTop=500, rowHeight=40, scrollTop=0,
                             viewHeight=0, contentHeight=0) is None


@needs_node
def test_a_no_op_move_reports_none():
    """Returning the current position would reset the programmatic flag for
    nothing, and could swallow the next genuine user scroll."""
    assert follow_scroll_top(rowTop=0, rowHeight=40, scrollTop=0,
                             viewHeight=400, contentHeight=2000, margin=0) is None


@needs_node
@pytest.mark.parametrize("row_top", [0, 37, 400, 1234, 1960])
def test_result_is_always_a_valid_scroll_position(row_top):
    target = follow_scroll_top(rowTop=row_top, **BAND)
    if target is not None:
        assert 0 <= target <= BAND["contentHeight"] - BAND["viewHeight"]


@needs_node
def test_following_converges_in_one_step():
    """After following, the row must be inside the band — no oscillation."""
    view = dict(rowTop=1500, **BAND)
    target = follow_scroll_top(**view)
    assert target is not None

    settled = dict(view, scrollTop=target)
    assert follow_scroll_top(**settled) is None, "a second tick would move again"


def _code_only(path: Path) -> str:
    """Source with comments stripped.

    Both files legitimately *mention* scrollIntoView in prose explaining why it
    is not used, so a bare substring check would fail on its own documentation.
    """
    source = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
    return re.sub(r"^\s*//.*$", "", source, flags=re.M)


def js_function_body(source: str, name: str) -> str:
    """The body of a named JS function, by brace matching.

    File-wide substring checks are why N-4 slipped through: every one of the
    four wiring regressions deletes a call from a *particular* function, and a
    string that still appears somewhere else in the file keeps the test green.
    Scoping each assertion to its enclosing function is what makes these bite.

    N-5 hardening, and it fails loudly rather than mis-scoping:

    * Braces inside string and template literals are skipped, so a literal
      "}" cannot truncate a body early. Quote tracking starts at the opening
      brace, not at the top of the file — an earlier stray quote elsewhere in
      the file then cannot cascade and swallow everything after it, which is
      exactly what a whole-file mask did when I tried that first.
    * A name defined more than once raises, rather than the first definition
      silently winning.

    Known limit, recorded rather than guessed: a regex literal containing an
    unbalanced brace would still confuse the scan. There are none in this
    codebase, and the failure would be a loud "unbalanced braces", not a
    quietly wrong body.
    """
    # One pattern only: "async function f(" contains "function f(", so
    # searching both forms counted a single async definition twice.
    hits = list(_find_all(source, f"function {name}("))
    if not hits:
        raise AssertionError(f"{name}() not found — was it renamed?")
    if len(hits) > 1:
        raise AssertionError(
            f"{name}() is defined {len(hits)} times; scoping to 'the' body would "
            "silently pick the first. Disambiguate before asserting on it."
        )

    brace = source.index("{", hits[0])
    depth = 0
    quote = None
    i = brace
    while i < len(source):
        ch = source[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[brace : i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces in {name}()")


def _find_all(haystack: str, needle: str):
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            return
        yield i
        start = i + 1


# ── the decisions, exercised through node ───────────────────────────
@needs_node
def call(fn: str, arg) -> object:
    script = (
        f"const m = require({json.dumps(str(FOLLOW_JS))});"
        f"process.stdout.write(JSON.stringify(m.{fn}({json.dumps(arg)})));"
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(result.stdout)


@needs_node
@pytest.mark.parametrize("playing,enabled,expected", [
    (True, True, True),
    (True, False, False),    # reader took over: no scrolling
    (False, True, False),    # paused video: nothing to follow
    (False, False, False),
])
def test_should_follow_requires_playing_and_enabled(playing, enabled, expected):
    assert call("shouldFollow", {"playing": playing, "followEnabled": enabled}) is expected


@needs_node
def test_should_follow_is_safe_on_missing_state():
    assert call("shouldFollow", None) is False


@needs_node
def test_our_own_scroll_is_not_the_reader_taking_over():
    """Mutation 3's logic: without this, auto-follow kills itself on tick one."""
    assert call("isReaderScroll", {"programmatic": True}) is False
    assert call("isReaderScroll", {"programmatic": False}) is True


@needs_node
def test_an_unflagged_scroll_counts_as_the_reader():
    """Fail safe: an unknown scroll hands control to the human, not away."""
    assert call("isReaderScroll", {}) is True
    assert call("isReaderScroll", None) is True


@needs_node
@pytest.mark.parametrize("enabled,hidden", [(True, True), (False, False)])
def test_chip_is_visible_exactly_when_follow_is_off(enabled, hidden):
    """The chip is the only way back, so it must appear whenever follow stops."""
    assert call("followChipHidden", {"followEnabled": enabled}) is hidden


# ── call sites: what the decisions cannot reach ─────────────────────
def test_playback_tick_asks_should_follow():
    body = js_function_body(_code_only(APP_JS), "updatePlayhead")
    assert "shouldFollow(" in body
    assert "followRow(" in body


def test_highlighting_is_not_gated_on_follow_state():
    """Property 3: pausing auto-follow pauses scrolling only."""
    body = js_function_body(_code_only(APP_JS), "updatePlayhead")
    assert body.index("classList.toggle('active'") < body.index("shouldFollow(")


def test_follow_row_flags_its_own_scroll():
    """Mutation 3: drop this and auto-follow disables itself on the first tick."""
    body = js_function_body(_code_only(APP_JS), "followRow")
    assert "programmaticScroll = true" in body
    assert "container.scrollTop = target" in body
    assert "followScrollTop(" in body


def test_the_scroll_listener_is_bound_when_the_video_screen_renders():
    """Mutation 1: without this call the listener never attaches at all."""
    body = js_function_body(_code_only(APP_JS), "render")
    assert "bindTranscriptFollow()" in body


def test_the_scroll_listener_consults_the_decision():
    """Mutation 3 again, from the listener's side."""
    body = js_function_body(_code_only(APP_JS), "bindTranscriptFollow")
    assert "addEventListener('scroll'" in body
    assert "isReaderScroll(" in body
    assert "setFollow(false)" in body


def test_setfollow_applies_the_chip_decision():
    """Mutation 4: without this the chip never appears and there is no way back."""
    body = js_function_body(_code_only(APP_JS), "setFollow")
    assert "followChipHidden(" in body
    assert "chip.hidden" in body


def test_navigating_to_a_moment_hands_scrolling_back():
    """Mutation 2: clicking a transcript row must re-enable following."""
    code = _code_only(APP_JS)
    handler = code[code.index("document.addEventListener('click'"):]
    branch = handler[handler.index("if (el.dataset.video)"):]
    branch = branch[: branch.index("if (el.dataset.topic)")]
    assert "setFollow(true)" in branch


# ── the old call must stay gone ─────────────────────────────────────
def test_the_page_scroll_is_never_part_of_the_calculation():
    """follow.js cannot touch the document even in principle."""
    code = _code_only(FOLLOW_JS)
    assert ".scrollIntoView(" not in code
    assert "window." not in code and "document." not in code


def test_app_no_longer_calls_scrollintoview():
    assert ".scrollIntoView(" not in _code_only(APP_JS), (
        "scrollIntoView scrolls every scrollable ancestor, including the page"
    )


def test_function_body_helper_actually_scopes():
    """The helper is load-bearing for every assertion above, so prove it works."""
    source = "function a() { alpha(); }\nfunction b() { beta(); }"
    assert "alpha()" in js_function_body(source, "a")
    assert "beta()" not in js_function_body(source, "a")
    with pytest.raises(AssertionError):
        js_function_body(source, "missing")


def test_helper_handles_nested_braces():
    """N-5: premature truncation at the first inner close-brace.

    Without depth counting this returns "{ if (x) {" and every assertion
    scoped to such a function silently checks almost nothing.
    """
    source = "function a() { if (x) { alpha(); } }\nfunction b() { beta(); }"
    body = js_function_body(source, "a")
    assert "alpha()" in body
    assert "beta()" not in body
    assert body.count("{") == body.count("}") == 2


@pytest.mark.parametrize("literal", [
    "'}'",                 # a close brace in a single-quoted string
    '"}"',                 # …double-quoted
    "`}`",                 # …a template literal
    "'{'",                 # an *opening* brace, which would unbalance upward
    "'\\'}'",              # an escaped quote before a brace
])
def test_braces_inside_literals_do_not_truncate(literal):
    """N-5: a quoted brace must not be counted by the matcher."""
    source = f"function a() {{ pick({literal}); alpha(); }}\nfunction b() {{ beta(); }}"
    body = js_function_body(source, "a")
    assert "alpha()" in body
    assert "beta()" not in body


def test_duplicate_definitions_raise_rather_than_picking_the_first():
    """N-5: silently taking the first would make an assertion meaningless."""
    source = "function a() { alpha(); }\nfunction a() { second(); }"
    with pytest.raises(AssertionError, match="defined 2 times"):
        js_function_body(source, "a")


def test_unbalanced_braces_fail_loudly():
    with pytest.raises(AssertionError, match="unbalanced"):
        js_function_body("function a() { alpha();", "a")


def test_async_functions_are_found():
    source = "async function a() { alpha(); }"
    assert "alpha()" in js_function_body(source, "a")
