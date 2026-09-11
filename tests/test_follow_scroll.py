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


# Characters after which a "/" begins a regex literal rather than a division.
# The standard heuristic: a regex can only start where a value is expected.
_REGEX_PRECEDERS = set("(,=:[!&|?{};\n") | {"return", "typeof", "case", "in", "of"}


def _skip_regex(source: str, i: int) -> int:
    """Index just past the regex literal starting at *i*, or raise.

    N-5 remnant: a regex may contain unbalanced braces — /\\d{2,}/ has one —
    so the brace scan has to step over it. Whether a "/" starts a regex or is
    a division is genuinely ambiguous without a full parser, so this uses the
    usual preceding-token heuristic and raises rather than guessing when the
    literal does not terminate on its line.
    """
    j = i + 1
    in_class = False
    while j < len(source):
        ch = source[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "\n":
            raise AssertionError(
                "unterminated regex literal while scanning; the brace matcher "
                "cannot be trusted here — simplify the function or assert "
                "against the whole file instead."
            )
        if in_class:
            if ch == "]":
                in_class = False
        elif ch == "[":
            in_class = True
        elif ch == "/":
            return j + 1
        j += 1
    raise AssertionError("unterminated regex literal at end of source")


def _looks_like_regex_start(source: str, i: int) -> bool:
    """True when the "/" at *i* opens a regex rather than dividing."""
    if source.startswith("//", i) or source.startswith("/*", i):
        return False  # a comment; callers strip these, but be safe
    k = i - 1
    while k >= 0 and source[k] in " \t":
        k -= 1
    if k < 0:
        return True
    if source[k] in _REGEX_PRECEDERS:
        return True
    word = ""
    while k >= 0 and (source[k].isalnum() or source[k] == "_"):
        word = source[k] + word
        k -= 1
    return word in _REGEX_PRECEDERS


def _scan_to_matching_brace(source: str, brace: int) -> int:
    """Index of the "}" that closes the "{" at *brace*, skipping literals."""
    depth = 0
    i = brace
    while i < len(source):
        ch = source[i]
        if ch in "\"'`":
            quote, i = ch, i + 1
            while i < len(source):
                if source[i] == "\\":
                    i += 2
                    continue
                if source[i] == quote:
                    break
                i += 1
        elif ch == "/" and _looks_like_regex_start(source, i):
            i = _skip_regex(source, i) - 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise AssertionError("unbalanced braces")


def js_function_body(source: str, name: str) -> str:
    """The body of a named JS function, by brace matching.

    File-wide substring checks are why N-4 slipped through: every one of those
    wiring regressions deletes a call from a *particular* function, and a
    string that still appears somewhere else in the file keeps the test green.
    Scoping each assertion to its enclosing function is what makes these bite.

    It fails loudly rather than mis-scoping, which matters most for *absence*
    assertions — a truncated body makes "X is not in this function" pass for
    free (N-6). Guards, each with its own test:

    * duplicate definitions raise instead of the first silently winning;
    * braces inside string and template literals are skipped;
    * braces inside regex literals are skipped, and an unterminated regex
      raises rather than being guessed at (N-5 remnant);
    * the returned body is re-verified independently: it must start with "{",
      end with "}", and balance to zero exactly once, at its final character.
      A body truncated at an inner brace fails that check.

    Quote tracking starts at the function's opening brace, not the top of the
    file — a whole-file mask was the first attempt and one stray quote
    cascaded, blanking most of the file including the function being sought.
    """
    hits = list(_find_all(source, f"function {name}("))
    if not hits:
        raise AssertionError(f"{name}() not found — was it renamed?")
    if len(hits) > 1:
        raise AssertionError(
            f"{name}() is defined {len(hits)} times; scoping to 'the' body would "
            "silently pick the first. Disambiguate before asserting on it."
        )

    brace = source.index("{", hits[0])
    close = _scan_to_matching_brace(source, brace)
    body = source[brace : close + 1]
    _assert_whole_body(body, name)
    return body


def _assert_whole_body(body: str, name: str) -> None:
    """Independently re-derive the balance, so truncation cannot pass silently.

    N-6: the reviewer showed a truncating helper shrinking a body from 670 to
    433 characters while a canary survived, leaving an absence assertion
    vacuous. A truncated body is unbalanced, so a balance check catches it
    without needing to know the true length.

    This deliberately does NOT reuse _scan_to_matching_brace. A cross-check
    that shares the mechanism it is checking agrees with that mechanism's
    mistakes: when both used the shared scanner, a deliberately truncating
    scanner produced a truncated body and then pronounced it balanced. The
    count below is simple and independent on purpose.
    """
    if not (body.startswith("{") and body.endswith("}")):
        raise AssertionError(f"{name}() body is not brace-delimited: {body[:40]!r}…")

    depth = 0
    i = 0
    while i < len(body):
        ch = body[i]
        if ch in "\"'`":
            # Shares the literal-skipping primitives, not the matching
            # decision — that separation is the whole point.
            quote, i = ch, i + 1
            while i < len(body):
                if body[i] == "\\":
                    i += 2
                    continue
                if body[i] == quote:
                    break
                i += 1
        elif ch == "/" and _looks_like_regex_start(body, i):
            i = _skip_regex(body, i) - 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and i != len(body) - 1:
                raise AssertionError(
                    f"{name}() body closes at offset {i} of {len(body) - 1} — it is "
                    "truncated or over-long, so any absence assertion against it "
                    "would pass vacuously."
                )
        i += 1
    if depth != 0:
        raise AssertionError(
            f"{name}() body never balances (depth {depth}) — it is truncated, so "
            "any absence assertion against it would pass vacuously."
        )


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


# ── N-6: truncation must not defang an absence assertion ───────────
def test_truncated_body_is_rejected_rather_than_returned():
    """The reviewer's scenario: a shorter body makes absence checks vacuous.

    _assert_whole_body re-derives the balance, so a body cut at an inner brace
    cannot be handed back as if it were the whole function.
    """
    truncated = "{ if (x) { alpha(); }"
    with pytest.raises(AssertionError, match="truncated"):
        _assert_whole_body(truncated, "fake")


def test_whole_body_accepts_a_complete_function():
    _assert_whole_body("{ if (x) { alpha(); } }", "fake")


def test_absence_assertions_cannot_pass_on_a_truncated_body():
    """The property N-6 is really about, stated directly.

    A canary surviving truncation is what made the old assertion vacuous, so
    the guard must fire even when the canary is present in the kept part.
    """
    source = (
        "function target() {\n"
        "  canary();\n"
        "  if (cond) { inner(); }\n"
        "  forbidden();\n"
        "}\n"
    )
    body = js_function_body(source, "target")
    assert "canary()" in body
    assert "forbidden()" in body, "the whole body must be returned, not a prefix"

    # Simulate the truncating helper the reviewer demonstrated.
    cut = source[source.index("{") : source.index("if (cond) { inner(); }") + len("if (cond) { inner(); }")]
    assert "canary()" in cut, "the canary survives truncation — that was the trap"
    assert "forbidden()" not in cut, "so an absence check would pass vacuously"
    with pytest.raises(AssertionError, match="truncated"):
        _assert_whole_body(cut, "target")


def test_body_must_be_brace_delimited():
    with pytest.raises(AssertionError, match="not brace-delimited"):
        _assert_whole_body("alpha();", "fake")


# ── N-5 remnant: braces inside regex literals ──────────────────────
@pytest.mark.parametrize("regex", [
    "/^\\{/",        # a lone opening brace — unbalanced upward
    "/\\}$/",        # a lone closing brace — closes the body early
    "/a{2,/",        # a malformed quantifier, still just text to the scanner
])
def test_unbalanced_braces_inside_a_regex_literal_do_not_break_scoping(regex):
    """The braces must be genuinely unbalanced for this to bite.

    My first version used /\\d{2,}/, whose braces balance — so removing regex
    handling altogether left the test green. Caught by mutation-testing my own
    fix, which is the third time in this task that a test of mine could not
    have failed.
    """
    source = f"function a() {{ const re = {regex}; alpha(); }}\nfunction b() {{ beta(); }}"
    body = js_function_body(source, "a")
    assert "alpha()" in body
    assert "beta()" not in body


def test_regex_with_a_brace_in_a_character_class():
    source = "function a() { const re = /[{}]/g; alpha(); }\nfunction b() { beta(); }"
    body = js_function_body(source, "a")
    assert "alpha()" in body
    assert "beta()" not in body


def test_division_is_not_mistaken_for_a_regex():
    """The heuristic must not swallow code after an ordinary division."""
    source = "function a() { const r = w / h; alpha(); }\nfunction b() { beta(); }"
    body = js_function_body(source, "a")
    assert "alpha()" in body
    assert "beta()" not in body


def test_an_unterminated_regex_raises_rather_than_guessing():
    source = "function a() { const re = /unterminated\n alpha(); }"
    with pytest.raises(AssertionError, match="unterminated regex"):
        js_function_body(source, "a")


def test_the_real_app_js_still_scopes_after_regex_handling():
    """Guards against the heuristic mis-firing on the actual file."""
    code = _code_only(APP_JS)
    for name in ("followRow", "updatePlayhead", "setFollow", "bindTranscriptFollow"):
        body = js_function_body(code, name)
        assert body.startswith("{") and body.endswith("}")
        assert len(body) > 40, f"{name}() body suspiciously short: {len(body)}"


def test_js_function_body_actually_calls_the_truncation_guard(monkeypatch):
    """Pins the WIRING, not just the guard's existence.

    Found by mutation-testing my own fix: deleting the _assert_whole_body call
    from js_function_body failed nothing, because every other N-6 test called
    the guard directly. That is the same definition-versus-call-site gap N-4
    was about, reproduced inside the fix for N-6.

    The guard is defence against a future broken matcher, so the only way to
    exercise it through the public function is to break the matcher on purpose.
    """
    import tests.test_follow_scroll as module

    source = "function target() {\n  canary();\n  if (c) { inner(); }\n  forbidden();\n}\n"
    assert "forbidden()" in js_function_body(source, "target"), "sanity: unbroken first"

    def truncating_scan(text, brace):
        # Return the first inner closing brace, i.e. the classic early exit.
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                return i
        raise AssertionError("unbalanced braces")

    monkeypatch.setattr(module, "_scan_to_matching_brace", truncating_scan)
    with pytest.raises(AssertionError, match="truncated"):
        module.js_function_body(source, "target")
