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

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


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


def test_row_already_visible_does_not_move_the_reader():
    """The whole point: no scrolling unless the row actually left the view."""
    assert follow_scroll_top(rowTop=200, **BAND) is None


def test_row_above_the_view_scrolls_up_to_it():
    target = follow_scroll_top(rowTop=20, **BAND)
    assert target is not None and target < BAND["scrollTop"]


def test_row_below_the_view_scrolls_down_to_it():
    target = follow_scroll_top(rowTop=900, **BAND)
    assert target is not None and target > BAND["scrollTop"]
    # Its bottom lands inside the visible band, not past it.
    assert target <= 900


def test_target_never_goes_negative():
    assert follow_scroll_top(rowTop=0, scrollTop=50, viewHeight=400,
                             contentHeight=2000, rowHeight=40, margin=12) == 0


def test_target_is_clamped_to_the_scrollable_range():
    target = follow_scroll_top(rowTop=1980, scrollTop=0, viewHeight=400,
                               contentHeight=2000, rowHeight=40)
    assert target == 2000 - 400


def test_a_row_taller_than_the_container_shows_its_start():
    """Otherwise a long chunk would scroll past its own first line."""
    target = follow_scroll_top(rowTop=1000, rowHeight=900, scrollTop=0,
                               viewHeight=400, contentHeight=3000, margin=12)
    assert target == 1000 - 12


def test_zero_height_container_is_a_no_op():
    """Before layout, or on a hidden panel, there is nothing to scroll."""
    assert follow_scroll_top(rowTop=500, rowHeight=40, scrollTop=0,
                             viewHeight=0, contentHeight=0) is None


def test_a_no_op_move_reports_none():
    """Returning the current position would reset the programmatic flag for
    nothing, and could swallow the next genuine user scroll."""
    assert follow_scroll_top(rowTop=0, rowHeight=40, scrollTop=0,
                             viewHeight=400, contentHeight=2000, margin=0) is None


@pytest.mark.parametrize("row_top", [0, 37, 400, 1234, 1960])
def test_result_is_always_a_valid_scroll_position(row_top):
    target = follow_scroll_top(rowTop=row_top, **BAND)
    if target is not None:
        assert 0 <= target <= BAND["contentHeight"] - BAND["viewHeight"]


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


def test_the_page_scroll_is_never_part_of_the_calculation():
    """The function only knows container-relative numbers.

    It cannot touch the document even in principle, which is the property the
    scrollIntoView version violated.
    """
    code = _code_only(FOLLOW_JS)
    assert ".scrollIntoView(" not in code
    assert "window." not in code and "document." not in code


def test_app_no_longer_calls_scrollintoview():
    code = _code_only(FOLLOW_JS.parent / "app.js")
    assert ".scrollIntoView(" not in code, (
        "scrollIntoView scrolls every scrollable ancestor, including the page"
    )


def test_auto_follow_scrolls_a_container_not_an_element():
    """The replacement must assign container.scrollTop, nothing wider."""
    code = _code_only(FOLLOW_JS.parent / "app.js")
    assert "container.scrollTop = target" in code
    assert "followScrollTop({" in code


def test_highlighting_is_not_gated_on_follow_state():
    """Property 3: pausing auto-follow pauses scrolling only."""
    code = _code_only(FOLLOW_JS.parent / "app.js")
    highlight = code.index("row.classList.toggle('active', on)")
    follow_call = code.index("if (activeRow && S.playing && S.followTranscript)")
    assert highlight < follow_call, "highlighting must happen before/independent of following"
