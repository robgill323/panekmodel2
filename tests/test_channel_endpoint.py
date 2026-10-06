"""GET /api/channel/videos, driven the way the SPA drives it.

app.js builds the query with URLSearchParams({url, count}) and reads failures
through api(), which shows ``body.detail`` as the message and now also passes
``body.kind`` on so the picker can tell "nothing to list" from a real error.
So these tests send string params and assert that every failure carries a
*string* detail — a dict or list there renders as "[object Object]".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from panekmodel2 import channel_resolver as cr
from panekmodel2.server.app import create_app
from panekmodel2.server.jobs import JobManager

from .conftest import FakeRunner
from .test_channel_resolver import FakeModule, FakeYDL, entry, vid


@pytest.fixture
def client(settings, cache_home, monkeypatch):
    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)
    FakeYDL.instances = []
    FakeYDL.info = {"channel": "Test Channel", "entries": [entry(vid(i)) for i in range(30)]}
    FakeYDL.error = None
    FakeYDL.gate = None
    monkeypatch.setattr(cr, "_yt_dlp", FakeModule)
    app = create_app(JobManager(runner_factory=lambda s: FakeRunner(s)))
    with TestClient(app) as test_client:
        yield test_client
    if FakeYDL.gate is not None:
        FakeYDL.gate.set()


def get(client, url, count=None):
    params = {"url": url}
    if count is not None:
        params["count"] = str(count)
    return client.get("/api/channel/videos", params=params)


def test_lists_videos_in_the_shape_the_picker_reads(client):
    res = get(client, "https://www.youtube.com/@NASA", 25)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["channel"] == "Test Channel"
    assert body["channel_url"] == "https://www.youtube.com/@NASA/videos"
    assert body["requested"] == 25
    assert len(body["videos"]) == 25
    assert set(body["videos"][0]) == {"id", "title", "duration_seconds", "url", "thumbnail_url"}


def test_the_route_is_not_shadowed_by_the_static_mount(client):
    """The SPA is mounted at "/", after the API. A JSON body proves the route won."""
    res = get(client, "@NASA", 3)
    assert res.headers["content-type"].startswith("application/json")


def test_count_defaults_to_25(client):
    assert get(client, "@NASA").json()["requested"] == 25
    assert FakeYDL.instances[-1].opts["playlistend"] == 25


def test_count_is_capped_at_100(client):
    FakeYDL.info = {"channel": "Big", "entries": [entry(vid(i)) for i in range(150)]}
    body = get(client, "@Big", 500).json()
    assert body["requested"] == 100 and len(body["videos"]) == 100
    assert FakeYDL.instances[-1].opts["playlistend"] == 100


def test_a_non_numeric_count_is_rejected_before_yt_dlp(client):
    assert get(client, "@NASA", "lots").status_code == 422
    assert FakeYDL.instances == []


@pytest.mark.parametrize("bad", [
    "file:///etc/passwd", "/etc/passwd", "https://evil.example/@NASA",
    "https://www.youtube.com/watch?v=ErAqN6gXqZQ", "",
])
def test_invalid_channel_urls_are_400_and_never_reach_yt_dlp(client, bad):
    res = get(client, bad, 5)
    assert res.status_code == 400
    body = res.json()
    assert body["kind"] == "invalid"
    assert isinstance(body["detail"], str) and "channel" in body["detail"].lower()
    assert FakeYDL.instances == []


def test_missing_url_is_rejected(client):
    assert client.get("/api/channel/videos").status_code == 422


@pytest.mark.parametrize("message,status,kind", [
    ("ERROR: [youtube:tab] UCx: YouTube said: This channel does not exist.", 404, "not_found"),
    ("ERROR: [youtube:tab] @x: This account has been terminated", 404, "private_or_empty"),
    ("ERROR: Unable to download webpage: <urlopen error [Errno 8] nodename nor servname provided>",
     502, "unreachable"),
    ("ERROR: something new", 502, "failed"),
])
def test_yt_dlp_failures_are_plain_json_errors(client, message, status, kind):
    FakeYDL.error = RuntimeError(message)
    res = get(client, "@NASA", 5)
    assert res.status_code == status
    body = res.json()
    assert body["kind"] == kind
    assert isinstance(body["detail"], str)
    assert "ERROR" not in body["detail"]


def test_a_keyword_handle_on_a_dead_network_is_502_not_empty(client, monkeypatch):
    """B-1, end to end as the reviewer ran it: real yt-dlp, closed local proxy."""
    yt_dlp = pytest.importorskip("yt_dlp")
    real_options = cr.ydl_options
    monkeypatch.setattr(cr, "_yt_dlp", yt_dlp)
    monkeypatch.setattr(cr, "ydl_options", lambda n: dict(real_options(n), proxy="http://127.0.0.1:9"))
    res = get(client, "@PrivateEquityTalks", 5)
    assert res.status_code == 502
    assert res.json()["kind"] == "unreachable"


def test_an_empty_channel_is_the_private_or_empty_kind(client):
    FakeYDL.info = {"channel": "Empty", "entries": []}
    res = get(client, "@Empty", 5)
    assert res.status_code == 404
    assert res.json()["kind"] == "private_or_empty"


def test_a_slow_listing_times_out_as_504(client, monkeypatch):
    import threading

    monkeypatch.setattr(cr, "FETCH_TIMEOUT_SECONDS", 0.05)
    FakeYDL.gate = threading.Event()
    res = get(client, "@Slow", 5)
    assert res.status_code == 504
    assert res.json()["kind"] == "timeout"


def test_every_emitted_id_is_strict_even_from_a_hostile_payload(client):
    FakeYDL.info = {"entries": [
        entry(vid(1)), entry('"><img src=x>'), entry("abcdefghijk\n"), entry(vid(2)),
    ]}
    body = get(client, "@x", 10).json()
    assert [v["id"] for v in body["videos"]] == [vid(1), vid(2)]
    assert body["skipped"] == 2
