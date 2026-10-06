"""Channel listing for the New Run picker, with yt-dlp replaced throughout.

The channel URL is new input surface: it is the first value in this app that a
user types and the server hands to yt-dlp as a *URL to resolve* rather than as
an 11-character ID. So most of what is pinned here is the boundary — what is
accepted, what is rebuilt, what never reaches yt-dlp at all — and that every
ID coming back out is held to the same strict check the pipeline uses.

Nothing in this file touches the network. The one test that builds a real
YoutubeDL does so to prove an offline property (no extractor will take a
non-YouTube URL), and it is given a file:// URL that no allowed extractor
claims.
"""

from __future__ import annotations

import threading

import pytest

from panekmodel2 import channel_resolver as cr
from panekmodel2.channel_resolver import (
    ChannelError,
    DEFAULT_COUNT,
    MAX_COUNT,
    clamp_count,
    classify_error,
    is_strict_video_id,
    list_channel_videos,
    normalize_channel_url,
    thumbnail_url,
    video_from_entry,
    ydl_options,
)

UC_ID = "UC" + "A1b2C3d4E5f6G7h8I9j0K_"  # UC + 22
assert len(UC_ID) == 24


def vid(n: int) -> str:
    """A distinct, valid 11-character video ID."""
    return f"v{n:010d}"


def entry(video_id: str, **extra) -> dict:
    """One flat-mode entry, shaped like the live payload captured for @NASA."""
    base = {
        "_type": "url",
        "ie_key": "Youtube",
        "id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": f"Video {video_id}",
        "duration": 62,
        "timestamp": None,
    }
    base.update(extra)
    return base


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL. Records every construction and call."""

    instances: list["FakeYDL"] = []
    info: dict | None = None
    error: BaseException | None = None
    gate: threading.Event | None = None

    def __init__(self, opts):
        self.opts = opts
        self.calls: list[tuple[str, dict]] = []
        FakeYDL.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if FakeYDL.gate is not None:
            FakeYDL.gate.wait(5)
        if FakeYDL.error is not None:
            raise FakeYDL.error
        return FakeYDL.info


class FakeModule:
    YoutubeDL = FakeYDL


@pytest.fixture
def ydl(monkeypatch):
    FakeYDL.instances = []
    FakeYDL.info = {"channel": "Test Channel", "entries": [entry(vid(i)) for i in range(5)]}
    FakeYDL.error = None
    FakeYDL.gate = None
    monkeypatch.setattr(cr, "_yt_dlp", FakeModule)
    yield FakeYDL
    if FakeYDL.gate is not None:
        FakeYDL.gate.set()


def only_call(fake=FakeYDL):
    assert len(fake.instances) == 1, f"expected one YoutubeDL, got {len(fake.instances)}"
    inst = fake.instances[0]
    assert len(inst.calls) == 1
    return inst.opts, inst.calls[0][0], inst.calls[0][1]


# ── normalization: every accepted form ──────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("https://www.youtube.com/@NASA", "https://www.youtube.com/@NASA/videos"),
    ("https://www.youtube.com/@NASA/videos", "https://www.youtube.com/@NASA/videos"),
    ("https://www.youtube.com/@NASA/", "https://www.youtube.com/@NASA/videos"),
    ("https://www.youtube.com/@NASA/videos/", "https://www.youtube.com/@NASA/videos"),
    ("http://youtube.com/@some.handle-2_x", "https://www.youtube.com/@some.handle-2_x/videos"),
    ("https://m.youtube.com/@NASA?si=abc#frag", "https://www.youtube.com/@NASA/videos"),
    ("youtube.com/@NASA", "https://www.youtube.com/@NASA/videos"),
    ("www.youtube.com/@NASA/videos", "https://www.youtube.com/@NASA/videos"),
    ("  @NASA  ", "https://www.youtube.com/@NASA/videos"),
    (f"https://www.youtube.com/channel/{UC_ID}", f"https://www.youtube.com/channel/{UC_ID}/videos"),
    (f"https://www.youtube.com/channel/{UC_ID}/videos", f"https://www.youtube.com/channel/{UC_ID}/videos"),
    ("https://www.youtube.com/c/NASA", "https://www.youtube.com/c/NASA/videos"),
    ("https://www.youtube.com/c/NASA/videos", "https://www.youtube.com/c/NASA/videos"),
    ("https://www.youtube.com/user/NASAtelevision", "https://www.youtube.com/user/NASAtelevision/videos"),
    ("https://www.youtube.com/user/NASAtelevision/videos", "https://www.youtube.com/user/NASAtelevision/videos"),
    ("HTTPS://WWW.YOUTUBE.COM/@NASA", "https://www.youtube.com/@NASA/videos"),
])
def test_accepted_forms_are_rebuilt_canonically(raw, expected):
    assert normalize_channel_url(raw) == expected


@pytest.mark.parametrize("raw", [
    # not a URL at all, or a local path
    "", "   ", "/etc/passwd", "../../etc/passwd", "file:///etc/passwd",
    "C:\\Users\\me\\videos", "~/channel",
    # the wrong host, or a host dressed up as YouTube
    "https://evil.example/@NASA", "https://youtube.com.evil.example/@NASA",
    "https://notyoutube.com/@NASA", "https://music.youtube.com/@NASA",
    "https://youtu.be/@NASA", "https://www.youtube.com@evil.example/@NASA",
    "https://user:pw@www.youtube.com/@NASA", "https://www.youtube.com:8080/@NASA",
    # the wrong scheme
    "javascript:alert(1)", "ftp://www.youtube.com/@NASA", "data:text/html,x",
    # YouTube, but not a channel's video list
    "https://www.youtube.com/watch?v=ErAqN6gXqZQ",
    "https://www.youtube.com/playlist?list=PL0123456789",
    "https://www.youtube.com/@NASA/shorts", "https://www.youtube.com/@NASA/streams",
    "https://www.youtube.com/@NASA/videos/extra", "https://www.youtube.com/",
    "https://www.youtube.com/NASA",
    # malformed identifiers
    "https://www.youtube.com/channel/UCshort", "https://www.youtube.com/channel/XX" + "a" * 22,
    "https://www.youtube.com/channel/" + UC_ID + "x",
    "https://www.youtube.com/@", "https://www.youtube.com/@..", "https://www.youtube.com/c/..",
    "https://www.youtube.com/@a%2F..", "https://www.youtube.com/@a/../b",
    "https://www.youtube.com/@na sa", "https://www.youtube.com/@ナサ", "@NA\nSA", "@NASA\x00",
    "https://www.youtube.com/@NASA\\videos",
    "https://www.youtube.com/@" + "a" * 400,
    # urlsplit silently DELETES \t \r \n, so without an explicit check these
    # would be accepted as @NASA. Found by mutation, not by inspection.
    "https://www.youtube.com/@NA\nSA", "https://www.youtube.com/@NA\tSA",
    "https://www.youtube.com/@NA\rSA",
    # The query is dropped, but the input is still refused whole: oversize or
    # non-ASCII anywhere means it is not a pasted channel link.
    "https://www.youtube.com/@NASA?si=" + "a" * 400,
    "https://www.youtube.com/@NASA?q=ナサ",
])
def test_everything_else_is_refused(raw):
    with pytest.raises(ChannelError) as info:
        normalize_channel_url(raw)
    assert info.value.kind == "invalid"
    assert "channel" in info.value.message.lower()


@pytest.mark.parametrize("raw", [None, 42, b"@NASA", ["@NASA"]])
def test_non_strings_are_refused(raw):
    with pytest.raises(ChannelError) as info:
        normalize_channel_url(raw)
    assert info.value.kind == "invalid"


def test_a_refused_url_never_reaches_yt_dlp(ydl):
    with pytest.raises(ChannelError):
        list_channel_videos("file:///etc/passwd")
    assert ydl.instances == []


def test_yt_dlp_is_handed_the_rebuilt_url_not_the_raw_input(ydl):
    list_channel_videos("m.youtube.com/@NASA/?si=tracking#x", 5)
    _, url, _ = only_call()
    assert url == "https://www.youtube.com/@NASA/videos"


# ── count ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    (25, 25), (1, 1), (100, 100), (0, 1), (-7, 1), (101, 100), (10_000, 100),
])
def test_count_is_clamped(raw, expected):
    assert clamp_count(raw) == expected


def test_count_defaults_and_limits_are_what_the_ui_promises():
    assert DEFAULT_COUNT == 25
    assert MAX_COUNT == 100


def test_count_cap_reaches_yt_dlp_and_the_result(ydl):
    ydl.info = {"channel": "Big", "entries": [entry(vid(i)) for i in range(150)]}
    out = list_channel_videos("@Big", 5000)
    opts, _, _ = only_call()
    assert opts["playlistend"] == 100
    assert len(out["videos"]) == 100
    assert out["requested"] == 100


def test_count_is_honoured_even_if_yt_dlp_returns_more(ydl):
    """playlistend is a request; the slice is the guarantee."""
    ydl.info = {"channel": "Big", "entries": [entry(vid(i)) for i in range(40)]}
    out = list_channel_videos("@Big", 7)
    assert only_call()[0]["playlistend"] == 7
    assert [v["id"] for v in out["videos"]] == [vid(i) for i in range(7)]


def test_entries_may_be_a_generator(ydl):
    """yt-dlp hands back lazy entries for tabs; they are drained in the worker."""
    ydl.info = {"channel": "Lazy", "entries": (entry(vid(i)) for i in range(3))}
    assert len(list_channel_videos("@Lazy", 10)["videos"]) == 3


# ── flat, metadata-only extraction ──────────────────────────────────
def test_extraction_is_flat_and_downloads_nothing(ydl):
    list_channel_videos("@NASA", 10)
    opts, _, kwargs = only_call()
    assert opts["extract_flat"] is True
    assert opts["skip_download"] is True
    assert kwargs.get("download") is False


def test_only_the_channel_tab_extractor_is_allowed(ydl):
    list_channel_videos("@NASA", 10)
    assert only_call()[0]["allowed_extractors"] == ["youtube:tab"]


def test_options_bound_the_network_wait():
    opts = ydl_options(10)
    assert 0 < opts["socket_timeout"] <= cr.FETCH_TIMEOUT_SECONDS


def test_real_yt_dlp_with_these_options_refuses_a_non_youtube_url(capfd):
    """Defence in depth behind the validator, checked against real yt-dlp.

    file:// is claimed by no extractor in the allow-list, so this fails inside
    yt-dlp before any I/O happens — it runs offline. It also shows the quiet
    logger working: yt-dlp prints ERROR lines to stderr even with quiet=True,
    which would otherwise land in the server's terminal on every bad channel.
    """
    yt_dlp = pytest.importorskip("yt_dlp")
    with yt_dlp.YoutubeDL(ydl_options(1)) as real:
        assert list(real._ies) == ["YoutubeTab"], "the allow-list loads exactly one extractor"
        with pytest.raises(yt_dlp.utils.DownloadError, match="No suitable extractor"):
            real.extract_info("file:///etc/passwd", download=False)
    assert "No suitable extractor" not in capfd.readouterr().err


# ── what comes back out ─────────────────────────────────────────────
def test_result_shape(ydl):
    out = list_channel_videos("@NASA", 25)
    assert out["channel"] == "Test Channel"
    assert out["channel_url"] == "https://www.youtube.com/@NASA/videos"
    assert out["requested"] == 25
    assert out["skipped"] == 0
    first = out["videos"][0]
    assert first == {
        "id": vid(0),
        "title": f"Video {vid(0)}",
        "duration_seconds": 62,
        "url": f"https://www.youtube.com/watch?v={vid(0)}",
        "thumbnail_url": f"https://i.ytimg.com/vi/{vid(0)}/mqdefault.jpg",
    }


def test_order_is_preserved_newest_first(ydl):
    out = list_channel_videos("@NASA", 25)
    assert [v["id"] for v in out["videos"]] == [vid(i) for i in range(5)]


def test_channel_name_falls_back_through_uploader_and_tab_title(ydl):
    ydl.info = {"uploader": "Uploader Name", "entries": [entry(vid(1))]}
    assert list_channel_videos("@x", 5)["channel"] == "Uploader Name"
    ydl.instances.clear()
    ydl.info = {"title": "Tab Name - Videos", "entries": [entry(vid(1))]}
    assert list_channel_videos("@x", 5)["channel"] == "Tab Name"
    ydl.instances.clear()
    ydl.info = {"entries": [entry(vid(1))]}
    assert list_channel_videos("@x", 5)["channel"] == ""


@pytest.mark.parametrize("bad_id", [
    "short", "x" * 12, "abc/def/ghi", "abcdefghij!", "abcdefghijk\n", "ab cdefghijk",
    "<script>xx>", None, 12345678901, "",
])
def test_ids_failing_strict_validation_are_never_emitted(ydl, bad_id):
    ydl.info = {"channel": "Mixed", "entries": [entry(vid(1)), entry(bad_id), entry(vid(2))]}
    out = list_channel_videos("@Mixed", 10)
    assert [v["id"] for v in out["videos"]] == [vid(1), vid(2)]
    assert out["skipped"] == 1


def test_the_strict_check_is_stricter_than_extract_video_ids_dollar():
    """extract_video_id uses re.match with "$", which also matches before a
    trailing newline. An ID bound for a URL and an HTML attribute must not
    carry one, so the emit check is a full match on top of it."""
    from panekmodel2.pipeline import extract_video_id

    assert extract_video_id("abcdefghijk\n") == "abcdefghijk\n"  # the gap
    assert is_strict_video_id("abcdefghijk") is True
    assert is_strict_video_id("abcdefghijk\n") is False


def test_the_strict_check_agrees_with_the_pipeline():
    """An ID the picker emits is one the pipeline will accept as-is."""
    from panekmodel2.pipeline import extract_video_id

    for good in ("ErAqN6gXqZQ", "a-b_c-d_e-f", vid(3)):
        assert is_strict_video_id(good) and extract_video_id(good) == good


def test_watch_and_thumbnail_urls_are_rebuilt_from_the_id(ydl):
    """An entry's own url field is never echoed: it is not ours to trust."""
    ydl.info = {"entries": [entry(vid(1), url="https://evil.example/watch?v=" + vid(1),
                                  thumbnails=[{"url": "https://evil.example/t.jpg"}])]}
    video = list_channel_videos("@x", 5)["videos"][0]
    assert video["url"] == f"https://www.youtube.com/watch?v={vid(1)}"
    assert video["thumbnail_url"] == f"https://i.ytimg.com/vi/{vid(1)}/mqdefault.jpg"


def test_nested_playlists_and_other_extractors_are_skipped(ydl):
    ydl.info = {"entries": [
        entry(vid(1)),
        {"_type": "playlist", "id": "PLxxxxxxxxx", "title": "a tab"},
        entry(vid(2), ie_key="YoutubeTab"),
        "not even a dict",
        None,
    ]}
    out = list_channel_videos("@x", 10)
    assert [v["id"] for v in out["videos"]] == [vid(1)]
    assert out["skipped"] == 4


def test_duplicates_are_listed_once(ydl):
    ydl.info = {"entries": [entry(vid(1)), entry(vid(1)), entry(vid(2))]}
    out = list_channel_videos("@x", 10)
    assert [v["id"] for v in out["videos"]] == [vid(1), vid(2)]


@pytest.mark.parametrize("raw,expected", [
    (62, 62), (61.6, 62), (0, 0), (None, None), (-1, None), (True, None),
    ("62", None), (float("nan"), None), (float("inf"), None),
])
def test_duration_is_reported_only_when_flat_mode_gave_a_real_number(raw, expected):
    assert video_from_entry(entry(vid(1), duration=raw))["duration_seconds"] == expected


def test_missing_duration_key_is_none():
    e = entry(vid(1))
    del e["duration"]
    assert video_from_entry(e)["duration_seconds"] is None


@pytest.mark.parametrize("raw,expected", [
    ("  Padded  ", "Padded"), (None, ""), (42, ""), ("", ""),
])
def test_title_is_a_trimmed_string(raw, expected):
    assert video_from_entry(entry(vid(1), title=raw))["title"] == expected


def test_no_listable_videos_is_the_private_or_empty_error(ydl):
    for info in ({"entries": []}, {"entries": None}, {}, {"entries": [entry("bad")]}):
        ydl.instances.clear()
        ydl.info = info
        with pytest.raises(ChannelError) as err:
            list_channel_videos("@x", 5)
        assert err.value.kind == "private_or_empty"


def test_a_non_dict_payload_is_the_private_or_empty_error(ydl):
    ydl.info = None
    with pytest.raises(ChannelError) as err:
        list_channel_videos("@x", 5)
    assert err.value.kind == "private_or_empty"


# ── thumbnails ──────────────────────────────────────────────────────
def test_thumbnail_url_is_derived_from_the_id():
    assert thumbnail_url("ErAqN6gXqZQ") == "https://i.ytimg.com/vi/ErAqN6gXqZQ/mqdefault.jpg"


@pytest.mark.parametrize("bad", ["", "short", "abcdefghijk\n", "../../x/abcd", None])
def test_thumbnail_url_refuses_anything_but_a_strict_id(bad):
    with pytest.raises(ValueError):
        thumbnail_url(bad)


# ── errors in plain language ────────────────────────────────────────
# The first two messages are verbatim from live yt-dlp 2026.08.19 runs.
@pytest.mark.parametrize("text,kind", [
    ("ERROR: [youtube:tab] @nope/videos: Unable to download API page: HTTP Error 404: Not Found "
     "(caused by <HTTPError 404: Not Found>)", "not_found"),
    ("ERROR: [youtube:tab] UCAAAAAAAAAAAAAAAAAAAAAA: YouTube said: This channel does not exist.", "not_found"),
    ("ERROR: [youtube:tab] UCxx: This channel does not have a videos tab", "private_or_empty"),
    ("ERROR: [youtube:tab] @x: This account has been terminated for a violation", "private_or_empty"),
    ("ERROR: [youtube:tab] @x: This channel is not available.", "private_or_empty"),
    ("ERROR: This playlist is private", "private_or_empty"),
    ("ERROR: Unable to download webpage: <urlopen error [Errno 8] nodename nor servname provided>",
     "unreachable"),
    ("ERROR: Unable to download API page: The read operation timed out", "unreachable"),
    ("ERROR: [youtube:tab] TransportError: Connection reset by peer", "unreachable"),
    ("ERROR: something nobody anticipated", "failed"),
])
def test_yt_dlp_errors_become_plain_language(text, kind):
    err = classify_error(RuntimeError(text))
    assert err.kind == kind
    assert "ERROR" not in err.message and "[youtube" not in err.message


# B-1 (review-10): yt-dlp's error text starts with "[youtube:tab] <item id>:",
# and for a handle URL the item id IS the handle. Every row above used a
# neutral id, so a handle containing a keyword captured every failure on that
# channel. These rows use the real prefix format with keyword-bearing ids and
# assert the kind the MESSAGE implies, not the handle.
@pytest.mark.parametrize("text,kind", [
    ("ERROR: [youtube:tab] @privatelab/videos: Unable to download API page: ('Unable to connect "
     "to proxy', NewConnectionError(\"HTTPSConnection(host='127.0.0.1', port=9): Failed to "
     "establish a new connection: [Errno 61] Connection refused\"))", "unreachable"),
    ("ERROR: [youtube:tab] @PrivateEquityTalks/videos: Unable to download webpage: <urlopen "
     "error [Errno 8] nodename nor servname provided>", "unreachable"),
    ("ERROR: [youtube:tab] @unavailableband/videos: Unable to download API page: HTTP Error 404: "
     "Not Found (caused by <HTTPError 404: Not Found>)", "not_found"),
    ("ERROR: [youtube:tab] @terminatedtv/videos: The read operation timed out", "unreachable"),
    ("ERROR: [youtube:tab] @connectionsmusic/videos: something nobody anticipated", "failed"),
    ("ERROR: [youtube:tab] @errnolab/videos: something nobody anticipated", "failed"),
    ("ERROR: [youtube:tab] @privatelab/videos: This channel does not have a videos tab",
     "private_or_empty"),
    ("ERROR: [youtube:tab] UCprivate000000000000000: YouTube said: This channel does not exist.",
     "not_found"),
])
def test_the_echoed_item_id_never_steers_the_classification(text, kind):
    assert classify_error(RuntimeError(text)).kind == kind


def test_the_channel_base_is_removed_wherever_the_error_echoes_it():
    """A transport message can repeat the request path after the prefix."""
    text = "ERROR: [youtube:tab] @privatelab/videos: Failed fetching /@privatelab/videos?x=1"
    assert classify_error(RuntimeError(text)).kind == "private_or_empty"  # the trap
    assert classify_error(RuntimeError(text), echoed="@privatelab").kind == "failed"


@pytest.mark.parametrize("echo", [
    "/c/privatelab/videos",
    "/c/privatelab?continuation=4qmFsgK",  # the base alone, without the tab suffix
])
def test_the_listing_passes_its_own_base_to_the_classifier(ydl, echo):
    ydl.error = RuntimeError(f"ERROR: [youtube:tab] c/privatelab/videos: Failed fetching {echo}")
    with pytest.raises(ChannelError) as err:
        list_channel_videos("https://www.youtube.com/c/privatelab", 5)
    assert err.value.kind == "failed"


def test_an_unwrapped_extractor_error_is_stripped_too(ydl):
    """ExtractorError formats as "[ie] id: msg" with no "ERROR:", and one can
    escape unwrapped — tab pages are fetched lazily, inside the worker's own
    drain of the entries rather than inside extract_info."""
    from yt_dlp.utils import ExtractorError

    raw = ExtractorError("The read operation timed out", ie="youtube:tab", video_id="@privatelab/videos")
    assert str(raw).startswith("[youtube:tab] @privatelab/videos: ")
    assert classify_error(raw).kind == "unreachable"
    ydl.error = ExtractorError("This channel does not exist.", ie="youtube:tab",
                               video_id="@unavailableband/videos", expected=True)
    with pytest.raises(ChannelError) as err:
        list_channel_videos("@unavailableband", 5)
    assert err.value.kind == "not_found"


def test_a_transport_failure_is_unreachable_whatever_the_text_says():
    """The structured cause outranks every substring."""
    from yt_dlp.networking.exceptions import ProxyError
    from yt_dlp.utils import DownloadError, ExtractorError

    cause = ProxyError("Unable to connect to proxy")
    direct = DownloadError("ERROR: [youtube:tab] @x/videos: This channel is private",
                           exc_info=(ProxyError, cause, None))
    assert classify_error(direct).kind == "unreachable"
    wrapped = DownloadError("ERROR: [youtube:tab] @x/videos: This channel is private",
                            exc_info=(ExtractorError, ExtractorError("x", cause=cause), None))
    assert classify_error(wrapped).kind == "unreachable"


def test_a_non_transport_cause_leaves_the_text_in_charge():
    from yt_dlp.utils import DownloadError

    err = DownloadError("ERROR: [youtube:tab] UCx: YouTube said: This channel does not exist.",
                        exc_info=(ValueError, ValueError("x"), None))
    assert classify_error(err).kind == "not_found"


@pytest.mark.parametrize("handle", ["@plainlab", "@PrivateEquityTalks", "@unavailableband", "@terminatedtv"])
def test_real_yt_dlp_network_failure_is_unreachable_offline(monkeypatch, handle):
    """The reviewer's reproduction, kept: real yt-dlp, real error, no network.

    The proxy is a closed local port, so the connection is refused on this
    machine and nothing leaves it. This replaces "needs a live capture" for
    the network-failure kind.
    """
    yt_dlp = pytest.importorskip("yt_dlp")
    real_options = cr.ydl_options
    monkeypatch.setattr(cr, "_yt_dlp", yt_dlp)
    monkeypatch.setattr(cr, "ydl_options", lambda n: dict(real_options(n), proxy="http://127.0.0.1:9"))
    with pytest.raises(ChannelError) as err:
        list_channel_videos(handle, 1)
    assert err.value.kind == "unreachable"


def test_every_kind_has_a_status_and_a_sentence():
    for kind in ("invalid", "not_found", "private_or_empty", "unreachable", "timeout", "failed"):
        err = ChannelError(kind)
        assert 400 <= err.status < 600
        assert err.message.endswith(".") and len(err.message) > 30


def test_a_yt_dlp_failure_surfaces_classified(ydl):
    ydl.error = RuntimeError("ERROR: [youtube:tab] UCx: YouTube said: This channel does not exist.")
    with pytest.raises(ChannelError) as err:
        list_channel_videos("@x", 5)
    assert err.value.kind == "not_found"


def test_the_fetch_is_timed_out(ydl, monkeypatch):
    ydl.gate = threading.Event()  # extract_info blocks until released
    with pytest.raises(ChannelError) as err:
        list_channel_videos("@slow", 5, timeout=0.05)
    assert err.value.kind == "timeout"
    assert err.value.status == 504


def test_an_abandoned_fetch_cannot_hold_up_process_exit(ydl):
    """The overrunning worker is left behind on timeout, so it must be a daemon:
    a non-daemon thread stuck on a socket would keep `panekmodel2 ui` alive
    after Ctrl-C until yt-dlp gave up."""
    ydl.gate = threading.Event()
    with pytest.raises(ChannelError):
        list_channel_videos("@slow", 5, timeout=0.05)
    stuck = [t for t in threading.enumerate() if t.name == cr.FETCH_THREAD_NAME and t.is_alive()]
    assert stuck, "the abandoned worker should still be running at this point"
    assert all(t.daemon for t in stuck)


def test_the_module_timeout_applies_by_default(ydl, monkeypatch):
    monkeypatch.setattr(cr, "FETCH_TIMEOUT_SECONDS", 0.05)
    ydl.gate = threading.Event()
    with pytest.raises(ChannelError) as err:
        list_channel_videos("@slow", 5)
    assert err.value.kind == "timeout"


def test_missing_yt_dlp_is_a_plain_failure(monkeypatch):
    monkeypatch.setattr(cr, "_yt_dlp", None)
    with pytest.raises(ChannelError) as err:
        list_channel_videos("@x", 5)
    assert err.value.kind == "failed"
    assert "yt-dlp" in err.value.message
