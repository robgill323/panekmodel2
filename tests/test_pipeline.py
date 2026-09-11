"""Video-ID parsing, cache-shape compatibility, and batch alignment."""

from __future__ import annotations

import pytest

from panekmodel2 import pipeline as pipeline_module
from panekmodel2.config import Settings
from panekmodel2.pipeline import (
    CACHE_ENTRY_KEYS,
    PipelineRunner,
    VideoCache,
    describe_failure,
    extract_video_id,
    redact_secrets,
)

from .conftest import FakeRunner

VALID_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        VALID_ID,
        f"https://www.youtube.com/watch?v={VALID_ID}",
        f"http://youtube.com/watch?v={VALID_ID}&t=42s",
        f"https://youtu.be/{VALID_ID}",
        f"https://youtu.be/{VALID_ID}?si=abc123",
        f"https://www.youtube.com/shorts/{VALID_ID}",
        f"https://www.youtube.com/live/{VALID_ID}",
        f"https://www.youtube.com/embed/{VALID_ID}",
        f"https://m.youtube.com/watch?v={VALID_ID}",
    ],
)
def test_extract_video_id_accepts_every_url_form(url):
    assert extract_video_id(url) == VALID_ID


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not a url",
        "https://vimeo.com/123456",
        "https://www.youtube.com/watch?v=tooshort",
        "https://example.com/watch?v=" + VALID_ID[:5],
        "'; DROP TABLE videos;--",
    ],
)
def test_extract_video_id_rejects_junk(value):
    with pytest.raises(ValueError):
        extract_video_id(value)


def test_cache_key_versioned():
    """Bumping CACHE_VERSION must change the key so stale entries are missed."""
    key = VideoCache.make_key(200, 60, "e", "s")
    assert key != VideoCache.make_key(200, 60, "e", "s2")
    assert len(key) == 10


def test_cache_rejects_entry_from_older_layout(tmp_path):
    cache = VideoCache(cache_dir=tmp_path)
    cache.save("vid", "key", {"metadata": {}, "segments": [], "chunks": []})  # pre-v2 shape
    assert cache.load("vid", "key") is None


def test_cache_round_trip(tmp_path):
    cache = VideoCache(cache_dir=tmp_path)
    entry = {k: [] for k in CACHE_ENTRY_KEYS}
    entry["video_id"] = "vid"
    cache.save("vid", "key", entry)
    assert cache.load("vid", "key")["video_id"] == "vid"


def test_cache_written_by_run_is_readable_by_run_multi(fake_runner, cache_home):
    """The regression that killed multi-video batches.

    run() used to write an entry without ``video_id``; run_multi() then read it
    and raised KeyError after all the expensive work was done.
    """
    single = fake_runner.run("https://youtu.be/" + "a" * 11, detect_people=False)
    assert single.video_id == "a" * 11
    assert fake_runner.embed_calls == 1

    result = fake_runner.run_multi(
        ["https://youtu.be/" + "a" * 11, "https://youtu.be/" + "b" * 11], detect_people=False
    )
    assert [o.video_id for o in result.outputs] == ["a" * 11, "b" * 11]
    # The first video came from cache, so only the second was embedded again.
    assert fake_runner.embed_calls == 2


def test_run_multi_result_unpacks_as_a_two_tuple(fake_runner):
    outputs, failures = fake_runner.run_multi(["https://youtu.be/" + "a" * 11], detect_people=False)
    assert len(outputs) == 1
    assert failures == []


def test_run_multi_alignment_survives_a_mid_batch_failure(fake_runner_with_failure):
    """A skipped URL must not shift attribution for every video after it."""
    runner = fake_runner_with_failure
    urls = [
        "https://youtu.be/" + "a" * 11,
        "https://youtu.be/" + "b" * 11,  # fails
        "https://youtu.be/" + "c" * 11,
        "not-a-youtube-url",             # fails to parse
    ]
    result = runner.run_multi(urls, detect_people=False)

    assert [o.video_id for o in result.outputs] == ["a" * 11, "c" * 11]
    assert [(o.url, o.status, o.video_id) for o in result.outcomes] == [
        (urls[0], "analyzed", "a" * 11),
        (urls[1], "skipped", "b" * 11),
        (urls[2], "analyzed", "c" * 11),
        (urls[3], "skipped", None),
    ]
    # Pairing by video_id — never positionally — keeps every video with its URL.
    by_id = {oc.video_id: oc.url for oc in result.outcomes if oc.status == "analyzed"}
    assert by_id["c" * 11] == urls[2]
    assert result.outputs_by_video_id()["c" * 11].metadata["title"] == "Video " + "c" * 11


def test_run_multi_rebases_chunk_index_per_video(fake_runner):
    """Each video's topics_df must index its own chunks, not the batch's."""
    result = fake_runner.run_multi(
        ["https://youtu.be/" + "a" * 11, "https://youtu.be/" + "b" * 11], detect_people=False
    )
    second = result.outputs[1]
    assert list(second.topics_df["chunk_index"]) == list(range(len(second.chunks)))
    # And the row for chunk 0 is really this video's first chunk.
    row = second.topics_df[second.topics_df["chunk_index"] == 0].iloc[0]
    assert row["text"] == second.chunks[0].text


def test_run_multi_with_every_url_failing(settings, cache_home):
    runner = FakeRunner(settings, fail_ids={"a" * 11})
    result = runner.run_multi(["https://youtu.be/" + "a" * 11], detect_people=False)
    assert result.outputs == []
    assert len(result.failures) == 1
    assert result.outcomes[0].status == "skipped"


@pytest.mark.parametrize(
    "message,expected_fragment",
    [
        ("Transcripts are disabled for this video", "disabled transcripts"),
        ("No transcript available. Errors: [...]", "No caption track"),
        ("Transcript produced no chunks (empty or unusable captions)", "came back empty"),
        ("The video was unavailable", "unavailable"),
    ],
)
def test_describe_failure_is_plain_language(message, expected_fragment):
    assert expected_fragment in describe_failure(RuntimeError(message))


def test_describe_failure_on_bad_url():
    exc = ValueError("Could not extract video id from: 'nope'")
    assert "recognizable YouTube URL" in describe_failure(exc)


@pytest.fixture
def fake_runner_with_failure(settings, cache_home):
    return FakeRunner(settings, fail_ids={"b" * 11})


# ── SEC-003: credentials must never reach the log or the UI ─────────
@pytest.mark.parametrize("text,expected_absent", [
    ("https://youtube.googleapis.com/youtube/v3/videos?id=x&key=AIzaSyREALKEY123", "AIzaSyREALKEY123"),
    ("returned 403 ... ?part=snippet&key=SECRET&alt=json", "SECRET"),
    ("access_token=ya29.LIVE_TOKEN expired", "ya29.LIVE_TOKEN"),
    ("KEY=UpperCaseSecret", "UpperCaseSecret"),
])
def test_redact_secrets_removes_credentials(text, expected_absent):
    cleaned = redact_secrets(text)
    assert expected_absent not in cleaned
    assert "REDACTED" in cleaned


def test_redact_secrets_keeps_the_rest_of_the_message():
    cleaned = redact_secrets("quotaExceeded for project 42 ?key=AIzaSyABC&part=snippet")
    assert "quotaExceeded for project 42" in cleaned
    assert "part=snippet" in cleaned


def test_redact_secrets_leaves_innocent_text_alone():
    assert redact_secrets("the monkey=wrench fell") == "the monkey=wrench fell"
    assert redact_secrets("no secrets here") == "no secrets here"


def test_redact_secrets_accepts_an_exception():
    exc = RuntimeError("HttpError 403 ... &key=AIzaSyLEAKED")
    assert "AIzaSyLEAKED" not in redact_secrets(exc)


def test_describe_failure_redacts_unrecognized_errors():
    """Unknown failures fall through verbatim to the UI and the exports."""
    exc = RuntimeError("something odd happened: https://api?key=AIzaSyLEAKED")
    reason = describe_failure(exc)
    assert "AIzaSyLEAKED" not in reason
    assert "key=REDACTED" in reason


def test_metadata_fetch_logs_no_api_key(settings, cache_home, caplog, monkeypatch):
    """The real leak path: googleapiclient puts the URL in HttpError's text."""
    import logging as _logging

    runner = FakeRunner(Settings(**{**settings.model_dump(), "youtube_api_key": "AIzaSyLEAKED"}))

    def _boom(*args, **kwargs):
        raise RuntimeError(
            "HttpError 403 when requesting "
            "https://youtube.googleapis.com/youtube/v3/videos?id=x&key=AIzaSyLEAKED returned quotaExceeded"
        )

    monkeypatch.setattr(pipeline_module, "build", _boom)
    # Keep the yt-dlp fallback offline: without this the test reaches YouTube.
    monkeypatch.setattr(pipeline_module, "_yt_dlp", None)
    with caplog.at_level(_logging.WARNING):
        meta = PipelineRunner.fetch_metadata(runner, "a" * 11)

    assert "AIzaSyLEAKED" not in caplog.text
    assert "key=REDACTED" in caplog.text
    assert isinstance(meta, dict)


def test_retired_settings_are_announced_not_silently_ignored(monkeypatch, caplog):
    """A formerly load-bearing env var must not vanish without a word."""
    import logging as _logging

    from panekmodel2.config import RETIRED_SETTINGS, warn_about_retired_settings

    monkeypatch.setenv("GOOGLE_CREDENTIALS_FILE", "/tmp/creds.json")
    with caplog.at_level(_logging.INFO):
        found = warn_about_retired_settings()

    assert found == ["GOOGLE_CREDENTIALS_FILE"]
    assert "GOOGLE_CREDENTIALS_FILE is set" in caplog.text
    assert "ignored" in caplog.text
    assert "GOOGLE_TOKEN_FILE" in RETIRED_SETTINGS


def test_no_notice_when_retired_settings_are_absent(monkeypatch, caplog):
    import logging as _logging

    from panekmodel2.config import warn_about_retired_settings

    monkeypatch.delenv("GOOGLE_CREDENTIALS_FILE", raising=False)
    monkeypatch.delenv("GOOGLE_TOKEN_FILE", raising=False)
    with caplog.at_level(_logging.INFO):
        assert warn_about_retired_settings() == []
    assert "is set but" not in caplog.text


def test_oauth_settings_are_gone_from_the_model():
    from panekmodel2.config import Settings

    fields = set(Settings().model_dump())
    assert "google_credentials_file" not in fields
    assert "google_token_file" not in fields


# ── entity detection moved into the per-video pass (story-3a) ───────
def test_entities_are_cached_with_the_video(settings, cache_home):
    """NER was ~84% of a real run's wall clock and re-ran on every batch.

    Detecting per video means the result lands in the cache, so a repeat run
    does no NER work at all.
    """
    runner = FakeRunner(settings)
    calls = []
    real_detect = runner._detect_people
    runner._detect_people = lambda chunks: (calls.append(len(chunks)) or real_detect(chunks))

    url = "https://youtu.be/" + "a" * 11
    first = runner.run_multi([url], detect_people=True)
    assert first.outputs[0].people, "entities should be detected on a cold run"
    assert len(calls) == 1

    second = runner.run_multi([url], detect_people=True)
    assert second.outputs[0].people == first.outputs[0].people
    assert len(calls) == 1, "a cached video must not be re-scanned for entities"


def test_cached_entry_without_entities_is_topped_up(settings, cache_home):
    """A run with detection off must not poison the cache for a later run."""
    runner = FakeRunner(settings)
    url = "https://youtu.be/" + "a" * 11

    off = runner.run_multi([url], detect_people=False)
    assert off.outputs[0].people == {}

    on = runner.run_multi([url], detect_people=True)
    assert on.outputs[0].people, "entities should be computed for the cached video"
    # Topping up must not have re-fetched the transcript.
    assert runner.fetch_calls.count("a" * 11) == 1


def test_entities_are_per_video_and_zero_based(settings, cache_home):
    """The old batch-wide pass needed re-indexing; this must not regress."""
    runner = FakeRunner(settings)
    result = runner.run_multi(
        ["https://youtu.be/" + c * 11 for c in "ab"], detect_people=True
    )
    for output in result.outputs:
        assert output.people
        assert all(0 <= i < len(output.chunks) for i in output.people)


def test_detection_off_yields_no_entities(settings, cache_home):
    runner = FakeRunner(settings)
    result = runner.run_multi(["https://youtu.be/" + "a" * 11], detect_people=False)
    assert result.outputs[0].people == {}


def test_cache_version_bumped_for_the_people_key():
    """v2 entries have no `people`; reading one as v3 must miss, not KeyError."""
    from panekmodel2.pipeline import CACHE_ENTRY_KEYS, CACHE_VERSION

    assert CACHE_VERSION >= 3
    assert "people" in CACHE_ENTRY_KEYS


def test_v2_shaped_entry_is_discarded(tmp_path):
    """Belt and braces: a stale entry lacking `people` is not handed back."""
    cache = VideoCache(cache_dir=tmp_path)
    v2_entry = {k: [] for k in CACHE_ENTRY_KEYS - {"people"}}
    v2_entry["video_id"] = "vid"
    cache.save("vid", "key", v2_entry)
    assert cache.load("vid", "key") is None
