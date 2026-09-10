"""Video-ID parsing, cache-shape compatibility, and batch alignment."""

from __future__ import annotations

import pytest

from panekmodel2.pipeline import (
    CACHE_ENTRY_KEYS,
    VideoCache,
    describe_failure,
    extract_video_id,
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
