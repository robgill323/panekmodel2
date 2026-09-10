"""Chunk-size settings: seconds are what the UI picks, so seconds must govern."""

from __future__ import annotations

import pytest

from panekmodel2.chunker import chunk_segments, words_for_seconds
from panekmodel2.transcript_fetcher import TranscriptSegment


@pytest.mark.parametrize("seconds", [15, 30, 60, 120])
def test_word_cap_leaves_room_for_the_chosen_duration(seconds):
    """The paired cap must exceed what normal speech fits in that window."""
    spoken_words = seconds * 2.6
    assert words_for_seconds(seconds) > spoken_words


def test_word_cap_grows_with_duration():
    caps = [words_for_seconds(s) for s in (15, 30, 60, 120)]
    assert caps == sorted(caps)
    assert len(set(caps)) == len(caps)


def test_word_cap_has_a_floor_for_tiny_windows():
    assert words_for_seconds(1) >= 40


def _segments(n, seconds_each=5.0, words=12):
    return [
        TranscriptSegment(text=" ".join(["word"] * words), start=i * seconds_each, duration=seconds_each)
        for i in range(n)
    ]


def test_duration_bound_splits_chunks():
    chunks = chunk_segments(_segments(12), max_words=words_for_seconds(30), max_seconds=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.end - chunk.start <= 40, "chunks should track the 30 s setting"


def test_longer_setting_yields_fewer_chunks():
    segments = _segments(48)
    short = chunk_segments(segments, max_words=words_for_seconds(15), max_seconds=15)
    long = chunk_segments(segments, max_words=words_for_seconds(120), max_seconds=120)
    assert len(short) > len(long)


def test_no_transcript_text_is_dropped():
    segments = _segments(10)
    chunks = chunk_segments(segments, max_words=words_for_seconds(30), max_seconds=30)
    assert sum(len(c.text.split()) for c in chunks) == sum(len(s.text.split()) for s in segments)


def test_empty_input():
    assert chunk_segments([], max_words=100, max_seconds=30) == []
