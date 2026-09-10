"""Shared fixtures.

Every test in the default suite runs without downloading model weights: the
embedding and sentiment stages are stubbed. Tests that genuinely need real
model behaviour are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panekmodel2 import pipeline as pipeline_mod
from panekmodel2.chunker import Chunk
from panekmodel2.config import Settings
from panekmodel2.pipeline import PipelineRunner
from panekmodel2.sentiment import SentimentResult
from panekmodel2.transcript_fetcher import TranscriptSegment


@pytest.fixture
def settings() -> Settings:
    return Settings(
        chunk_max_words=50,
        chunk_max_seconds=30,
        embedding_model="test-embed",
        sentiment_model="test-sentiment",
        topic_reduce_to=0,
    )


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    """Point VideoCache at a temp directory instead of the real ~/."""
    monkeypatch.setattr(pipeline_mod.Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path / ".panekmodel2_cache"


def make_segments(n: int = 6, video: str = "a") -> list[TranscriptSegment]:
    return [
        TranscriptSegment(text=f"{video} sentence number {i} about budgets and schools", start=i * 10.0, duration=10.0)
        for i in range(n)
    ]


def make_chunks(n: int = 3, video: str = "a") -> list[Chunk]:
    return [
        Chunk(text=f"{video} chunk {i} about budgets and schools", start=i * 30.0, end=(i + 1) * 30.0, source_indices=[i])
        for i in range(n)
    ]


def make_sentiments(n: int, label: str = "positive", score: float = 0.9) -> list[SentimentResult]:
    return [SentimentResult(label=label, score=score) for _ in range(n)]


class FakeRunner(PipelineRunner):
    """PipelineRunner with the three expensive stages replaced.

    Fetch/embed/sentiment are deterministic stubs; the topic model assigns
    chunks round-robin to two topics. Everything else — caching, batching,
    per-URL outcome bookkeeping, index rebasing — is the real code.
    """

    def __init__(self, settings: Settings, chunks_per_video: int = 3, fail_ids: set | None = None):
        super().__init__(settings)
        self.chunks_per_video = chunks_per_video
        self.fail_ids = fail_ids or set()
        self.embed_calls = 0
        self.fetch_calls: list[str] = []

        def _fetch(video_id: str, prefer_official: bool = True):
            self.fetch_calls.append(video_id)
            if video_id in self.fail_ids:
                raise RuntimeError("No transcript available for this video")
            return make_segments(self.chunks_per_video * 2, video_id)

        self.fetcher.fetch = _fetch  # type: ignore[method-assign]

        def _embed(chunks):
            self.embed_calls += 1
            return np.ones((len(chunks), 4), dtype="float32")

        self.topic_modeler.embed_chunks = _embed  # type: ignore[method-assign]

        def _fit(chunks, embeddings=None):
            # Round-robin two topics, with every third chunk an outlier that
            # "reduce_outliers" then moves into topic 0 — so the reassigned
            # path is exercised, along with a spread of probabilities.
            topics, probs, reassigned = [], [], []
            for i in range(len(chunks)):
                if i % 3 == 2:
                    topics.append(0)
                    probs.append(0.0)
                    reassigned.append(True)
                else:
                    topics.append(i % 2)
                    probs.append(0.95 if i % 2 == 0 else 0.35)
                    reassigned.append(False)
            self.topic_modeler.reassigned = reassigned
            return object(), topics, probs

        self.topic_modeler.fit = _fit  # type: ignore[method-assign]

        def _topic_dataframe(chunks, topics, probs):
            reassigned = self.topic_modeler.reassigned or [False] * len(chunks)
            return pd.DataFrame(
                [
                    {
                        "chunk_index": i,
                        "topic": t,
                        "prob": p,
                        "reassigned": reassigned[i],
                        "start": c.start,
                        "end": c.end,
                        "text": c.text,
                    }
                    for i, (c, t, p) in enumerate(zip(chunks, topics, probs))
                ]
            )

        self.topic_modeler.topic_dataframe = _topic_dataframe  # type: ignore[method-assign]

        def _analyze(chunks):
            # Alternate polarity so valences are not degenerate.
            return [
                SentimentResult(label="positive" if i % 2 == 0 else "negative", score=0.8)
                for i in range(len(chunks))
            ]

        self.sentiment_analyzer.analyze = _analyze  # type: ignore[method-assign]

    def fetch_metadata(self, video_id: str):
        return {"title": f"Video {video_id}", "channel": "Test Channel", "published": "2026-01-01"}

    def topic_keywords(self, top_n: int = 10):
        return {0: ["budget", "school", "levy"], 1: ["road", "bridge", "transit"]}

    @staticmethod
    def _detect_people(chunks):
        return {0: ["Ada Lovelace"]}


@pytest.fixture
def fake_runner(settings, cache_home):
    return FakeRunner(settings)
