"""Does topic modelling survive a very short video?

A YouTube Short can yield 1–3 chunks. The audit flagged this as an unresolved
risk: if UMAP/HDBSCAN raise at tiny n, a single Short would kill the whole
batch, because the shared model is fitted over every video at once.

These tests run a real BERTopic fit at n ∈ {1, 2, 3, 5}. They are marked slow
because they load ``all-MiniLM-L6-v2`` (~90 MB, cached after first run).
Run them with: pytest -m slow
"""

from __future__ import annotations

import pytest

from panekmodel2.chunker import Chunk
from panekmodel2.topic_model import TopicModeler, mark_reassigned

SENTENCES = [
    "The school board approved the levy after a long budget hearing this evening.",
    "Traffic on the bridge was closed again while crews repaired the north span.",
    "Grocery prices are still climbing and wages have not kept pace this year.",
    "The clinic reported a rise in appointments following the new guidance.",
    "Rain flooded the low-lying roads and knocked out power for the county.",
]


def chunks_for(n: int) -> list[Chunk]:
    return [
        Chunk(text=SENTENCES[i % len(SENTENCES)], start=i * 30.0, end=(i + 1) * 30.0, source_indices=[i])
        for i in range(n)
    ]


@pytest.mark.slow
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_fit_survives_tiny_corpora(n):
    """A short video must never raise — the whole batch depends on this fit."""
    modeler = TopicModeler(embedding_model="all-MiniLM-L6-v2", reduce_to=0)
    chunks = chunks_for(n)
    embeddings = modeler.embed_chunks(chunks)

    _model, topics, probs = modeler.fit(chunks, embeddings=embeddings)

    assert len(topics) == n, "every chunk must receive a topic assignment"
    assert len(probs) == n
    assert all(isinstance(int(t), int) for t in topics)
    # -1 (outlier bin) is a legitimate outcome at tiny n; a crash is not.
    assert all(int(t) >= -1 for t in topics)

    df = modeler.topic_dataframe(chunks, topics, probs)
    assert list(df["chunk_index"]) == list(range(n))

    if n < 3:
        # Too small to cluster: exactly one topic, and it still has real
        # keywords derived from the text rather than an invented structure.
        assert set(topics) == {0}
        assert modeler.model.get_topic(0)


@pytest.mark.slow
def test_fit_rejects_an_empty_corpus():
    modeler = TopicModeler(embedding_model="all-MiniLM-L6-v2", reduce_to=0)
    with pytest.raises(ValueError, match="No chunks"):
        modeler.fit([], embeddings=None)


def test_empty_corpus_raises_before_touching_a_model():
    """The n=0 guard must not require the embedder to be loaded."""
    modeler = TopicModeler(embedding_model="definitely-not-a-model", reduce_to=0)
    with pytest.raises(ValueError, match="No chunks"):
        modeler.fit([], embeddings=None)


def test_mark_reassigned_flags_only_chunks_moved_out_of_the_outlier_bin():
    before = [-1, -1, 0, 1, -1]
    after = [2, -1, 0, 1, 3]
    assert mark_reassigned(before, after) == [True, False, False, False, True]


def test_mark_reassigned_ignores_chunks_that_were_never_outliers():
    """A normal chunk whose topic merely changed id is not "reassigned"."""
    assert mark_reassigned([0, 1], [1, 0]) == [False, False]


def test_mark_reassigned_on_an_untouched_batch():
    assert mark_reassigned([0, 1, 2], [0, 1, 2]) == [False, False, False]


def test_mark_reassigned_empty():
    assert mark_reassigned([], []) == []


@pytest.mark.slow
def test_reassignment_flag_is_populated_and_consistent():
    """The flag must always be per-chunk, boolean, and never mark an outlier.

    Whether HDBSCAN actually produces outliers on a given corpus is not
    something a test can force, so this asserts the invariants that must hold
    either way; mark_reassigned() above covers the moved-chunk logic directly.
    """
    modeler = TopicModeler(embedding_model="all-MiniLM-L6-v2", reduce_to=0)
    chunks = chunks_for(24)
    embeddings = modeler.embed_chunks(chunks)

    _model, topics, probs = modeler.fit(chunks, embeddings=embeddings)

    assert len(modeler.reassigned) == len(chunks)
    assert all(isinstance(flag, bool) for flag in modeler.reassigned)
    for flag, topic in zip(modeler.reassigned, topics):
        if flag:
            assert int(topic) != -1, "a flagged chunk was moved OUT of the bin"

    df = modeler.topic_dataframe(chunks, topics, probs)
    assert "reassigned" in df.columns
    assert list(df["reassigned"]) == list(modeler.reassigned)


@pytest.mark.slow
def test_reassigned_flag_resets_between_fits():
    modeler = TopicModeler(embedding_model="all-MiniLM-L6-v2", reduce_to=0)
    chunks = chunks_for(24)
    modeler.fit(chunks, embeddings=modeler.embed_chunks(chunks))

    small = chunks_for(2)
    modeler.fit(small, embeddings=modeler.embed_chunks(small))
    assert modeler.reassigned == [False, False]
