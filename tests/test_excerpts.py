"""Representative-excerpt selection and the stale-probability handling it rests on.

BERTopic's ``reduce_outliers`` rewrites topic assignments but leaves
``probabilities_`` untouched, so a chunk moved out of the outlier bin keeps a
probability describing a topic it is no longer in. Ranking "most
representative" by that number would be ranking by noise.
"""

from __future__ import annotations

import pytest

from panekmodel2.server.results import (
    polarized_excerpts,
    representative_excerpts,
)


def excerpt(prob, valence, start=0.0, reassigned=False):
    return {
        "text": f"chunk at {start}",
        "video_id": "v",
        "video_title": "V",
        "channel": "C",
        "start": start,
        "valence": valence,
        "topic_prob": prob,
        "topic_reassigned": reassigned,
    }


def test_representative_ranks_by_assignment_probability():
    pool = [
        excerpt(0.42, 0.9, start=0),
        excerpt(0.98, 0.05, start=1),
        excerpt(0.71, -0.8, start=2),
    ]
    picked = representative_excerpts(pool)
    assert [e["topic_prob"] for e in picked] == [0.98, 0.71, 0.42]


def test_representative_does_not_rank_by_sentiment():
    """The mock ranked by |valence|; the real thing must not."""
    pool = [excerpt(0.99, 0.01, start=0), excerpt(0.10, -0.99, start=1)]
    assert representative_excerpts(pool)[0]["topic_prob"] == 0.99


def test_reassigned_chunks_rank_last():
    pool = [
        excerpt(None, 0.9, start=0, reassigned=True),
        excerpt(0.30, 0.1, start=1),
    ]
    picked = representative_excerpts(pool)
    assert picked[0]["topic_prob"] == 0.30
    assert picked[1]["topic_reassigned"] is True


def test_reassigned_chunks_are_still_shown_when_that_is_all_there_is():
    """A topic built entirely from reassigned chunks must not show zero quotes."""
    pool = [excerpt(None, 0.5, start=i, reassigned=True) for i in range(3)]
    picked = representative_excerpts(pool)
    assert len(picked) == 3


def test_representative_is_deterministic_on_ties():
    pool = [excerpt(0.5, 0.1, start=s) for s in (30.0, 10.0, 20.0)]
    assert [e["start"] for e in representative_excerpts(pool)] == [10.0, 20.0, 30.0]


def test_polarized_ranks_by_absolute_valence():
    pool = [excerpt(0.9, 0.10, start=0), excerpt(0.2, -0.95, start=1), excerpt(0.5, 0.60, start=2)]
    assert [e["valence"] for e in polarized_excerpts(pool)] == [-0.95, 0.60, 0.10]


def test_both_modes_cap_at_six():
    pool = [excerpt(0.5, 0.5, start=i) for i in range(20)]
    assert len(representative_excerpts(pool)) == 6
    assert len(polarized_excerpts(pool)) == 6


def test_empty_pool():
    assert representative_excerpts([]) == []
    assert polarized_excerpts([]) == []


@pytest.mark.parametrize("selector", [representative_excerpts, polarized_excerpts])
def test_selectors_do_not_mutate_the_pool(selector):
    pool = [excerpt(0.5, -0.9, start=1), excerpt(0.9, 0.1, start=0)]
    before = list(pool)
    selector(pool)
    assert pool == before
