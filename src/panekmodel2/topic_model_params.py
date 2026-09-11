"""Topic-granularity parameters.

Split out of topic_model.py so config.py can validate a granularity value
without importing BERTopic — that import pulls torch and costs seconds.

How finely a batch is split into topics. A 13-video trial showed small batches
under-splitting badly — 90-minute homogeneous videos collapsing to a single
topic — so this is the knob a researcher needs before pointing the tool at a
real corpus.

The value is HDBSCAN's ``min_cluster_size``: the smallest number of chunks that
may form a topic. **Smaller means finer** — more, narrower topics — and larger
means coarser.

Each level is its own curve rather than a multiplier on a shared baseline. The
multiplier design it replaced was broken in a way the arithmetic hid: the
standard baseline caps at 5, ``5 * 0.5`` banker-rounds to 2, and 2 is the
floor — so "fine" was the constant 2 at every corpus size from 1 to 5000,
scaling with nothing, and identical to "standard" for corpora of 10 chunks or
fewer. Per-level curves make each level's behaviour readable at a glance and
impossible to collapse by accident.

``standard`` is byte-identical to the behaviour that predates this knob, which
is load-bearing: existing runs must not shift because an option was added.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

# Each level: the value for tiny and small corpora, then a ceiling and a
# divisor for everything larger (``n // divisor``, clamped to [mid, ceiling]).
#
# standard's numbers reproduce the original function exactly:
#   n <= 10 -> 2;  n <= 50 -> 3;  else max(3, min(5, n // 40))
GRANULARITY_LEVELS: Dict[str, Dict[str, int]] = {
    "coarse": {"tiny": 4, "small": 6, "ceiling": 10, "divisor": 20},
    "standard": {"tiny": 2, "small": 3, "ceiling": 5, "divisor": 40},
    "fine": {"tiny": 2, "small": 2, "ceiling": 3, "divisor": 80},
}

DEFAULT_GRANULARITY = "standard"

# HDBSCAN's own floor: fewer than two chunks is not a cluster.
MIN_CLUSTER_FLOOR = 2

TINY_CORPUS = 10
SMALL_CORPUS = 50

# Shown in the UI so the choice is legible without reading the source.
GRANULARITY_DESCRIPTIONS = {
    "coarse": "Fewer, broader topics. Good for asking what a batch is broadly about.",
    "standard": "The balanced default.",
    "fine": "More, narrower topics. Use when one long video collapses into a single topic.",
}

GRANULARITY_ORDER: List[str] = ["coarse", "standard", "fine"]


def validate_granularity(granularity: object) -> str:
    """Return *granularity* if known, else raise. Used at startup and on input."""
    if granularity in GRANULARITY_LEVELS:
        return str(granularity)
    raise ValueError(
        f"Unknown topic granularity {granularity!r}; expected one of "
        f"{sorted(GRANULARITY_LEVELS)}."
    )


def min_cluster_size_for(n_samples: int, granularity: str = DEFAULT_GRANULARITY) -> int:
    """HDBSCAN ``min_cluster_size`` for *n_samples* chunks at *granularity*.

    Guarantees, each pinned by a test:

    * ``standard`` reproduces the pre-knob behaviour exactly.
    * ``fine`` is never coarser than ``standard``, and is strictly finer
      wherever the floor allows it — i.e. whenever ``standard`` exceeds 2.
      At or below 10 chunks ``standard`` is already 2, so nothing can be finer.
    * ``coarse`` is never finer than ``standard``, and is strictly coarser
      wherever the corpus allows it — i.e. for more than 2 chunks. At 2 chunks
      the cap and the floor meet.
    * The result is always at least 2 and never more than the batch holds.
      A minimum above ``n_samples`` makes HDBSCAN raise, which is how a single
      Short used to kill a whole batch.
    """
    level = GRANULARITY_LEVELS[validate_granularity(granularity)]

    if n_samples <= TINY_CORPUS:
        size = level["tiny"]
    elif n_samples <= SMALL_CORPUS:
        size = level["small"]
    else:
        size = max(level["small"], min(level["ceiling"], n_samples // level["divisor"]))

    # Floor first, then cap to the corpus: a batch of 3 cannot support a
    # minimum of 4, and a minimum below 2 means nothing to HDBSCAN.
    return max(MIN_CLUSTER_FLOOR, min(size, max(MIN_CLUSTER_FLOOR, n_samples)))


def granularity_table(sizes: Sequence[int]) -> Dict[int, Dict[str, int]]:
    """The computed minimum for each level across *sizes*. For tests and docs."""
    return {
        n: {level: min_cluster_size_for(n, level) for level in GRANULARITY_ORDER}
        for n in sizes
    }
