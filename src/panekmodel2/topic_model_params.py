"""Topic-granularity parameters.

Split out of topic_model.py so config.py can validate a granularity value
without importing BERTopic — that import pulls torch and costs seconds.
"""

from __future__ import annotations

# How finely to split the batch into topics. A 13-video trial showed small
# batches under-splitting badly — 90-minute homogeneous videos collapsing to a
# single topic — so this is the knob a researcher needs before pointing the
# tool at a real corpus.
#
# The value scales HDBSCAN's min_cluster_size: a larger minimum cluster means
# fewer, broader topics, a smaller one means more, narrower topics. It is a
# multiplier on the corpus-size baseline rather than an absolute, so it
# composes with batch size instead of fighting it.
GRANULARITY_FACTORS = {
    "coarse": 2.0,
    "standard": 1.0,
    "fine": 0.5,
}
DEFAULT_GRANULARITY = "standard"

# Shown in the UI so the choice is legible without reading the source.
GRANULARITY_DESCRIPTIONS = {
    "coarse": "Fewer, broader topics. Good for asking what a batch is broadly about.",
    "standard": "The balanced default.",
    "fine": "More, narrower topics. Use when one long video collapses into a single topic.",
}


def granularity_factor(granularity: str) -> float:
    """Multiplier for min_cluster_size. Raises on an unknown name."""
    try:
        return GRANULARITY_FACTORS[granularity]
    except KeyError:
        raise ValueError(
            f"Unknown topic granularity {granularity!r}; expected one of "
            f"{sorted(GRANULARITY_FACTORS)}."
        ) from None


def min_cluster_size_for(n_samples: int, granularity: str = DEFAULT_GRANULARITY) -> int:
    """HDBSCAN min_cluster_size for a corpus of *n_samples* at *granularity*.

    The corpus-size baseline is unchanged at "standard", so existing runs keep
    their behaviour; granularity scales it. Never returns less than 2, which is
    HDBSCAN's own floor for a meaningful cluster.
    """
    if n_samples <= 10:
        baseline = 2
    elif n_samples <= 50:
        baseline = 3
    else:
        baseline = max(3, min(5, n_samples // 40))
    scaled = round(baseline * granularity_factor(granularity))
    # Never exceed what the corpus can actually support: a min_cluster_size
    # above n_samples makes HDBSCAN raise, which is how Shorts used to kill a
    # whole batch.
    return max(2, min(int(scaled), max(2, n_samples)))
