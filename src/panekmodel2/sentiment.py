from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .chunker import Chunk

logger = logging.getLogger(__name__)

# Single source of truth for turning a classifier label into a signed valence
# multiplier.  Every consumer — rollups, CSV export, the Streamlit UI and the
# HTTP API — must go through :func:`normalize_sentiment` rather than
# re-deriving the sign from the label text, which is how the export and the
# on-screen metrics came to disagree.
#
# Deliberately does NOT map opaque labels such as ``LABEL_0``/``LABEL_1``:
# their polarity is model-specific, so guessing would silently invert a whole
# run.  Unknown labels raise instead.
LABEL_MAP: Dict[str, float] = {
    "positive": 1.0,
    "pos": 1.0,
    "negative": -1.0,
    "neg": -1.0,
    "neutral": 0.0,
    "neu": 0.0,
}

# Labels that map to exactly 0.0 valence regardless of model confidence.
NEUTRAL_LABELS = frozenset(k for k, v in LABEL_MAP.items() if v == 0.0)


class UnmappableSentimentLabel(ValueError):
    """Raised when a model emits labels this project cannot interpret."""


@dataclass
class SentimentResult:
    label: str
    score: float


def normalize_sentiment(label: str, score: float) -> float:
    """Convert a (label, confidence) pair into a valence in −1 … +1.

    ``neutral`` maps to 0.0 no matter how confident the model is: a neutral
    chunk is not a negative one.  Raises :class:`UnmappableSentimentLabel` for
    labels outside :data:`LABEL_MAP` so an unknown model fails loudly instead
    of rendering a fabricated all-neutral result.
    """
    key = (label or "").strip().lower()
    if key not in LABEL_MAP:
        raise UnmappableSentimentLabel(
            f"Sentiment label {label!r} is not in the known label set "
            f"({sorted(LABEL_MAP)}). The configured sentiment model emits "
            "labels this project cannot interpret as a valence."
        )
    return LABEL_MAP[key] * float(score)


def normalize_all(sentiments: Sequence[SentimentResult]) -> List[float]:
    """Vectorized :func:`normalize_sentiment` over a sequence of results."""
    return [normalize_sentiment(s.label, s.score) for s in sentiments]


def validate_labels(labels: Iterable[str]) -> None:
    """Raise if any label in *labels* cannot be mapped to a valence."""
    unknown = sorted({(l or "").strip().lower() for l in labels} - set(LABEL_MAP))
    if unknown:
        raise UnmappableSentimentLabel(
            f"Sentiment model emitted unmappable labels: {unknown}. "
            f"Known labels: {sorted(LABEL_MAP)}."
        )


def produces_neutral(labels: Iterable[str]) -> bool:
    """True when the observed label set actually contains a neutral class.

    Binary models (the default ``siebert/…`` is one) never emit neutral, so a
    UI must not imply that a chunk near 0.0 was scored neutral by the model.
    """
    return bool({(l or "").strip().lower() for l in labels} & NEUTRAL_LABELS)


class SentimentAnalyzer:
    """Chunk-level sentiment scoring.

    The transformer pipeline is built lazily on first use so that importing
    this module — or constructing a :class:`~panekmodel2.pipeline.PipelineRunner`
    for a transcript-only command — does not pull ~1.4 GB of weights.
    """

    def __init__(self, model_name: str, batch_size: int = 16, use_cuda: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = 0 if use_cuda else -1
        self._pipe = None
        self.label_map = LABEL_MAP

    @property
    def pipe(self):
        if self._pipe is None:
            from transformers import logging as hf_logging  # noqa: PLC0415
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415

            # Disable HF/tqdm progress bars during weight loading to avoid
            # BrokenPipeError when stderr is redirected (e.g. inside Streamlit).
            os.environ.setdefault("TQDM_DISABLE", "1")
            hf_logging.disable_progress_bar()
            logger.info("Loading sentiment model: %s", self.model_name)
            self._pipe = hf_pipeline("sentiment-analysis", model=self.model_name, device=self.device)
        return self._pipe

    def analyze(self, chunks: Sequence[Chunk]) -> List[SentimentResult]:
        texts = [c.text for c in chunks]
        if not texts:
            return []
        logger.info("Running sentiment on %d chunks", len(texts))
        outputs = self.pipe(texts, batch_size=self.batch_size, truncation=True)
        # HF pipeline returns list[dict] for default top-1
        results = [
            SentimentResult(label=out["label"].lower(), score=float(out["score"]))
            for out in outputs
        ]
        validate_labels(r.label for r in results)
        return results

    def aggregate(self, sentiments: Sequence[SentimentResult]) -> dict:
        numeric = normalize_all(sentiments)
        mean = float(np.mean(numeric)) if numeric else 0.0
        median = float(np.median(numeric)) if numeric else 0.0
        sd = float(np.std(numeric)) if numeric else 0.0
        counts: Dict[str, int] = {}
        for s in sentiments:
            counts[s.label] = counts.get(s.label, 0) + 1
        total = len(sentiments) or 1
        fractions = {k: v / total for k, v in counts.items()}
        return {
            "mean": mean,
            "median": median,
            "sd": sd,
            "counts": counts,
            "fractions": fractions,
        }
