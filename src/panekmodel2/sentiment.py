from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from transformers import logging as hf_logging
from transformers import pipeline as hf_pipeline

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    label: str
    score: float


class SentimentAnalyzer:
    def __init__(self, model_name: str, batch_size: int = 16, use_cuda: bool = False):
        device = 0 if use_cuda else -1
        # Disable HF/tqdm progress bars during weight loading to avoid
        # BrokenPipeError when stderr is redirected (e.g. inside Streamlit).
        os.environ.setdefault("TQDM_DISABLE", "1")
        hf_logging.disable_progress_bar()
        self.pipe = hf_pipeline("sentiment-analysis", model=model_name, device=device)
        self.batch_size = batch_size
        self.label_map = {
            "positive": 1.0,
            "pos": 1.0,
            "negative": -1.0,
            "neg": -1.0,
            "neutral": 0.0,
            "neu": 0.0,
        }

    def analyze(self, chunks: Sequence[Chunk]) -> List[SentimentResult]:
        texts = [c.text for c in chunks]
        logger.info("Running sentiment on %d chunks", len(texts))
        outputs = self.pipe(texts, batch_size=self.batch_size, truncation=True)
        # HF pipeline returns list[dict] for default top-1
        results: List[SentimentResult] = []
        for out in outputs:
            results.append(SentimentResult(label=out["label"].lower(), score=float(out["score"])))
        return results

    def aggregate(self, sentiments: Sequence[SentimentResult]) -> dict:
        numeric = [self.label_map.get(s.label.lower(), 0.0) * s.score for s in sentiments]
        mean = float(np.mean(numeric)) if numeric else 0.0
        median = float(np.median(numeric)) if numeric else 0.0
        counts = {}
        for s in sentiments:
            key = s.label
            counts[key] = counts.get(key, 0) + 1
        total = len(sentiments) or 1
        fractions = {k: v / total for k, v in counts.items()}
        return {"mean": mean, "median": median, "counts": counts, "fractions": fractions}
