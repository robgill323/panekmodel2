"""Process-wide PipelineRunner registry.

Model weights are ~1.8 GB. They are loaded once per distinct settings
signature and shared across every request, never rebuilt per request.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict

from ..config import Settings
from ..pipeline import PipelineRunner

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_runners: Dict[str, PipelineRunner] = {}


def _signature(settings: Settings) -> str:
    """Only the fields that change model or chunk behaviour matter here."""
    return "|".join(
        str(getattr(settings, field))
        for field in (
            "embedding_model",
            "sentiment_model",
            "chunk_max_words",
            "chunk_max_seconds",
            "topic_reduce_to",
            "use_whisper_fallback",
            "whisper_model",
            "cuda",
            "sentiment_batch_size",
        )
    ) + "|" + ",".join(sorted(settings.custom_stopwords))


def get_runner(settings: Settings) -> PipelineRunner:
    key = _signature(settings)
    with _lock:
        runner = _runners.get(key)
        if runner is None:
            logger.info("Building PipelineRunner for settings signature %s", key)
            runner = PipelineRunner(settings)
            _runners[key] = runner
        return runner


def reset() -> None:
    """Drop all cached runners. Used by tests."""
    with _lock:
        _runners.clear()
