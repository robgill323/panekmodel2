"""panekmodel2: transcript ingestion, topic modeling, and sentiment."""

from . import logging_redaction

# Credentials must never reach a log sink, so this is armed at import time
# rather than left to whichever entry point happens to configure logging.
logging_redaction.install()

__all__ = [
    "config",
    "transcript_fetcher",
    "chunker",
    "topic_model",
    "sentiment",
    "pipeline",
    "logging_redaction",
]
