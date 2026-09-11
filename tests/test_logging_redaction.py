"""SEC-003: an API key must never reach a log sink.

What matters is the *final rendered string*, not that a redaction helper
exists. These tests drive the real log paths and assert on captured output.
"""

from __future__ import annotations

import io
import logging

import pytest

from panekmodel2.logging_redaction import (
    PACKAGE,
    RedactingLogRecord,
    redact_secrets,
)

KEY = "AIzaSyFAKE_LIVE_KEY_123"
REQUEST_URL = (
    "https://youtube.googleapis.com/youtube/v3/videos"
    f"?part=snippet&id=abc&key={KEY}"
)


def capture(logger_name: str, message: str, *args) -> str:
    """Log through the real logging machinery and return the rendered text."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.warning(message, *args)
    finally:
        logger.removeHandler(handler)
    return stream.getvalue()


@pytest.mark.parametrize(
    "logger_name",
    [
        "panekmodel2",
        "panekmodel2.pipeline",
        "panekmodel2.transcript_fetcher",
        "panekmodel2.server.jobs",
    ],
)
def test_api_key_never_reaches_the_log(logger_name):
    """Child loggers matter most: a filter on the parent would miss them."""
    exc = RuntimeError(f"<HttpError 403 when requesting {REQUEST_URL} returned quotaExceeded>")
    out = capture(logger_name, "metadata fetch failed: %s", exc)

    assert KEY not in out
    assert "key=REDACTED" in out
    # The rest of the message must survive — a redaction that eats the error
    # is not an improvement.
    assert "quotaExceeded" in out
    assert "part=snippet" in out


def test_multi_argument_log_lines_still_format():
    """Redacting the format string would break %s interpolation."""
    exc = RuntimeError(f"boom {REQUEST_URL}")
    out = capture("panekmodel2.pipeline", "Skipping %s due to error: %s", "vid123", exc)

    assert "vid123" in out
    assert KEY not in out
    assert "Traceback" not in out, "the record must render, not raise"


def test_other_libraries_are_left_alone():
    """Scrubbing is scoped to this package, not installed globally."""
    out = capture("somelib.http", "requesting %s", REQUEST_URL)
    assert KEY in out


@pytest.mark.parametrize("param", ["key", "KEY", "developerKey", "access_token", "api_key", "client_secret"])
def test_every_credential_parameter_is_covered(param):
    assert KEY not in redact_secrets(f"https://x/y?{param}={KEY}&z=1")


def test_redaction_keeps_surrounding_query_parameters():
    cleaned = redact_secrets(f"?part=snippet&key={KEY}&id=abc")
    assert cleaned == "?part=snippet&key=REDACTED&id=abc"


@pytest.mark.parametrize("text", ["the monkey=wrench fell", "no secrets here", "a=b", ""])
def test_innocent_text_is_untouched(text):
    assert redact_secrets(text) == text


def test_record_class_is_used_for_package_records():
    record = logging.getLogger("panekmodel2.pipeline").makeRecord(
        "panekmodel2.pipeline", logging.WARNING, "f", 1, "?key=%s", (KEY,), None
    )
    assert isinstance(record, RedactingLogRecord)
    assert KEY not in record.getMessage()


def test_record_class_is_not_used_for_other_packages():
    record = logging.getLogger("somelib").makeRecord(
        "somelib", logging.WARNING, "f", 1, "?key=%s", (KEY,), None
    )
    assert not isinstance(record, RedactingLogRecord)


def test_package_constant_matches_the_real_package():
    import panekmodel2

    assert panekmodel2.__name__ == PACKAGE
