"""SEC-003: an API key must never reach a log sink.

What matters is the *final rendered string*, not that a redaction helper
exists. These tests drive the real log paths and assert on captured output.
"""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager

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


@contextmanager
def attached(logger_name: str, level: int = logging.INFO):
    """Attach a capturing handler, then put the logger back exactly as found.

    Restoring the LEVEL matters as much as removing the handler: leaving a
    package logger at ERROR silently suppresses warnings that other tests
    assert on, which is a test-pollution bug rather than a product one.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(logger_name)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def capture(logger_name: str, message: str, *args) -> str:
    """Log through the real logging machinery and return the rendered text."""
    with attached(logger_name) as stream:
        logging.getLogger(logger_name).warning(message, *args)
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


# ── D-1: a pre-existing record factory must survive install() ───────
def test_install_delegates_to_a_pre_existing_factory():
    """structlog / OpenTelemetry / correlation-id setups install one first.

    Constructing a bare LogRecord silently discarded whatever they had done.
    """
    from panekmodel2 import logging_redaction

    logging_redaction.uninstall()
    previous = logging.getLogRecordFactory()
    try:
        def third_party(*args, **kwargs):
            record = previous(*args, **kwargs)
            record.correlation_id = "corr-123"
            return record

        logging.setLogRecordFactory(third_party)
        logging_redaction.install()

        for name in ("panekmodel2.pipeline", "somelib.http"):
            record = logging.getLogger(name).makeRecord(
                name, logging.WARNING, "f", 1, "hello", (), None
            )
            assert record.correlation_id == "corr-123", f"{name} lost the third-party stamp"
    finally:
        logging_redaction.uninstall()
        logging.setLogRecordFactory(previous)
        logging_redaction.install()


def test_package_records_keep_the_third_party_class_behaviour():
    """Re-classing must mix redaction in, not replace what the factory built."""
    from panekmodel2 import logging_redaction

    logging_redaction.uninstall()
    previous = logging.getLogRecordFactory()
    try:
        class CustomRecord(logging.LogRecord):
            def custom_marker(self):
                return "custom"

        logging.setLogRecordFactory(
            lambda *a, **k: CustomRecord(*a, **k)
        )
        logging_redaction.install()

        record = logging.getLogger("panekmodel2.pipeline").makeRecord(
            "panekmodel2.pipeline", logging.WARNING, "f", 1, "?key=%s", (KEY,), None
        )
        assert record.custom_marker() == "custom", "the custom class was thrown away"
        assert KEY not in record.getMessage(), "redaction was not mixed in"
    finally:
        logging_redaction.uninstall()
        logging.setLogRecordFactory(previous)
        logging_redaction.install()


# ── D-2: exc_info tracebacks are a separate rendering path ──────────
def test_traceback_text_is_redacted():
    """Formatter.format renders exc_info separately from getMessage()."""
    with attached("panekmodel2.server.jobs", logging.ERROR) as stream:
        try:
            raise RuntimeError(f"HttpError 403 {REQUEST_URL}")
        except RuntimeError:
            logging.getLogger("panekmodel2.server.jobs").exception("run failed")

    out = stream.getvalue()
    assert "Traceback" in out, "the traceback must still be rendered"
    assert KEY not in out, "the API key leaked through the traceback"
    assert "key=REDACTED" in out
    assert "RuntimeError" in out, "redaction must not eat the exception type"


def test_stack_info_is_redacted():
    with attached("panekmodel2.pipeline", logging.ERROR) as stream:
        logging.getLogger("panekmodel2.pipeline").error(
            "boom %s", REQUEST_URL, stack_info=True
        )

    out = stream.getvalue()
    assert "Stack (most recent call last)" in out
    assert KEY not in out


def test_traceback_redaction_does_not_disturb_other_packages():
    with attached("somelib.worker", logging.ERROR) as stream:
        try:
            raise RuntimeError(f"HttpError 403 {REQUEST_URL}")
        except RuntimeError:
            logging.getLogger("somelib.worker").exception("their failure")

    assert KEY in stream.getvalue()
