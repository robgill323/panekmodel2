"""Scrub credentials out of this package's log records.

Google API clients stringify errors to the full request URL, which carries
``key=<API key>``. Redacting at each ``logger.warning(..., exc)`` call site
works until someone adds the next one — and there are already several — so the
scrubbing happens at the point a record renders itself instead. Any log call
added anywhere in this package is covered the moment it is written.

Three things this deliberately does NOT do, each learned by trying it and
watching the test fail:

* It is not a ``logging.Filter`` on the package logger. Filters attached to a
  logger run only for records logged *through that logger*; records from child
  loggers such as ``panekmodel2.pipeline`` reach the parent's handlers without
  ever consulting the parent's filters, so a key logged by any module would
  have sailed straight through.
* It does not redact ``record.msg``. That is the format string — rewriting
  ``"failed: %s"`` mangles the placeholder and raises during formatting.
  Redaction has to happen *after* interpolation.
* It does not construct a bare ``LogRecord``. Doing so silently discarded any
  record factory installed earlier by structlog, OpenTelemetry or a
  correlation-ID setup. The factory here delegates to whatever it replaced and
  mixes redaction into the class that factory returned.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Type

# Query parameters whose values are credentials, matched case-insensitively.
import re

SECRET_PARAMS = (
    "key", "developerkey", "access_token", "refresh_token", "token",
    "api_key", "apikey", "password", "client_secret",
)

_SECRET_RE = re.compile(r"(?i)\b(" + "|".join(SECRET_PARAMS) + r")=([^&\s\"'<>\\)\]}]+)")

REDACTED = "REDACTED"

PACKAGE = "panekmodel2"

# Used only to render exception text exactly as the stdlib would, so that
# redacting it cannot change the shape of a traceback.
_RENDERER = logging.Formatter()


def redact_secrets(value: object) -> str:
    """Strip credential-bearing query parameters out of *value*'s text."""
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", str(value))


class RedactingMixin:
    """Redacts a record's rendered message.

    A mixin rather than a ``LogRecord`` subclass so it can be combined with
    whatever record class another library's factory already produces.
    """

    def getMessage(self) -> str:  # type: ignore[override]
        return redact_secrets(super().getMessage())  # type: ignore[misc]


def redact_exception_text(record: logging.LogRecord) -> None:
    """Pre-render and redact a record's traceback.

    ``getMessage()`` covers the interpolated message only; ``Formatter.format``
    renders ``exc_info`` separately through ``formatException()``, so a
    traceback whose final line embeds the request URL would print the key
    verbatim. Populating ``exc_text`` here means the formatter reuses this
    redacted rendering instead of producing its own.
    """
    if record.exc_info and not record.exc_text:
        record.exc_text = redact_secrets(_RENDERER.formatException(record.exc_info))
    if record.stack_info:
        record.stack_info = redact_secrets(record.stack_info)


# Redacting variants are built once per underlying record class.
_redacting_classes: Dict[type, type] = {}


def _redacting_class(base: type) -> type:
    cls = _redacting_classes.get(base)
    if cls is None:
        cls = type(f"Redacting{base.__name__}", (RedactingMixin, base), {})
        _redacting_classes[base] = cls
    return cls


# Kept for the tests that assert on the plain case and for uninstall().
RedactingLogRecord: Type[logging.LogRecord] = _redacting_class(logging.LogRecord)  # type: ignore[assignment]

_original_factory: Optional[Callable] = None


def install(package: str = PACKAGE) -> None:
    """Route this package's log records through the redacting variant.

    Idempotent. Records from every other library keep both their content and
    the factory that built them.
    """
    global _original_factory
    if _original_factory is not None:
        return
    previous = logging.getLogRecordFactory()
    _original_factory = previous
    prefix = package + "."

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        name = getattr(record, "name", "")
        if name == package or name.startswith(prefix):
            record.__class__ = _redacting_class(type(record))
            redact_exception_text(record)
        return record

    logging.setLogRecordFactory(factory)


def uninstall() -> None:
    """Restore the previous record factory. Used by tests."""
    global _original_factory
    if _original_factory is not None:
        logging.setLogRecordFactory(_original_factory)
        _original_factory = None
