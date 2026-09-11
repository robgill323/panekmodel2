"""Scrub credentials out of this package's log records.

Google API clients stringify errors to the full request URL, which carries
``key=<API key>``. Redacting at each ``logger.warning(..., exc)`` call site
works until someone adds the next one — and there are already several — so the
scrubbing happens at the point a record renders itself instead.

Two things this deliberately does NOT do, both learned by trying them:

* It is not a ``logging.Filter`` on the package logger. Filters attached to a
  logger run only for records logged *through that logger*; records from child
  loggers such as ``panekmodel2.pipeline`` propagate to the parent's handlers
  without ever consulting the parent's filters, so a key logged by any module
  would have sailed straight through.
* It does not redact ``record.msg``. That is the format string — rewriting
  ``"failed: %s"`` mangles the placeholder and raises a formatting error.
  Redaction has to happen *after* interpolation, which is what overriding
  ``getMessage`` achieves.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

# Query parameters whose values are credentials, matched case-insensitively.
SECRET_PARAMS = (
    "key", "developerkey", "access_token", "refresh_token", "token",
    "api_key", "apikey", "password", "client_secret",
)

_SECRET_RE = re.compile(r"(?i)\b(" + "|".join(SECRET_PARAMS) + r")=([^&\s\"'<>\\)\]}]+)")

REDACTED = "REDACTED"

PACKAGE = "panekmodel2"


def redact_secrets(value: object) -> str:
    """Strip credential-bearing query parameters out of *value*'s text."""
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", str(value))


class RedactingLogRecord(logging.LogRecord):
    """A record that redacts credentials from its own rendered message."""

    def getMessage(self) -> str:
        return redact_secrets(super().getMessage())


_original_factory: Optional[object] = None


def install(package: str = PACKAGE) -> None:
    """Route this package's log records through :class:`RedactingLogRecord`.

    Idempotent, and leaves records from every other library untouched.
    """
    global _original_factory
    if _original_factory is not None:
        return
    _original_factory = logging.getLogRecordFactory()
    prefix = package + "."

    def factory(name, level, fn, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs):
        cls = RedactingLogRecord if (name == package or name.startswith(prefix)) else logging.LogRecord
        return cls(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)

    logging.setLogRecordFactory(factory)


def uninstall() -> None:
    """Restore the previous record factory. Used by tests."""
    global _original_factory
    if _original_factory is not None:
        logging.setLogRecordFactory(_original_factory)  # type: ignore[arg-type]
        _original_factory = None
