"""One shared password in front of the whole server, and the bind it permits.

Two controls, deliberately coupled:

* ``PasswordGateMiddleware`` rejects every unauthenticated request.
* ``require_auth_for_bind`` **refuses to start** when asked to bind a
  network-facing interface with no password configured.

The refusal is the permanent fix for SEC-001, whose shape was an
unauthenticated UI reachable from the LAN running unbounded compute. The audit
suggested a warning; a warning printed into a scrollback nobody reads is not a
control, so the process does not come up instead.

**The password is read from ``os.environ`` and never from ``Settings``.**
``Settings`` is ``model_dump()``ed into ``/api/health``, into the results
payload of every run and into the metadata of every CSV export. A secret held
in that object is one allowlist edit away from being published; keeping it out
of the model means that mistake cannot be made here.

What this is not: there is no user account, no session, no logout, and no
per-request rate limit. Brute-force resistance comes from password entropy
alone, which is why ``warn_if_weak`` exists and why the runbook tells the
operator to generate the value with ``secrets.token_urlsafe``. Basic
credentials are also base64, not encrypted — over plain HTTP they are readable
by anything on the path, so the runbook scopes this to a trusted network or a
TLS-terminating proxy. All three limits are recorded rather than papered over.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import ipaddress
import logging
import os
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

PASSWORD_ENV = "THROUGHLINE_PASSWORD"

REALM = "Throughline"

# Below this, a password is guessable faster than anyone would notice, and
# there is no rate limit to slow that down. Warned about, not enforced —
# refusing here would lock an operator out of their own deployment mid-session
# over a policy this module invented.
MIN_RECOMMENDED_LENGTH = 16

# Names that mean "this machine" without being IP literals.
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


class BindRefused(RuntimeError):
    """Raised instead of binding a network interface with no password set."""


def configured_password() -> Optional[str]:
    """The configured password, or ``None`` when auth is not configured.

    A blank or whitespace-only value counts as **unset**. Treating it as "the
    empty password" would arm a gate that admits every credential and satisfy
    the non-loopback bind check at the same time — a public bind that believes
    it is authenticated, which is the one bypass worth designing against.

    The value is stripped: ``THROUGHLINE_PASSWORD=$(cat secret)`` and a
    here-doc in a compose file both pick up a trailing newline, and a password
    that cannot be typed into a browser prompt is not a usable one.
    """
    raw = os.environ.get(PASSWORD_ENV)
    if raw is None:
        return None
    return raw.strip() or None


def is_loopback_host(host: str) -> bool:
    """True only when *host* provably names this machine.

    Hostnames are **not** resolved. A name pointing at 127.0.0.1 today can
    point elsewhere tomorrow, and resolving it would make whether the server
    starts depend on DNS. Anything not provably loopback is treated as
    network-facing, so the unknown case fails safe.
    """
    candidate = host.strip()
    if not candidate:
        return False  # uvicorn reads "" as every interface
    if candidate.lower() in _LOOPBACK_NAMES:
        return True
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]  # uvicorn accepts bracketed IPv6
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def warn_if_weak(password: str) -> bool:
    """Log a warning for a short password. Returns whether it warned.

    Logs the *length*, never the value — this runs at startup, where the log
    is most likely to be captured by a supervisor or shipped somewhere.
    """
    if len(password) >= MIN_RECOMMENDED_LENGTH:
        return False
    logger.warning(
        "%s is only %d characters. There is no rate limit on authentication "
        "attempts, so use at least %d (python -c 'import secrets; "
        "print(secrets.token_urlsafe(32))').",
        PASSWORD_ENV,
        len(password),
        MIN_RECOMMENDED_LENGTH,
    )
    return True


def require_auth_for_bind(host: str) -> Optional[str]:
    """The password to enforce for a bind to *host*, or ``None`` for open loopback.

    Raises :class:`BindRefused` when *host* is network-facing and no password
    is configured. The message names both ways out, because a refusal the
    operator cannot act on just gets worked around.
    """
    password = configured_password()
    if password is None and not is_loopback_host(host):
        raise BindRefused(
            f"Refusing to bind {host!r}: {PASSWORD_ENV} is not set.\n"
            f"This server runs unbounded compute on request and would be "
            f"reachable by anything that can route to this machine.\n"
            f"Either set {PASSWORD_ENV} to a high-entropy value, or bind "
            f"127.0.0.1 (the default) and reach it through an SSH tunnel."
        )
    if password is not None:
        warn_if_weak(password)
    return password


def _basic_password(token: str) -> Optional[str]:
    """The password half of a Basic ``user:pass`` token, or None if malformed."""
    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    if ":" not in decoded:
        return None  # RFC 7617 requires user-id ":" password
    return decoded.partition(":")[2]


class PasswordGateMiddleware:
    """Rejects unauthenticated requests before they reach the router.

    Raw ASGI rather than a route dependency or ``BaseHTTPMiddleware``. A
    dependency only protects the routes someone remembered to decorate, and
    ``app.mount()`` — which is how the whole SPA is served — is not a route at
    all. Sitting above the router means ``/``, the static mount, ``/api/docs``,
    ``/openapi.json``, the CSV exports and every route added later are covered
    by construction, and an unknown path answers 401 rather than disclosing
    whether it exists.

    Accepts two schemes for the same single password: ``Basic`` (any username;
    this is what makes a browser show a prompt) and ``Bearer`` (curl, the
    container healthcheck, scripted exports).

    Nothing is logged per request — not the header, not a failure, not the
    client address. An unauthenticated stranger must not be able to drive log
    volume, and the ``Authorization`` header is the one string that must never
    reach a sink. ``logging_redaction`` scrubs it if some other layer logs it
    anyway; this layer simply does not.
    """

    def __init__(self, app, password: str) -> None:
        if not password:
            raise ValueError(
                "PasswordGateMiddleware requires a non-empty password; "
                "leave the gate off instead of arming it with no secret."
            )
        self._app = app
        self._expected = password.encode("utf-8")

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        # "lifespan" is the application's own startup/shutdown, not a client
        # request. Answering 401 here would leave the app never started and
        # the job worker never shut down (D-4).
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return

        if self._authorized(scope):
            await self._app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})  # policy violation
            return

        await self._challenge(send)

    def _authorized(self, scope: dict) -> bool:
        """True when any Authorization header carries the password.

        Every matching header is checked, not just the first, so a request
        carrying a decoy alongside a valid credential is not a way to smuggle
        one past a parser that stops early.
        """
        return any(
            self._matches(value)
            for name, value in scope.get("headers", ())
            if name == b"authorization"
        )

    def _matches(self, raw: bytes) -> bool:
        # latin-1 is the header wire encoding and cannot raise, so a junk
        # header becomes a failed comparison rather than a 500.
        scheme, _, credentials = raw.decode("latin-1").partition(" ")
        scheme = scheme.strip().lower()
        credentials = credentials.strip()
        if not credentials:
            return False

        if scheme == "basic":
            offered = _basic_password(credentials)
        elif scheme == "bearer":
            offered = credentials
        else:
            return False

        if not offered:
            return False
        return hmac.compare_digest(offered.encode("utf-8"), self._expected)

    async def _challenge(self, send: Callable[[dict], Awaitable[None]]) -> None:
        body = b'{"detail":"Authentication required."}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                # Makes a browser prompt for the password, which is the whole
                # of the login UX — there is no login page to keep in sync.
                (b"www-authenticate", f'Basic realm="{REALM}", charset="UTF-8"'.encode()),
                (b"content-length", str(len(body)).encode()),
                (b"cache-control", b"no-store"),
            ],
        })
        await send({"type": "http.response.body", "body": body})
