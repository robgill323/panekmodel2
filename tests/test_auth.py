"""The password gate, the bind refusal behind it, and the leak paths of both.

Three things are under test here, and they fail in different ways:

* ``configured_password`` / ``is_loopback_host`` — pure decisions, table-driven.
* ``require_auth_for_bind`` — the SEC-001 permanent fix. The failure mode is a
  process that comes up anyway, so the test asserts the refusal, and the
  mutation note in the evidence bundle records that removing the guard turns
  this file red.
* ``PasswordGateMiddleware`` — the failure mode is one unprotected route, so
  the surface is enumerated from ``app.routes`` rather than listed by hand. A
  route added later that nobody remembers to gate fails
  ``test_every_registered_route_is_probed``.
"""

from __future__ import annotations

import base64
import logging

import pytest
from fastapi.testclient import TestClient

from panekmodel2 import logging_redaction
from panekmodel2.server import auth
from panekmodel2.server.app import create_app
from panekmodel2.server.jobs import JobManager

from .conftest import FakeRunner

PASSWORD = "correct-horse-battery-staple"
VID_A = "a" * 11
URL_A = f"https://youtu.be/{VID_A}"


def basic(password: str, user: str = "throughline") -> dict:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def bearer(password: str) -> dict:
    return {"Authorization": f"Bearer {password}"}


@pytest.fixture
def gated(settings, cache_home, monkeypatch):
    """A fully wired app with the gate armed, pipeline stubbed."""
    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)
    app = create_app(JobManager(runner_factory=FakeRunner), password=PASSWORD)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def ungated(settings, cache_home, monkeypatch):
    """The same app with no password configured — the loopback default."""
    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)
    app = create_app(JobManager(runner_factory=FakeRunner), password=None)
    with TestClient(app) as client:
        yield client


# ── configured_password: blank is UNSET, not "the empty password" ────
def test_password_is_none_when_the_variable_is_absent(monkeypatch):
    monkeypatch.delenv(auth.PASSWORD_ENV, raising=False)
    assert auth.configured_password() is None


@pytest.mark.parametrize("raw", ["", " ", "\t", "\n", "   \t\n "])
def test_blank_password_counts_as_unset(monkeypatch, raw):
    """The bypass that matters.

    If a blank value counted as "set", it would arm a gate that admits every
    credential AND satisfy the non-loopback bind check at the same time — an
    unauthenticated public bind that believes it is authenticated.
    """
    monkeypatch.setenv(auth.PASSWORD_ENV, raw)
    assert auth.configured_password() is None


def test_password_is_stripped(monkeypatch):
    """``THROUGHLINE_PASSWORD=$(cat secret)`` picks up the trailing newline."""
    monkeypatch.setenv(auth.PASSWORD_ENV, f"  {PASSWORD}\n")
    assert auth.configured_password() == PASSWORD


def test_password_keeps_interior_whitespace(monkeypatch):
    monkeypatch.setenv(auth.PASSWORD_ENV, "two words")
    assert auth.configured_password() == "two words"


# ── is_loopback_host ────────────────────────────────────────────────
@pytest.mark.parametrize("host", [
    "127.0.0.1", "127.0.0.53", "127.1.2.3", "localhost", "LOCALHOST",
    "::1", "[::1]", " 127.0.0.1 ",
])
def test_loopback_hosts_are_recognized(host):
    assert auth.is_loopback_host(host) is True


@pytest.mark.parametrize("host", [
    "0.0.0.0", "::", "[::]", "192.168.1.10", "10.0.0.5", "0.0.0.0 ",
    "example.com", "workstation.lan", "", "   ", "fe80::1%lo0",
])
def test_non_loopback_hosts_are_recognized(host):
    """Anything not provably loopback is treated as network-facing.

    A hostname is not resolved — a name that happens to point at 127.0.0.1
    today can point elsewhere tomorrow, and resolving it would make the
    refusal depend on DNS.
    """
    assert auth.is_loopback_host(host) is False


# ── the bind refusal (SEC-001's permanent fix) ──────────────────────
def test_non_loopback_bind_is_refused_without_a_password(monkeypatch):
    monkeypatch.delenv(auth.PASSWORD_ENV, raising=False)
    with pytest.raises(auth.BindRefused) as excinfo:
        auth.require_auth_for_bind("0.0.0.0")
    message = str(excinfo.value)
    assert auth.PASSWORD_ENV in message, "the refusal must name the fix"
    assert "0.0.0.0" in message


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_every_network_facing_host_is_refused(monkeypatch, host):
    monkeypatch.delenv(auth.PASSWORD_ENV, raising=False)
    with pytest.raises(auth.BindRefused):
        auth.require_auth_for_bind(host)


@pytest.mark.parametrize("raw", ["", "   "])
def test_a_blank_password_does_not_unlock_a_network_bind(monkeypatch, raw):
    monkeypatch.setenv(auth.PASSWORD_ENV, raw)
    with pytest.raises(auth.BindRefused):
        auth.require_auth_for_bind("0.0.0.0")


def test_loopback_bind_needs_no_password(monkeypatch):
    monkeypatch.delenv(auth.PASSWORD_ENV, raising=False)
    assert auth.require_auth_for_bind("127.0.0.1") is None


def test_network_bind_is_allowed_once_a_password_is_set(monkeypatch):
    monkeypatch.setenv(auth.PASSWORD_ENV, PASSWORD)
    assert auth.require_auth_for_bind("0.0.0.0") == PASSWORD


def test_a_password_also_gates_a_loopback_bind(monkeypatch):
    """Setting the password is never downgraded just because it is localhost."""
    monkeypatch.setenv(auth.PASSWORD_ENV, PASSWORD)
    assert auth.require_auth_for_bind("127.0.0.1") == PASSWORD


# ── the gate: every route, not the ones someone remembered ──────────
# Concrete probe URLs. Parameterized segments are filled with values that
# would resolve if the request ever reached the router, so a 401 here cannot
# be mistaken for a 404 or a 422.
PROBES = [
    ("GET", "/"),
    ("GET", "/index.html"),
    ("GET", "/app.js"),
    ("GET", "/styles.css"),
    ("GET", "/follow.js"),
    ("GET", "/api/health"),
    ("GET", "/api/defaults"),
    ("POST", "/api/runs"),
    ("GET", "/api/runs/any-job"),
    ("GET", "/api/runs/any-job/results"),
    ("GET", "/api/runs/any-job/export/combined.csv"),
    ("GET", "/api/docs"),
    ("GET", "/openapi.json"),
    # FastAPI registers these whether or not anyone asked for them. /redoc is
    # a second rendering of the whole API schema and was not in the first
    # version of this list — test_every_registered_route_is_probed found it.
    ("GET", "/redoc"),
    ("GET", "/docs/oauth2-redirect"),
]


@pytest.mark.parametrize("method,path", PROBES)
def test_every_probe_is_401_without_credentials(gated, method, path):
    res = gated.request(method, path)
    assert res.status_code == 401, f"{method} {path} was reachable unauthenticated"
    assert res.headers["www-authenticate"].startswith("Basic ")


@pytest.mark.parametrize("method,path", PROBES)
def test_every_probe_succeeds_with_credentials(gated, method, path):
    """The gate must open, not merely close.

    A middleware that 401s unconditionally would pass every test above.
    """
    res = gated.request(method, path, headers=basic(PASSWORD))
    assert res.status_code != 401, f"{method} {path} rejected a valid password"


def test_every_registered_route_is_probed(gated):
    """Pin the surface so a new route cannot quietly arrive unprobed.

    The gate itself covers new routes by construction — it runs before the
    router. This test protects the CLAIM that the probe list above is the
    whole surface, which is the part that rots.
    """
    probed = {path for _method, path in PROBES}
    registered = set()
    for route in gated.app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue
        concrete = path.replace("{job_id}", "any-job").replace("{kind}", "combined")
        # Starlette normalizes the "/" StaticFiles mount to an empty path;
        # the probe for "/" covers it, plus four concrete asset paths.
        registered.add(concrete or "/")

    unprobed = registered - probed
    assert unprobed == set(), f"routes exist that no probe covers: {sorted(unprobed)}"


def test_an_unknown_path_is_401_not_404(gated):
    """The gate must not answer "does this path exist" to a stranger."""
    assert gated.get("/definitely-not-a-route").status_code == 401
    assert gated.get("/api/secret-admin").status_code == 401


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
def test_the_gate_covers_every_method(gated, method):
    assert gated.request(method, "/api/health").status_code == 401


def test_the_ungated_app_serves_normally(ungated):
    """No password configured is still a working localhost instrument."""
    assert ungated.get("/api/health").status_code == 200
    assert ungated.get("/").status_code == 200
    assert "www-authenticate" not in ungated.get("/api/health").headers


# ── credential checking ─────────────────────────────────────────────
def test_correct_basic_password_is_accepted(gated):
    assert gated.get("/api/health", headers=basic(PASSWORD)).status_code == 200


def test_the_username_is_ignored(gated):
    """One shared password; there are no user accounts to get wrong."""
    for user in ("", "admin", "professor", "anything at all"):
        assert gated.get("/api/health", headers=basic(PASSWORD, user=user)).status_code == 200


def test_correct_bearer_password_is_accepted(gated):
    """For curl, the container healthcheck, and scripted exports."""
    assert gated.get("/api/health", headers=bearer(PASSWORD)).status_code == 200


@pytest.mark.parametrize("header", [
    {},
    {"Authorization": ""},
    {"Authorization": "Basic"},
    {"Authorization": "Basic "},
    {"Authorization": "Bearer"},
    {"Authorization": "Bearer "},
    {"Authorization": PASSWORD},
    {"Authorization": f"Basic {PASSWORD}"},
    {"Authorization": "Digest abc"},
    {"Authorization": "Negotiate abc"},
    {"Authorization": "Basic !!!not-base64!!!"},
    {"Authorization": "Basic " + base64.b64encode(b"no-colon-here").decode()},
    {"Authorization": "Basic " + base64.b64encode(b"\xff\xfe").decode()},
    {"Authorization": "Bearer " + PASSWORD + "x"},
    {"Authorization": "Bearer " + PASSWORD[:-1]},
    {"Authorization": "Bearer " + PASSWORD.upper()},
])
def test_bad_credentials_are_rejected(gated, header):
    assert gated.get("/api/health", headers=header).status_code == 401


def test_every_authorization_header_is_considered(gated):
    """A valid credential behind a decoy still authenticates; two decoys do not.

    Pins the documented "any matching header" rule, so a refactor to
    first-header-only is a visible behaviour change rather than a silent one.
    """
    decoy = ("Authorization", "Bearer not-it")
    valid = ("Authorization", basic(PASSWORD)["Authorization"])
    assert gated.get("/api/health", headers=[decoy, valid]).status_code == 200
    assert gated.get("/api/health", headers=[decoy, decoy]).status_code == 401


def test_a_websocket_is_refused_without_credentials(settings, cache_home, monkeypatch):
    """The websocket branch of the gate, which no current route exercises.

    A route is added AFTER the gate is armed, which also demonstrates that the
    gate covers routes registered later. Without credentials the handshake is
    closed with 1008 (policy violation); with them, it is accepted.
    """
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)
    app = create_app(JobManager(runner_factory=FakeRunner), password=PASSWORD)

    async def echo(ws):
        await ws.accept()
        await ws.send_text("hello")
        await ws.close()

    # Inserted ahead of the "/" StaticFiles mount, which would otherwise
    # match first (mounts are prefix matches) and reject the websocket scope.
    from starlette.routing import WebSocketRoute

    app.router.routes.insert(0, WebSocketRoute("/ws-probe", echo))
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/ws-probe"):
                pass
        assert excinfo.value.code == 1008

        with client.websocket_connect("/ws-probe", headers=bearer(PASSWORD)) as ws:
            assert ws.receive_text() == "hello"


@pytest.mark.parametrize("scheme", ["Digest", "Negotiate", "Token", "Foo"])
def test_only_basic_and_bearer_carry_the_password(gated, scheme):
    """The right password under any other scheme is still refused.

    Not a bypass either way (the password is required regardless), but the
    documented contract is "Basic or Bearer", and accepting arbitrary schemes
    survived the mutation run until this pinned it (M11).
    """
    res = gated.get("/api/health", headers={"Authorization": f"{scheme} {PASSWORD}"})
    assert res.status_code == 401


def test_wrong_basic_password_is_rejected(gated):
    assert gated.get("/api/health", headers=basic("wrong")).status_code == 401
    assert gated.get("/api/health", headers=basic("")).status_code == 401


def test_the_scheme_is_case_insensitive(gated):
    """RFC 7235: the auth-scheme token is case-insensitive."""
    token = base64.b64encode(f"u:{PASSWORD}".encode()).decode()
    for scheme in ("Basic", "basic", "BASIC", "bAsIc"):
        res = gated.get("/api/health", headers={"Authorization": f"{scheme} {token}"})
        assert res.status_code == 200, scheme


def test_the_password_is_compared_in_constant_time():
    """A byte-at-a-time comparison leaks the password over a LAN.

    Timing is too noisy to assert on directly here, so this pins the use of
    the stdlib primitive instead and the behavioural tests above cover the
    decision itself.
    """
    import inspect

    source = inspect.getsource(auth)
    assert "hmac.compare_digest" in source
    assert "self._password ==" not in source
    assert "offered == " not in source


def test_every_credential_decision_goes_through_compare_digest(gated, monkeypatch):
    """The behavioural half of the test above, which only reads source text.

    A mutation to ``offered.encode() == self._expected`` keeps every source
    assertion true and every accept/reject test green — the decision is still
    right, only the timing leaks. Spying on the primitive catches it: both an
    accepted and a rejected credential must have been decided by it.
    """
    calls = []
    real = auth.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    assert gated.get("/api/health", headers=bearer(PASSWORD)).status_code == 200
    assert len(calls) == 1
    assert gated.get("/api/health", headers=basic("wrong")).status_code == 401
    assert len(calls) == 2


def test_the_gate_is_the_outermost_user_middleware(settings, cache_home, monkeypatch):
    """Starlette's add_middleware inserts at index 0: the LAST one added is
    outermost. The gate is outermost today only because it is the only user
    middleware; anything added after it would wrap it and see the raw
    Authorization header of every unauthenticated request (review-11 A-4)."""
    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)
    app = create_app(JobManager(runner_factory=FakeRunner), password=PASSWORD)
    assert app.user_middleware[0].cls is auth.PasswordGateMiddleware


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded"])
def test_a_cross_site_shaped_run_request_is_rejected(gated, content_type):
    """Basic credentials are ambient: a browser attaches them to cross-site
    requests. What stops a hostile page queueing runs is that POST /api/runs
    only accepts a JSON content type — the simple-request shapes a page can
    send without a CORS preflight (no type, text/plain, form) are refused.
    That is FastAPI's strictness, not the gate's, so it is pinned here."""
    headers = {**basic(PASSWORD), "Origin": "https://evil.example"}
    if content_type:
        headers["Content-Type"] = content_type
    body = '{"urls": ["https://youtu.be/aaaaaaaaaaa"], "settings": {"detect_people": false}}'
    res = gated.post("/api/runs", content=body, headers=headers)
    assert res.status_code == 422, res.text


def test_the_middleware_refuses_to_arm_with_a_blank_password():
    """Constructing the gate with no secret is a programming error, loudly."""
    for blank in ("", None):
        with pytest.raises(ValueError):
            auth.PasswordGateMiddleware(lambda *a: None, password=blank)


# ── the gate must not break the job flow (D-4) ──────────────────────
def test_a_full_authenticated_run_completes(gated):
    """D-4: runs are serialized on a worker thread started by lifespan.

    Gating the ASGI ``lifespan`` scope would leave the app never started and
    the worker never shut down, so this exercises submit → poll → results →
    export end to end through the gate.
    """
    import time

    res = gated.post(
        "/api/runs",
        json={"urls": [URL_A], "settings": {"detect_people": False}},
        headers=basic(PASSWORD),
    )
    assert res.status_code == 201, res.text
    job_id = res.json()["job_id"]

    deadline = time.time() + 30
    while time.time() < deadline:
        progress = gated.get(f"/api/runs/{job_id}", headers=basic(PASSWORD)).json()
        if progress["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert progress["status"] == "done", progress

    results = gated.get(f"/api/runs/{job_id}/results", headers=basic(PASSWORD))
    assert results.status_code == 200
    assert results.json()["totals"]["n_analyzed"] == 1

    export = gated.get(f"/api/runs/{job_id}/export/combined.csv", headers=basic(PASSWORD))
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/csv")


def test_the_lifespan_shutdown_still_reaches_the_job_worker(settings, cache_home):
    """Observable consequence of gating ``lifespan``, rather than a tautology.

    The app's shutdown hook is the only thing that stops the worker thread. If
    the gate answered 401 for the ``lifespan`` scope, the hook would never run
    and the worker would outlive the process — measured here as a live thread.
    """
    import threading

    from panekmodel2.server.jobs import JOB_THREAD_NAME

    def workers() -> list[str]:
        return [t.name for t in threading.enumerate()
                if t.name == JOB_THREAD_NAME and t.is_alive()]

    manager = JobManager(runner_factory=FakeRunner)
    app = create_app(manager, password=PASSWORD)
    with TestClient(app) as client:
        res = client.post(
            "/api/runs",
            json={"urls": [URL_A], "settings": {"detect_people": False}},
            headers=basic(PASSWORD),
        )
        assert res.status_code == 201, "startup did not complete through the gate"
        assert manager.join(timeout=30)
        assert workers(), "no worker thread to shut down — test proves nothing"

    assert not workers(), "the lifespan shutdown hook never ran"


# ── the refusal through a real entry point ──────────────────────────
def run_cli(args: list[str]):
    from typer.testing import CliRunner

    from panekmodel2.cli import app as cli_app

    return CliRunner().invoke(cli_app, args)


def flat(output: str) -> str:
    """Rich wraps at the terminal width; join lines before matching."""
    return " ".join(output.split())


def test_the_cli_exits_nonzero_instead_of_binding_an_open_interface(monkeypatch):
    """The refusal reaches the operator as an exit code, not a traceback.

    ``serve`` raises before uvicorn is called. uvicorn.run is still stubbed to
    FAIL: a guard test must never be able to perform what it guards. Unstubbed,
    a regressed bind guard made this test genuinely serve an unauthenticated
    app on *:8000 and hang (review-11 A-1), and the "kill" then depended on
    whether port 8000 happened to be busy.
    """
    import uvicorn

    reached = []

    def must_not_serve(app, **kwargs):
        reached.append(kwargs)
        pytest.fail("uvicorn.run reached: the bind guard did not refuse")

    monkeypatch.setattr(uvicorn, "run", must_not_serve)
    monkeypatch.delenv(auth.PASSWORD_ENV, raising=False)
    result = run_cli(["ui", "--host", "0.0.0.0", "--no-open"])

    assert reached == [], "uvicorn.run was called for an unauthenticated network bind"
    assert result.exit_code == 2
    assert auth.PASSWORD_ENV in flat(result.output)
    assert "Refused to start" in flat(result.output)


def test_the_cli_serves_an_open_interface_once_a_password_is_set(monkeypatch):
    """The other half: the refusal must not be unconditional."""
    monkeypatch.setenv(auth.PASSWORD_ENV, PASSWORD)
    called = {}

    def fake_run(app, **kwargs):
        called.update(kwargs, app=app)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = run_cli(["ui", "--host", "0.0.0.0", "--no-open"])

    assert result.exit_code == 0, result.output
    assert called["host"] == "0.0.0.0"
    assert "Authentication required" in flat(result.output)

    # The app actually handed to uvicorn must be the gated one. Without this,
    # serve() building create_app(password=None) — an open server on 0.0.0.0
    # with the password set — passed the whole suite (mutation M09).
    served = TestClient(called["app"])
    assert served.get("/").status_code == 401
    assert served.get("/", headers=bearer(PASSWORD)).status_code == 200


def test_the_cli_serves_loopback_with_no_password(monkeypatch):
    monkeypatch.delenv(auth.PASSWORD_ENV, raising=False)
    called = {}

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: called.update(kw))
    result = run_cli(["ui", "--host", "127.0.0.1", "--no-open"])

    assert result.exit_code == 0, result.output
    assert called["host"] == "127.0.0.1"


def test_the_cli_takes_no_password_option():
    """A secret on argv lands in shell history and in ``ps`` output.

    Asserted against the command's declared parameters, not against --help
    text — the help output quotes the docstring, which mentions the flag it
    deliberately does not offer.
    """
    import typer

    from panekmodel2.cli import app as cli_app

    ui_command = typer.main.get_command(cli_app).commands["ui"]
    options = {opt for param in ui_command.params for opt in param.opts}

    assert "--password" not in options
    assert {"--host", "--port"} <= options, "sanity: the options were found at all"


# ── create_app reads the environment by default ─────────────────────
def test_create_app_arms_the_gate_from_the_environment(settings, cache_home, monkeypatch):
    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)
    monkeypatch.setenv(auth.PASSWORD_ENV, PASSWORD)
    with TestClient(create_app(JobManager(runner_factory=FakeRunner))) as client:
        assert client.get("/api/health").status_code == 401
        assert client.get("/api/health", headers=basic(PASSWORD)).status_code == 200


def test_create_app_is_open_when_the_environment_is_empty(settings, cache_home, monkeypatch):
    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)
    monkeypatch.delenv(auth.PASSWORD_ENV, raising=False)
    with TestClient(create_app(JobManager(runner_factory=FakeRunner))) as client:
        assert client.get("/api/health").status_code == 200


def test_the_password_is_not_a_settings_field():
    """It must not ride along in ``model_dump()``.

    ``Settings`` is dumped into /api/health, the results payload and the run
    metadata of every CSV export. A secret held there is one allowlist edit
    away from being published, so the gate reads os.environ directly.
    """
    from panekmodel2.config import Settings

    fields = set(Settings.model_fields)
    assert not [name for name in fields if "password" in name or "throughline" in name]


def test_the_password_does_not_reach_the_settings_summary(settings, monkeypatch):
    from panekmodel2.server.jobs import settings_summary

    monkeypatch.setenv(auth.PASSWORD_ENV, PASSWORD)
    assert PASSWORD not in repr(settings_summary(settings))


# ── the password must not reach a log sink ──────────────────────────
class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.NOTSET)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


@pytest.fixture
def captured():
    """Everything any logger emits, formatted as a handler would write it."""
    handler = Capture()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


def test_no_auth_path_logs_the_password(gated, captured, monkeypatch):
    """Both the plaintext and the base64 a Basic header would carry."""
    monkeypatch.setenv(auth.PASSWORD_ENV, PASSWORD)
    auth.configured_password()
    auth.require_auth_for_bind("0.0.0.0")
    auth.warn_if_weak(PASSWORD)
    gated.get("/api/health")
    gated.get("/api/health", headers=basic("wrong"))
    gated.get("/api/health", headers=basic(PASSWORD))
    gated.get("/api/health", headers=bearer(PASSWORD))

    blob = "\n".join(captured.lines)
    encoded = base64.b64encode(f"throughline:{PASSWORD}".encode()).decode()
    assert PASSWORD not in blob
    assert encoded not in blob


def test_the_weak_password_warning_names_the_length_not_the_password(captured):
    assert auth.warn_if_weak("short") is True
    blob = "\n".join(captured.lines)
    assert "short" not in blob
    assert str(len("short")) in blob


def test_a_long_password_draws_no_warning(captured):
    assert auth.warn_if_weak("x" * auth.MIN_RECOMMENDED_LENGTH) is False
    assert captured.lines == []


def test_the_refusal_message_is_actionable_without_echoing_the_variable(monkeypatch):
    """A refusal is only useful if it says how to get past it."""
    monkeypatch.setenv(auth.PASSWORD_ENV, "  ")
    with pytest.raises(auth.BindRefused) as excinfo:
        auth.require_auth_for_bind("0.0.0.0")
    message = str(excinfo.value)
    assert auth.PASSWORD_ENV in message
    assert "127.0.0.1" in message, "the refusal must name the safe alternative"


# ── the redactor has to learn the header form ───────────────────────
# SECRET_PARAMS already contains "password", so `?password=x` is scrubbed.
# `Authorization: Basic <base64>` is a header, not a query parameter, and the
# `\bname=value` pattern cannot match it — base64 is encoding, not protection.
def test_a_basic_authorization_header_is_redacted_from_a_log_line():
    token = base64.b64encode(f"u:{PASSWORD}".encode()).decode()
    line = f"request failed: Authorization: Basic {token}"
    scrubbed = logging_redaction.redact_secrets(line)
    assert token not in scrubbed
    assert PASSWORD not in scrubbed
    assert logging_redaction.REDACTED in scrubbed


@pytest.mark.parametrize("scheme", ["Basic", "basic", "Bearer", "bearer", "BASIC", "Token"])
def test_every_authorization_scheme_is_redacted(scheme):
    line = f"headers={{'Authorization': '{scheme} s3cr3t-material'}}"
    scrubbed = logging_redaction.redact_secrets(line)
    assert "s3cr3t-material" not in scrubbed


def test_the_bare_authorization_header_form_is_redacted():
    """As an access log or a header dict repr would render it."""
    for line in [
        "authorization: Bearer abcdef123456",
        "Authorization:Basic dXNlcjpwYXNz",
        '{"authorization": "Bearer abcdef123456"}',
    ]:
        scrubbed = logging_redaction.redact_secrets(line)
        assert "abcdef123456" not in scrubbed
        assert "dXNlcjpwYXNz" not in scrubbed


def test_redaction_still_leaves_ordinary_text_alone():
    """The new pattern must not eat unrelated prose."""
    for line in [
        "authorization is required for this endpoint",
        "the user is not authorized",
        "fetching 12 segments",
    ]:
        assert logging_redaction.redact_secrets(line) == line


def test_the_query_parameter_form_still_works():
    """Regression guard: the new pattern must not displace the old one."""
    scrubbed = logging_redaction.redact_secrets("https://x/y?key=AIzaSecret&z=1")
    assert "AIzaSecret" not in scrubbed
    assert "z=1" in scrubbed


def test_a_logged_authorization_header_is_scrubbed_through_real_logging(captured):
    """End to end through the record factory, not just the helper."""
    token = base64.b64encode(f"u:{PASSWORD}".encode()).decode()
    logging.getLogger("panekmodel2.server.auth").warning(
        "unexpected request state: %s", f"Authorization: Basic {token}"
    )
    blob = "\n".join(captured.lines)
    assert token not in blob
    assert PASSWORD not in blob
