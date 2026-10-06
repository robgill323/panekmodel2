"""Invariants of the packaging baseline.

Two very different kinds of test live here, and the difference matters when
reading a failure:

* **Static assertions** about the Dockerfile, the compose files, the locks and
  the runbook. They pin decisions that are easy to undo by accident —
  publishing the image, dropping the password requirement, exposing the port
  on every interface, baking a secret into a layer. They are assertions about
  TEXT and prove nothing about whether the image builds or runs; that needs a
  Docker daemon (docs/DEPLOYMENT.md §11).
* **Real-socket tests**, which start uvicorn on a loopback port: the container
  healthcheck run as a subprocess, the gate over a genuine HTTP connection,
  and a capture of every log record at uvicorn's most verbose level while
  credentials are presented.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from panekmodel2.server.app import create_app
from panekmodel2.server.jobs import JobManager

from .conftest import FakeRunner

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
COMPOSE_GPU = ROOT / "docker-compose.gpu.yml"
DOCKERIGNORE = ROOT / ".dockerignore"
RUNBOOK = ROOT / "docs/DEPLOYMENT.md"
HEALTHCHECK = ROOT / "docker/healthcheck.py"
LOCK = ROOT / "requirements.lock"
LOCK_DEV = ROOT / "requirements-dev.lock"
LOCK_SCRIPT = ROOT / "scripts/lock.sh"
PYPROJECT = ROOT / "pyproject.toml"

PASSWORD = "correct-horse-battery-staple"

HOME = "/home/throughline"


# ── the files exist at the paths the runbook cites ──────────────────
@pytest.mark.parametrize("path", [
    DOCKERFILE, COMPOSE, COMPOSE_GPU, DOCKERIGNORE, RUNBOOK, HEALTHCHECK,
    LOCK, LOCK_DEV, LOCK_SCRIPT,
])
def test_the_packaging_files_exist(path):
    assert path.is_file(), f"{path.relative_to(ROOT)} is missing"


def test_the_lock_script_is_executable():
    assert os.access(LOCK_SCRIPT, os.X_OK)


def compose(path: Path = COMPOSE) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def service(path: Path = COMPOSE) -> dict:
    return compose(path)["services"]["throughline"]


@pytest.mark.parametrize("path", [COMPOSE, COMPOSE_GPU])
def test_the_compose_files_are_valid_yaml(path):
    """Text matching cannot see a file docker compose would refuse to parse.

    An unquoted ``:?`` message containing ": " is read by YAML as a nested
    mapping — every substring test passes and ``compose up`` fails.
    """
    assert "throughline" in compose(path)["services"]


def test_the_gpu_override_only_overrides_and_does_not_redefine():
    """An override that re-declares image/ports/volumes would silently diverge."""
    overlay = service(COMPOSE_GPU)
    allowed = {"build", "environment", "deploy"}
    assert set(overlay) <= allowed, f"the override redefines {sorted(set(overlay) - allowed)}"


def test_the_gpu_override_is_marked_untested():
    assert "UNTESTED-PENDING-RIG" in COMPOSE_GPU.read_text().splitlines()[0]


def docker_compose_config(*files: Path, password: str | None):
    """Run Docker's own compose parser. Client-side: needs the docker CLI,
    NOT a running daemon. ``--env-file /dev/null`` keeps a developer's real
    .env (which may hold a password) out of the result."""
    import shutil

    if shutil.which("docker") is None:
        pytest.skip("docker CLI not installed")
    env = {k: v for k, v in os.environ.items() if k != "THROUGHLINE_PASSWORD"}
    if password is not None:
        env["THROUGHLINE_PASSWORD"] = password
    args = ["docker", "compose", "--env-file", os.devnull]
    for f in files:
        args += ["-f", str(f)]
    return subprocess.run(args + ["config", "--format", "json"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("password", [None, ""], ids=["unset", "blank"])
def test_docker_compose_itself_refuses_without_a_password(password):
    """A whitespace-only value gets past Compose (it is not empty) and is then
    refused by the server's own bind check, which strips it — the second line
    of the same rule, tested in test_auth.py."""
    result = docker_compose_config(COMPOSE, password=password)
    assert result.returncode != 0
    assert "THROUGHLINE_PASSWORD" in result.stderr


@pytest.mark.parametrize("files", [(COMPOSE,), (COMPOSE, COMPOSE_GPU)], ids=["cpu", "cpu+gpu"])
def test_docker_compose_parses_the_files_and_keeps_loopback(files):
    """What Docker resolves, not what the text says."""
    import json

    result = docker_compose_config(*files, password="parse-only-placeholder")
    assert result.returncode == 0, result.stderr
    svc = json.loads(result.stdout)["services"]["throughline"]
    assert [p["host_ip"] for p in svc["ports"]] == ["127.0.0.1"]
    assert {v["source"]: v["target"] for v in svc["volumes"]} == EXPECTED_MOUNTS
    assert svc["restart"] == "unless-stopped"
    gpu = len(files) == 2
    assert svc["environment"]["CUDA"] == ("true" if gpu else "false")
    assert ("cu1" in svc["build"]["args"]["TORCH_INDEX_URL"]) is gpu


# ── the image must not be publishable ───────────────────────────────
REGISTRY_MARKERS = [
    "docker push", "docker.io/", "ghcr.io", "gcr.io", "quay.io",
    "amazonaws.com", "azurecr.io", "registry.hub.docker.com",
]


@pytest.mark.parametrize("marker", REGISTRY_MARKERS)
def test_no_packaging_file_references_a_registry(marker):
    for path in (DOCKERFILE, COMPOSE, COMPOSE_GPU, LOCK_SCRIPT):
        assert marker not in path.read_text(), \
            f"{path.relative_to(ROOT)} references {marker!r} — the image is not published"


def test_the_compose_image_tag_has_no_registry_host():
    tag = service()["image"]
    assert tag == "throughline:local", tag
    assert "/" not in tag, "a slash in an image tag means a registry namespace"


def test_there_is_no_ci_workflow_building_the_image():
    for path in (ROOT / ".github", ROOT / ".gitlab-ci.yml", ROOT / ".circleci"):
        assert not path.exists(), f"{path.name} exists — check it does not publish"


# ── the password requirement must survive ───────────────────────────
def test_compose_refuses_to_start_without_a_password():
    """``:?`` is Compose's own required-variable syntax — a second, earlier
    enforcement of the server's bind-time refusal."""
    value = service()["environment"]["THROUGHLINE_PASSWORD"]
    assert value.startswith("${THROUGHLINE_PASSWORD:?"), \
        f"the required-variable guard was weakened to {value!r}"
    assert ":-" not in value, "a default value would silence the guard entirely"


@pytest.mark.parametrize("name", ["YOUTUBE_API_KEY", "HF_TOKEN"])
def test_optional_credentials_are_passed_through_from_the_host(name):
    """Interpolated from the host env / .env — never a literal in the file."""
    assert service()["environment"][name] == f"${{{name}:-}}"


def test_compose_restarts_unless_stopped():
    assert service()["restart"] == "unless-stopped"


def test_the_container_binds_all_interfaces_so_the_gate_is_mandatory():
    """If the CMD bound loopback, the server would start with no password and
    the published port would reach a dead interface."""
    text = DOCKERFILE.read_text()
    assert '"--host", "0.0.0.0"' in text
    assert '"--no-open"' in text, "there is no browser in a container"


def test_the_host_port_is_published_on_loopback_by_default():
    ports = service()["ports"]
    assert len(ports) == 1, ports
    assert ports[0].startswith("${THROUGHLINE_BIND_ADDR:-127.0.0.1}:"), \
        f"the published port no longer defaults to loopback: {ports[0]}"


# ── volumes: the three caches persist, and the runtime user owns them ─
EXPECTED_MOUNTS = {
    "throughline-transcripts": f"{HOME}/.panekmodel2_cache",
    "throughline-models": f"{HOME}/.cache",
    "throughline-exports": f"{HOME}/exports",
}


def test_each_cache_has_its_own_named_volume():
    mounts = dict(v.split(":", 1) for v in service()["volumes"])
    assert mounts == EXPECTED_MOUNTS
    volumes = compose()["volumes"]
    assert set(volumes) == set(EXPECTED_MOUNTS), "volumes must be named, not bind mounts"
    for key, spec in volumes.items():
        assert spec == {"name": key}, f"{key} needs a fixed name; the backup commands use it"


def test_the_volume_mount_points_exist_owned_by_the_runtime_user():
    """Docker seeds an empty named volume from the image directory, ownership
    included. A mount point the Dockerfile did not create is root-owned, and
    the non-root process cannot write its own cache."""
    text = dockerfile_instructions()
    create = re.search(r"install -d -o throughline -g throughline((?:\s*\\\n\s*\S+)+)", text)
    assert create, "the mount points are not created with the runtime user's ownership"
    created = set(create.group(1).replace("\\", " ").split())
    for mount in EXPECTED_MOUNTS.values():
        assert mount in created, f"{mount} is mounted but not created by the image"


def test_nltk_data_lands_inside_the_model_volume():
    text = dockerfile_instructions()
    assert f"NLTK_DATA={HOME}/.cache/nltk_data" in text
    assert f"{HOME}/.cache/nltk_data" in text.split("install -d", 1)[1], \
        "NLTK only downloads into NLTK_DATA if the directory already exists"


# ── no secret enters a layer ────────────────────────────────────────
def dockerfile_instructions() -> str:
    """The Dockerfile with comments stripped — the parts Docker acts on."""
    return "\n".join(line for line in DOCKERFILE.read_text().splitlines()
                     if not line.lstrip().startswith("#"))


@pytest.mark.parametrize("secret", ["THROUGHLINE_PASSWORD", "YOUTUBE_API_KEY", "HF_TOKEN"])
def test_no_secret_is_named_in_any_dockerfile_instruction(secret):
    """An ENV would bake it into the image, an ARG would put it in the build
    history, a shell-form CMD/HEALTHCHECK would expand it onto argv."""
    assert secret not in dockerfile_instructions()


def test_every_copy_source_is_on_an_explicit_allowlist():
    """No ``COPY . .`` — the only way a stray .env or credential file could
    reach a layer despite .dockerignore."""
    allowed = {"pyproject.toml", "README.md", "src", "requirements*.lock", "docker/healthcheck.py"}
    sources = set()
    for line in dockerfile_instructions().splitlines():
        if line.startswith(("COPY", "ADD")):
            assert line.startswith("COPY "), "ADD can fetch URLs and unpack archives; use COPY"
            sources.update(line.split()[1:-1])
    assert sources <= allowed, f"unexpected COPY sources: {sorted(sources - allowed)}"
    assert sources, "found no COPY instructions at all — the parser is broken"


def test_the_healthcheck_is_exec_form_so_no_shell_expands_anything():
    assert 'CMD ["python", "/usr/local/bin/throughline-healthcheck"]' in dockerfile_instructions()


def test_the_healthcheck_script_reads_the_password_from_the_environment():
    script = HEALTHCHECK.read_text()
    assert 'os.environ.get("THROUGHLINE_PASSWORD")' in script
    assert "sys.argv" not in script


def test_the_image_does_not_run_as_root():
    text = dockerfile_instructions()
    assert "USER throughline" in text
    assert text.index("USER throughline") < text.index('CMD ["panekmodel2"'), \
        "USER must precede CMD or the process still runs as root"


@pytest.mark.parametrize("pattern", [
    ".env", ".env.*", "*.pem", "*.key", "credentials*.json", "client_secret*.json",
    ".venv/", ".git/", "artifacts/",
])
def test_the_dockerignore_excludes_host_state_and_secrets(pattern):
    lines = {line.strip() for line in DOCKERIGNORE.read_text().splitlines()}
    assert pattern in lines, f"{pattern} would be sent to the build context"


# ── the locks must actually be locks ────────────────────────────────
def pins(path: Path) -> dict:
    """{normalized name: version} for every top-level requirement line."""
    out = {}
    for line in path.read_text().splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            name, _, version = line.split(" ")[0].partition("==")
            out[re.sub(r"[-_.]+", "-", name).lower()] = version
    return out


@pytest.mark.parametrize("path", [LOCK, LOCK_DEV])
def test_every_requirement_is_pinned_and_hashed(path):
    """A lock without hashes is a suggestion."""
    text = path.read_text()
    assert "--hash=sha256:" in text, "not hash-pinned"
    unpinned = [name for name, version in pins(path).items() if not version]
    assert unpinned == [], f"unpinned requirements: {unpinned}"
    # Each pin is followed by at least one hash line before the next pin.
    blocks = re.split(r"\n(?=[^\s#])", text.split("\n\n", 1)[-1])
    unhashed = [b.split()[0] for b in blocks if "==" in b and "--hash=sha256:" not in b]
    assert unhashed == [], f"pins without hashes: {unhashed}"


@pytest.mark.parametrize("path", [LOCK, LOCK_DEV])
def test_the_locks_carry_no_unpinned_warning(path):
    """pip-tools emits this when a package (e.g. setuptools) is left for pip
    to resolve, which makes --require-hashes fail on a clean machine."""
    assert "The following packages were not pinned" not in path.read_text()


def declared(extra: str | None = None) -> set:
    tomllib = pytest.importorskip("tomllib", reason="stdlib from Python 3.11")
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    reqs = project["optional-dependencies"][extra] if extra else project["dependencies"]
    return {re.sub(r"[-_.]+", "-", re.split(r"[<>=!~;\[ ]", r, maxsplit=1)[0]).lower() for r in reqs}


def test_the_runtime_lock_covers_every_declared_dependency():
    """Adding a dependency to pyproject without relocking fails here, not on
    a clean machine at install time."""
    missing = declared() - set(pins(LOCK))
    assert missing == set(), f"declared but not locked (run scripts/lock.sh): {missing}"


def test_the_dev_lock_is_the_runtime_lock_plus_the_dev_extra():
    runtime, dev = pins(LOCK), pins(LOCK_DEV)
    missing = declared("dev") - set(dev)
    assert missing == set(), f"dev extra not locked: {missing}"
    drifted = {n: (v, dev.get(n)) for n, v in runtime.items() if dev.get(n) != v}
    assert drifted == {}, f"runtime pins differ between the two locks: {drifted}"
    assert "pytest" in dev and "pytest" not in runtime


def test_pyproject_is_the_lock_input():
    """pyproject.toml plays the requirements.in role; a second dependency
    list would drift from it."""
    assert not (ROOT / "requirements.in").exists()
    for path in (LOCK, LOCK_DEV):
        header = "".join(path.read_text().splitlines(keepends=True)[:6])
        assert "pyproject.toml" in header, f"{path.name} was not compiled from pyproject.toml"


# ── the runbook ─────────────────────────────────────────────────────
@pytest.mark.parametrize("ghost", ["streamlit", "pytube", "oauth", "google_credentials"])
def test_no_packaging_file_mentions_a_deleted_surface(ghost):
    """D-5 removed Streamlit, the OAuth captions tier and pytube."""
    for path in (DOCKERFILE, COMPOSE, COMPOSE_GPU, RUNBOOK, DOCKERIGNORE, LOCK_SCRIPT):
        assert ghost not in path.read_text().lower(), \
            f"{path.relative_to(ROOT)} refers to {ghost}, removed in 0.2.0"


@pytest.mark.parametrize("topic", [
    "Install", "First start", "Update", "Rollback", "Back up", "Access",
    "Tailscale", "campus IT", "Password", "Apptainer", "WSL2",
])
def test_the_runbook_covers_each_operator_task(topic):
    headings = [l for l in RUNBOOK.read_text().splitlines() if l.startswith("#")]
    assert any(topic.lower() in h.lower() for h in headings), \
        f"no runbook heading covers {topic!r}"


def test_the_runbook_says_how_to_make_a_password():
    assert "secrets.token_urlsafe" in RUNBOOK.read_text()


def test_the_runbook_declares_what_was_never_executed():
    """The disclosure most likely to be tidied away as an embarrassment."""
    text = RUNBOOK.read_text()
    section = text.split("## 11. What has actually been verified", 1)[1]
    never = section.split("**Not executed", 1)[1]
    assert "image" in never and "GPU" in never and "Linux" in never


# ── real sockets ────────────────────────────────────────────────────
def free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@pytest.fixture
def live_server_factory(settings, cache_home, monkeypatch):
    """Start a real uvicorn server on a loopback port with the gate armed.

    Everything else in the suite talks to the app through TestClient's
    in-process transport. This is the one place a genuine TCP connection and
    HTTP parser are involved — which is what a browser and the healthcheck
    face.
    """
    import uvicorn

    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)
    started = []

    def start(log_level: str = "warning") -> int:
        port = free_port()
        app = create_app(JobManager(runner_factory=FakeRunner), password=PASSWORD)
        # log_config=None: do not let uvicorn dictConfig the process's logging
        # out from under the rest of the suite. log_level still sets the
        # uvicorn loggers' levels, which is what decides what they emit.
        server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level=log_level, log_config=None,
        ))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 30
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        assert server.started, "uvicorn did not start"
        started.append((server, thread))
        return port

    yield start
    for server, thread in started:
        server.should_exit = True
        thread.join(timeout=30)


@pytest.fixture
def live_server(live_server_factory):
    return live_server_factory()


def run_healthcheck(port: int, password: str | None):
    env = dict(os.environ)
    env["THROUGHLINE_HEALTHCHECK_URL"] = f"http://127.0.0.1:{port}/api/health"
    env.pop("THROUGHLINE_PASSWORD", None)
    if password is not None:
        env["THROUGHLINE_PASSWORD"] = password
    return subprocess.run(
        [sys.executable, str(HEALTHCHECK)], env=env, capture_output=True, text=True, timeout=60,
    )


def test_the_healthcheck_passes_against_a_real_gated_server(live_server):
    result = run_healthcheck(live_server, PASSWORD)
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_the_healthcheck_fails_distinguishably_when_the_password_is_wrong(live_server):
    result = run_healthcheck(live_server, "not-the-password")
    assert result.returncode == 1
    assert "401" in result.stderr and "credentials rejected" in result.stderr


def test_the_healthcheck_fails_when_no_password_is_supplied(live_server):
    result = run_healthcheck(live_server, None)
    assert result.returncode == 1
    assert "401" in result.stderr


def test_the_healthcheck_fails_when_nothing_is_listening():
    result = run_healthcheck(free_port(), PASSWORD)
    assert result.returncode == 1
    assert "unhealthy" in result.stderr


def basic_header(password: str, user: str = "u") -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def test_the_gate_rejects_over_a_real_connection(live_server):
    """If the 401 depended on something the ASGI test transport does, every
    other gate test would pass while a real deployment served the SPA."""
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"http://127.0.0.1:{live_server}/", timeout=30)
    assert excinfo.value.code == 401
    assert excinfo.value.headers["www-authenticate"].startswith("Basic ")


def test_the_spa_is_served_over_a_real_connection_with_credentials(live_server):
    request = urllib.request.Request(f"http://127.0.0.1:{live_server}/")
    request.add_header("Authorization", basic_header(PASSWORD))
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200
        assert "Throughline" in response.read().decode()


class _Everything(logging.Handler):
    """Renders every record fully: message, args, traceback and extras."""

    def __init__(self):
        super().__init__(level=1)
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))

    def emit(self, record):
        self.lines.append(self.format(record))
        self.lines.append(repr(record.args))
        self.lines.append(repr(vars(record)))


def test_no_credential_reaches_any_log_record_over_a_real_connection(live_server_factory):
    """The brief's CRITICAL: no header logging may expose the credential.

    Run at uvicorn's most verbose level ("trace", which logs every ASGI scope
    and message), capture every record from every logger in the process, and
    present the password every way a client can: Basic, Bearer, a wrong Basic
    credential, and an unauthenticated request. Neither the password, its
    base64 encoding, nor the wrong attempt may appear anywhere.

    Non-vacuous by construction: the test also requires the capture to have
    seen uvicorn's access line and a trace-level ASGI record, so a handler
    that captured nothing cannot pass it.
    """
    wrong = "canary-wrong-password-7f3e"
    secrets_ = [
        PASSWORD,
        base64.b64encode(f"u:{PASSWORD}".encode()).decode(),
        wrong,
        base64.b64encode(f"u:{wrong}".encode()).decode(),
    ]

    handler = _Everything()
    root = logging.getLogger()
    package = logging.getLogger("panekmodel2")
    old_package_level = package.level
    root.addHandler(handler)
    package.setLevel(1)
    try:
        port = live_server_factory(log_level="trace")
        base = f"http://127.0.0.1:{port}/api/health"
        for header in (basic_header(PASSWORD), f"Bearer {PASSWORD}", basic_header(wrong), None):
            request = urllib.request.Request(base)
            if header:
                request.add_header("Authorization", header)
            try:
                urllib.request.urlopen(request, timeout=30).close()
            except urllib.error.HTTPError as exc:
                assert exc.code == 401
        time.sleep(0.2)  # the access log is written after the response is sent
    finally:
        root.removeHandler(handler)
        package.setLevel(old_package_level)

    captured = "\n".join(handler.lines)
    assert '"GET /api/health HTTP/1.1" 200' in captured, "the access log was not captured"
    assert '"GET /api/health HTTP/1.1" 401' in captured, "the rejected requests were not captured"
    assert "ASGI [" in captured, "trace-level logging was not active"
    for secret in secrets_:
        assert secret not in captured, f"a credential reached a log record: {secret[:6]}…"
