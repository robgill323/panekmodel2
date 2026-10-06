"""Settings against the .env file the runbook tells the operator to write (B-1).

docs/DEPLOYMENT.md puts THROUGHLINE_PASSWORD and the compose-only keys of
Appendix B into ``.env`` at the top of the checkout. On the native (systemd)
path that is also the server's working directory, so ``Settings()`` reads that
same file. With pydantic-settings' default ``extra="forbid"``, every key that
is not a Settings field raised ``extra_forbidden`` — an authenticated
/api/health answered 500, and every run failed the same way.

The fix is ``extra="ignore"``, and it must be IGNORE, not ALLOW: ``allow``
would store ``throughline_password`` on the model, and the model is serialized
into /api/health, every run's results and every CSV export. The leak tests
below pin that distinction; a field-only check cannot, because ``allow`` adds
no declared field.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from panekmodel2 import config
from panekmodel2.config import Settings
from panekmodel2.server.app import create_app
from panekmodel2.server.jobs import JobManager, settings_summary

from .conftest import FakeRunner

PASSWORD = "dotenv-canary-password-4c1d9e"

# Exactly what the runbook directs into .env: §3's password, every Appendix B
# key (compose-only ones included), and the retired GOOGLE_* keys the README
# says an old .env may still carry.
DOTENV = f"""\
THROUGHLINE_PASSWORD={PASSWORD}
THROUGHLINE_BIND_ADDR=127.0.0.1
THROUGHLINE_PORT=8000
YOUTUBE_API_KEY=
HF_TOKEN=
EMBEDDING_MODEL=all-mpnet-base-v2
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
TOPIC_GRANULARITY=standard
TOPIC_REDUCE_TO=10
CHUNK_MAX_SECONDS=60
USE_WHISPER_FALLBACK=false
WHISPER_MODEL=small
CUDA=false
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
REQUIREMENTS_FILE=
GPU_COUNT=1
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_TOKEN_FILE=token.json
"""


@pytest.fixture
def runbook_dotenv(tmp_path, monkeypatch):
    """A working directory holding the runbook's .env, as systemd's would."""
    (tmp_path / ".env").write_text(DOTENV)
    monkeypatch.chdir(tmp_path)
    config.get_settings.cache_clear()  # lru_cache: do not reuse another test's Settings
    yield tmp_path
    config.get_settings.cache_clear()


def test_settings_load_with_the_runbook_dotenv(runbook_dotenv):
    settings = Settings()
    # The file was genuinely read — a Settings field from it took effect.
    assert settings.chunk_max_seconds == 60
    assert settings.topic_granularity == "standard"


def test_get_settings_loads_with_the_runbook_dotenv(runbook_dotenv):
    """The app's own entry point, including its validation and warnings."""
    assert config.get_settings().embedding_model == "all-mpnet-base-v2"


def test_the_dotenv_password_is_not_stored_on_the_model(runbook_dotenv):
    """ignore, not allow: an allowed extra would ride along in model_dump()."""
    settings = Settings()
    dumped = repr(settings.model_dump())
    assert "throughline_password" not in dumped.lower()
    assert PASSWORD not in dumped
    assert PASSWORD not in repr(settings)
    assert not getattr(settings, "model_extra", None), "extras are being retained"


def test_the_dotenv_password_is_not_in_the_settings_summary(runbook_dotenv):
    """settings_summary is what results and CSV exports are stamped with."""
    assert PASSWORD not in repr(settings_summary(Settings()))


def test_authenticated_health_is_200_with_the_runbook_dotenv(runbook_dotenv, monkeypatch, cache_home):
    """The reviewer's reproduction, end to end: gate armed from the process
    environment (as systemd's EnvironmentFile= would export it), the same
    .env in the working directory, real get_settings()."""
    monkeypatch.setenv("THROUGHLINE_PASSWORD", PASSWORD)
    app = create_app(JobManager(runner_factory=FakeRunner))
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 401
        res = client.get("/api/health", headers={"Authorization": f"Bearer {PASSWORD}"})
        assert res.status_code == 200, res.text
        assert PASSWORD not in res.text


def test_an_authenticated_run_can_be_queued_with_the_runbook_dotenv(runbook_dotenv, monkeypatch, cache_home):
    """POST /api/runs builds Settings(**overrides), which read the .env too."""
    monkeypatch.setenv("THROUGHLINE_PASSWORD", PASSWORD)
    manager = JobManager(runner_factory=FakeRunner)
    app = create_app(manager)
    with TestClient(app) as client:
        res = client.post(
            "/api/runs",
            json={"urls": ["https://youtu.be/" + "a" * 11], "settings": {"detect_people": False}},
            headers={"Authorization": f"Bearer {PASSWORD}"},
        )
        assert res.status_code == 201, res.text
        assert manager.join(timeout=30)
