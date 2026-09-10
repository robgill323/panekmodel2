"""FastAPI app serving the Throughline UI and the pipeline API.

Binds 127.0.0.1 by default: this is a local research instrument with no
authentication, and it must not be reachable from the network by accident.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..chunker import words_for_seconds
from ..config import Settings, get_settings
from . import exports
from .jobs import JobError, JobManager, settings_summary

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class RunSettings(BaseModel):
    """Settings a run may override. Everything else comes from the env/.env."""

    chunk_max_seconds: Optional[int] = Field(default=None, ge=5, le=600)
    chunk_max_words: Optional[int] = Field(default=None, ge=20, le=2000)
    embedding_model: Optional[str] = None
    sentiment_model: Optional[str] = None
    topic_reduce_to: Optional[int] = Field(default=None, ge=0, le=50)
    use_whisper_fallback: Optional[bool] = None
    detect_people: bool = True


class RunRequest(BaseModel):
    urls: List[str] = Field(default_factory=list)
    settings: RunSettings = Field(default_factory=RunSettings)


def _resolve_settings(overrides: RunSettings) -> Settings:
    base = get_settings().model_dump()
    for field in (
        "chunk_max_seconds",
        "chunk_max_words",
        "embedding_model",
        "sentiment_model",
        "topic_reduce_to",
        "use_whisper_fallback",
    ):
        value = getattr(overrides, field)
        if value is not None:
            base[field] = value
    # The UI picks chunk size in seconds. Derive the paired word cap unless the
    # caller set one explicitly, so the chosen duration is what actually governs
    # chunking rather than being cut short by a fixed word limit.
    if overrides.chunk_max_seconds is not None and overrides.chunk_max_words is None:
        base["chunk_max_words"] = words_for_seconds(overrides.chunk_max_seconds)
    return Settings(**base)


def create_app(manager: JobManager | None = None) -> FastAPI:
    app = FastAPI(title="Throughline", version="2.0", docs_url="/api/docs")
    app.state.jobs = manager or JobManager()

    @app.get("/api/health")
    def health() -> dict:
        settings = get_settings()
        return {"status": "ok", "settings": settings_summary(settings)}

    @app.get("/api/defaults")
    def defaults() -> dict:
        """Defaults the New Run screen pre-fills from."""
        return settings_summary(get_settings())

    @app.post("/api/runs", status_code=201)
    def start_run(request: RunRequest = Body(...)) -> dict:
        settings = _resolve_settings(request.settings)
        try:
            job = app.state.jobs.create(
                request.urls, settings, detect_people=request.settings.detect_people
            )
        except JobError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"job_id": job.id, "status": job.status, "n_urls": len(job.urls)}

    @app.get("/api/runs/{job_id}")
    def run_progress(job_id: str) -> dict:
        return _job(app, job_id).progress_dict()

    @app.get("/api/runs/{job_id}/results")
    def run_results(job_id: str) -> dict:
        job = _job(app, job_id)
        if job.status == "failed":
            raise HTTPException(status_code=409, detail=job.error or "Run failed.")
        if job.results is None:
            raise HTTPException(status_code=409, detail="Run is still in progress.")
        return job.results

    @app.get("/api/runs/{job_id}/export/{kind}.csv")
    def run_export(job_id: str, kind: str) -> PlainTextResponse:
        job = _job(app, job_id)
        if job.results is None:
            raise HTTPException(status_code=409, detail="Run has no results yet.")
        if kind not in exports.EXPORT_KINDS:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown export {kind!r}; expected one of {list(exports.EXPORT_KINDS)}",
            )
        body = exports.to_csv(job.results, kind)
        filename = exports.filename_for(job.results, kind)
        return PlainTextResponse(
            body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if STATIC_DIR.is_dir():
        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


def _job(app: FastAPI, job_id: str):
    try:
        return app.state.jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No run {job_id!r}") from exc


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn  # noqa: PLC0415

    uvicorn.run(create_app(), host=host, port=port, reload=reload, log_level="info")
