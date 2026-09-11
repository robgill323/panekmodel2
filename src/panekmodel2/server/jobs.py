"""Background run jobs with per-stage and per-URL progress.

One job == one batch of URLs. Jobs live in memory for the life of the process,
matching the app's session-scoped promise: closing it discards the results.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..config import Settings
from ..pipeline import URLOutcome, describe_failure, extract_video_id
from . import results as results_builder
from .runners import get_runner

logger = logging.getLogger(__name__)

# The pipeline's ordered stages, mirrored in the UI's progress screen.
STAGES: List[tuple] = [
    ("fetch", "Fetching transcripts"),
    ("chunk", "Chunking"),
    ("embed", "Embedding"),
    ("topics", "Topic modelling (BERTopic, whole batch)"),
    ("sentiment", "Sentiment"),
]

MAX_URLS = 200

# Runners — and the TopicModeler inside them — are shared between jobs and
# carry per-run mutable state that fit() overwrites. Two fits overlapping would
# corrupt each other regardless of how results are read back, so exactly one
# pipeline execution runs at a time in this process. Queueing is the honest
# behaviour for a single-user local instrument.
_PIPELINE_LOCK = threading.Lock()


class JobError(Exception):
    pass


@dataclass
class Stage:
    key: str
    name: str
    status: str = "queued"  # queued | running | done | failed
    progress: float = 0.0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "status": self.status,
            "progress": round(self.progress, 3),
            "note": self.note,
        }


@dataclass
class Job:
    id: str
    urls: List[str]
    settings: Settings
    detect_people: bool = True
    status: str = "queued"  # queued | running | done | failed
    error: str = ""
    # True while this job is blocked waiting for another run's pipeline to end.
    waiting: bool = False
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    stages: List[Stage] = field(default_factory=list)
    url_states: Dict[str, dict] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)
    results: Optional[dict] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def stage(self, key: str) -> Stage:
        return next(s for s in self.stages if s.key == key)

    def progress_dict(self) -> dict:
        with self._lock:
            return {
                "job_id": self.id,
                "status": self.status,
                "waiting": self.waiting,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_s": round((self.finished_at or time.time()) - (self.started_at or self.created_at), 1),
                "stages": [s.as_dict() for s in self.stages],
                "urls": [self.url_states[u] for u in self.urls],
                "counts": self._counts(),
                "log": self.log[-40:],
            }

    def _counts(self) -> dict:
        states = [self.url_states[u]["status"] for u in self.urls]
        return {
            "total": len(states),
            "ok": sum(1 for s in states if s == "analyzed"),
            "skipped": sum(1 for s in states if s == "skipped"),
            "queued": sum(1 for s in states if s in ("queued", "running")),
        }


def normalize_urls(raw: List[str]) -> List[str]:
    """Trim, drop blanks, de-duplicate preserving order."""
    seen = set()
    out = []
    for value in raw:
        cleaned = (value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


class JobManager:
    """Creates and tracks run jobs. One instance per app."""

    def __init__(self, runner_factory: Callable[[Settings], object] = get_runner):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._runner_factory = runner_factory

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def list_jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def create(self, urls: List[str], settings: Settings, detect_people: bool = True) -> Job:
        cleaned = normalize_urls(urls)
        if not cleaned:
            raise JobError("Provide at least one YouTube URL or video ID.")
        if len(cleaned) > MAX_URLS:
            raise JobError(f"Too many URLs: {len(cleaned)}. This tool runs at most {MAX_URLS} per batch.")

        job = Job(
            id=uuid.uuid4().hex,
            urls=cleaned,
            settings=settings,
            detect_people=detect_people,
            stages=[Stage(key=k, name=n) for k, n in STAGES],
        )
        job.stage("chunk").name = f"Chunking ({settings.chunk_max_seconds} s)"
        job.stage("embed").name = f"Embedding · {settings.embedding_model}"
        job.stage("sentiment").name = f"Sentiment · {settings.sentiment_model}"
        for url in cleaned:
            job.url_states[url] = {
                "url": url,
                "video_id": _safe_video_id(url),
                "title": url,
                "status": "queued",
                "reason": "",
            }
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._execute, args=(job,), daemon=True).start()
        return job

    # ---- execution -----------------------------------------------------

    def _execute(self, job: Job) -> None:
        job.status = "running"
        job.started_at = time.time()
        acquired = False
        try:
            runner = self._runner_factory(job.settings)

            def on_progress(message: str) -> None:
                self._on_progress(job, message)

            # Wait our turn if another run is mid-pipeline, and say so rather
            # than sitting on a "running" label that is doing nothing.
            acquired = _PIPELINE_LOCK.acquire(blocking=False)
            if not acquired:
                job.waiting = True
                self._mark_stage(
                    job, "fetch", "queued", note="waiting for another run to finish"
                )
                job.log.append("Queued behind another run — one pipeline runs at a time.")
                _PIPELINE_LOCK.acquire()
                acquired = True
                job.waiting = False

            self._mark_stage(job, "fetch", "running")
            multi = runner.run_multi(
                job.urls, progress=on_progress, detect_people=job.detect_people
            )
            self._apply_outcomes(job, multi.outcomes)

            for key in ("fetch", "chunk", "embed", "topics", "sentiment"):
                self._mark_stage(job, key, "done", progress=1.0)

            if not multi.outputs:
                reasons = {oc.reason for oc in multi.outcomes if oc.status == "skipped"}
                raise JobError(
                    "No video in this batch could be analyzed. "
                    + (" ".join(sorted(reasons)) if reasons else "")
                )

            job.results = results_builder.build_results(
                outputs=multi.outputs,
                outcomes=multi.outcomes,
                # From the run itself — never read back off the shared runner,
                # whose model may already belong to the next job.
                keywords=multi.keywords,
                settings_summary=settings_summary(job.settings),
                run={
                    "id": job.id,
                    "started_at": job.started_at,
                    "finished_at": time.time(),
                    "n_submitted": len(job.urls),
                },
            )
            job.status = "done"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Run %s failed", job.id)
            job.status = "failed"
            # A JobError is already plain language and already embeds the
            # per-URL reasons; running it through describe_failure() again
            # matched one of those reasons and collapsed the whole message to
            # it, telling someone who submitted 30 URLs about "the video".
            job.error = str(exc) if isinstance(exc, JobError) else (describe_failure(exc) or str(exc))
            for stage in job.stages:
                if stage.status in ("queued", "running"):
                    stage.status = "failed"
        finally:
            if acquired:
                _PIPELINE_LOCK.release()
            job.waiting = False
            job.finished_at = time.time()

    def _apply_outcomes(self, job: Job, outcomes: List[URLOutcome]) -> None:
        for outcome in outcomes:
            state = job.url_states.get(outcome.url)
            if state is None:
                continue
            state["status"] = outcome.status
            state["video_id"] = outcome.video_id
            state["reason"] = outcome.reason

    def _mark_stage(self, job: Job, key: str, status: str, progress: float | None = None, note: str = "") -> None:
        stage = job.stage(key)
        stage.status = status
        if progress is not None:
            stage.progress = progress
        elif status == "running" and stage.progress == 0.0:
            stage.progress = 0.05
        if note:
            stage.note = note

    # The pipeline reports progress as human-readable lines; map the ones that
    # identify a stage transition onto the structured stage list.
    _RE_CHUNKS = re.compile(r"(\w[\w-]*): (\d+) segments → (\d+) chunks")
    _RE_CACHE = re.compile(r"([\w-]+): loaded from cache \((\d+) chunks\)")

    def _on_progress(self, job: Job, message: str) -> None:
        job.log.append(message)
        done_urls = sum(
            1 for s in job.url_states.values() if s["status"] in ("analyzed", "skipped")
        )
        total = len(job.urls) or 1

        if message.startswith("Fetching transcript:"):
            video_id = message.split(":", 1)[1].strip()
            self._touch_url(job, video_id, "running")
            self._mark_stage(job, "fetch", "running", progress=done_urls / total,
                             note=f"{done_urls}/{total} URLs")
        elif self._RE_CACHE.search(message):
            m = self._RE_CACHE.search(message)
            self._touch_url(job, m.group(1), "analyzed", note=f"cached · {m.group(2)} chunks")
            self._mark_stage(job, "fetch", "running", progress=(done_urls + 1) / total)
        elif self._RE_CHUNKS.search(message):
            m = self._RE_CHUNKS.search(message)
            self._touch_url(job, m.group(1), "running", note=f"{m.group(3)} chunks")
            self._mark_stage(job, "fetch", "done", progress=1.0)
            self._mark_stage(job, "chunk", "running", progress=(done_urls + 1) / total)
            self._mark_stage(job, "embed", "running", progress=(done_urls + 0.5) / total,
                             note=f"{done_urls}/{total} videos embedded")
        elif "running sentiment" in message:
            self._mark_stage(job, "sentiment", "running", progress=(done_urls + 0.5) / total)
        elif message.startswith("Fitting topic model"):
            for key in ("fetch", "chunk", "embed", "sentiment"):
                self._mark_stage(job, key, "done", progress=1.0)
            self._mark_stage(job, "topics", "running", progress=0.5, note=message)
        elif message.startswith("Detecting people"):
            self._mark_stage(job, "topics", "done", progress=1.0)
        elif message.startswith("⚠ Skipping"):
            self._mark_stage(job, "fetch", "running", progress=(done_urls + 1) / total)

    def _touch_url(self, job: Job, video_id: str, status: str, note: str = "") -> None:
        for state in job.url_states.values():
            if state["video_id"] == video_id:
                if state["status"] not in ("analyzed", "skipped"):
                    state["status"] = status
                if note:
                    state["reason"] = note
                return


def _safe_video_id(url: str) -> Optional[str]:
    try:
        return extract_video_id(url)
    except Exception:  # noqa: BLE001
        return None


def settings_summary(settings: Settings) -> dict:
    """The settings stamped onto results and exports."""
    return {
        "chunk_max_seconds": settings.chunk_max_seconds,
        "chunk_max_words": settings.chunk_max_words,
        "embedding_model": settings.embedding_model,
        "sentiment_model": settings.sentiment_model,
        "topic_reduce_to": settings.topic_reduce_to,
        "use_whisper_fallback": settings.use_whisper_fallback,
    }
