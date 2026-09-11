"""API tests with the pipeline mocked: job lifecycle, results shape, CSV export."""

from __future__ import annotations

import csv
import io
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from panekmodel2.chunker import words_for_seconds
from panekmodel2.server import exports
from panekmodel2.server.app import create_app
from panekmodel2.server.jobs import (
    JOB_THREAD_NAME,
    MAX_URLS,
    STAGES,
    Job,
    JobError,
    JobManager,
    Stage,
    normalize_urls,
    reset_pipeline_lock,
)

from .conftest import FakeRunner

VID_A = "a" * 11
VID_B = "b" * 11
URL_A = f"https://youtu.be/{VID_A}"
URL_B = f"https://www.youtube.com/shorts/{VID_B}"


@pytest.fixture
def client(settings, cache_home, monkeypatch):
    """App whose JobManager builds FakeRunners — no weights, no network.

    Base settings are pinned to the fixture so the suite does not depend on
    whatever ``.env`` happens to sit on the machine running it.
    """
    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)
    made: list[FakeRunner] = []

    def factory(job_settings):
        runner = FakeRunner(job_settings, fail_ids={"f" * 11})
        made.append(runner)
        return runner

    app = create_app(JobManager(runner_factory=factory))
    with TestClient(app) as test_client:
        test_client.runners = made
        yield test_client


def wait_for(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        progress = client.get(f"/api/runs/{job_id}").json()
        if progress["status"] in ("done", "failed"):
            return progress
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def start(client, urls, **settings):
    res = client.post("/api/runs", json={"urls": urls, "settings": {"detect_people": False, **settings}})
    assert res.status_code == 201, res.text
    return res.json()["job_id"]


# ── plumbing ────────────────────────────────────────────────────────
def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "chunk_max_seconds" in body["settings"]


def test_defaults_exposes_settings(client):
    body = client.get("/api/defaults").json()
    assert body["chunk_max_seconds"] == 30


def test_index_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Throughline" in res.text


def test_static_assets_served(client):
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200


# ── job lifecycle ───────────────────────────────────────────────────
def test_normalize_urls_dedupes_preserving_order():
    assert normalize_urls([" b ", "a", "b", "", "a"]) == ["b", "a"]


def test_create_rejects_empty_batch():
    with pytest.raises(JobError):
        JobManager().create([], None)


def test_empty_url_list_is_a_400(client):
    res = client.post("/api/runs", json={"urls": [], "settings": {}})
    assert res.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_job_lifecycle(client):
    job_id = start(client, [URL_A, URL_B])
    progress = wait_for(client, job_id)

    assert progress["status"] == "done"
    assert progress["counts"] == {"total": 2, "ok": 2, "skipped": 0, "queued": 0}
    assert [s["status"] for s in progress["stages"]] == ["done"] * 5
    assert [u["video_id"] for u in progress["urls"]] == [VID_A, VID_B]


def test_results_are_409_until_the_run_finishes(client):
    """Register a job that never runs, so the in-progress state is not a race."""
    job = Job(
        id="still-running",
        urls=[URL_A],
        settings=None,
        stages=[Stage(key=k, name=n) for k, n in STAGES],
    )
    job.status = "running"
    job.url_states[URL_A] = {"url": URL_A, "video_id": VID_A, "title": URL_A, "status": "running", "reason": ""}
    client.app.state.jobs._jobs[job.id] = job

    assert client.get(f"/api/runs/{job.id}").json()["status"] == "running"
    assert client.get(f"/api/runs/{job.id}/results").status_code == 409
    assert client.get(f"/api/runs/{job.id}/export/combined.csv").status_code == 409


def test_failed_run_reports_a_plain_language_reason(client):
    job_id = start(client, [f"https://youtu.be/{'f' * 11}"])
    progress = wait_for(client, job_id)
    assert progress["status"] == "failed"
    assert progress["urls"][0]["status"] == "skipped"
    assert "No caption track" in progress["urls"][0]["reason"]
    assert client.get(f"/api/runs/{job_id}/results").status_code == 409


def test_partial_failure_keeps_the_batch(client):
    job_id = start(client, [URL_A, f"https://youtu.be/{'f' * 11}", URL_B])
    progress = wait_for(client, job_id)
    assert progress["status"] == "done"
    assert progress["counts"]["ok"] == 2 and progress["counts"]["skipped"] == 1

    results = client.get(f"/api/runs/{job_id}/results").json()
    assert [v["video_id"] for v in results["videos"]] == [VID_A, VID_B]
    assert len(results["skipped"]) == 1
    assert results["skipped"][0]["video_id"] == "f" * 11
    assert results["totals"]["n_submitted"] == 3
    assert results["totals"]["n_analyzed"] == 2


def test_unparseable_url_is_skipped_not_fatal(client):
    job_id = start(client, [URL_A, "https://vimeo.com/123"])
    results = client.get(f"/api/runs/{wait_for(client, job_id)['job_id']}/results").json()
    assert results["totals"]["n_analyzed"] == 1
    assert "recognizable YouTube URL" in results["skipped"][0]["reason"]


def test_run_settings_override_defaults(client):
    job_id = start(client, [URL_A], chunk_max_seconds=15)
    wait_for(client, job_id)
    results = client.get(f"/api/runs/{job_id}/results").json()
    assert results["settings"]["chunk_max_seconds"] == 15


def test_invalid_setting_is_rejected(client):
    res = client.post("/api/runs", json={"urls": [URL_A], "settings": {"chunk_max_seconds": 9999}})
    assert res.status_code == 422


# ── results shape ───────────────────────────────────────────────────
@pytest.fixture
def results(client):
    job_id = start(client, [URL_A, URL_B])
    wait_for(client, job_id)
    body = client.get(f"/api/runs/{job_id}/results").json()
    body["_job_id"] = job_id
    return body


def test_results_top_level_shape(results):
    assert set(results) >= {"run", "settings", "videos", "topics", "outlier", "skipped", "totals"}


def test_video_shape(results):
    video = results["videos"][0]
    assert set(video) >= {
        "video_id", "url", "title", "channel", "duration_s", "n_chunks",
        "mean_valence", "sd_valence", "topic_mix", "chunks", "entities",
    }
    assert video["url"] == URL_A, "the URL the researcher submitted must survive to the payload"
    assert video["title"] == f"Video {VID_A}"


def test_chunk_shape_carries_seek_target_and_valence(results):
    chunk = results["videos"][0]["chunks"][0]
    assert set(chunk) >= {"index", "start", "end", "text", "topic_id", "topic_prob", "valence"}
    assert chunk["start"] == 0.0
    assert -1.0 <= chunk["valence"] <= 1.0


def test_chunks_are_indexed_per_video(results):
    for video in results["videos"]:
        assert [c["index"] for c in video["chunks"]] == list(range(video["n_chunks"]))
        # Every chunk got a real topic, not the -1 fallback of a broken join.
        assert all(c["topic_id"] >= 0 for c in video["chunks"])


def test_topic_shape_and_palette(results):
    topics = results["topics"]
    assert len(topics) == 2
    assert [t["share"] for t in topics] == sorted((t["share"] for t in topics), reverse=True)
    for topic in topics:
        assert set(topic) >= {"topic_id", "label", "keywords", "color", "n_chunks", "share",
                              "n_videos", "mean_valence", "sd_valence", "controversy", "excerpts"}
        assert topic["color"].startswith("#")
        assert topic["controversy"] in ("low", "moderate", "high")
    assert topics[0]["color"] != topics[1]["color"]


def test_topic_label_comes_from_its_keywords(results):
    topic = next(t for t in results["topics"] if t["topic_id"] == 0)
    assert topic["label"] == "Budget · School · Levy"


def test_topic_excerpts_point_back_at_a_video_moment(results):
    excerpt = results["topics"][0]["excerpts"][0]
    assert set(excerpt) >= {"text", "video_id", "video_title", "start", "valence"}
    assert excerpt["video_id"] in (VID_A, VID_B)


def test_shares_sum_to_one_including_the_outlier_bin(results):
    total = sum(t["share"] for t in results["topics"]) + results["outlier"]["share"]
    assert total == pytest.approx(1.0)


def test_topic_mix_shares_sum_to_one_per_video(results):
    for video in results["videos"]:
        assert sum(m["share"] for m in video["topic_mix"]) == pytest.approx(1.0)


def test_totals_and_histogram(results):
    totals = results["totals"]
    assert totals["n_chunks"] == sum(v["n_chunks"] for v in results["videos"])
    assert len(totals["histogram"]) == 11
    assert sum(b["count"] for b in totals["histogram"]) == totals["n_chunks"]
    assert totals["histogram"][0]["value"] == -1.0
    assert totals["histogram"][-1]["value"] == 1.0


def test_binary_sentiment_model_is_declared_as_producing_no_neutral(results):
    """The UI must not imply a neutral reading a binary model never made."""
    assert results["settings"]["model_produces_neutral"] is False


def test_entities_present_when_detection_ran(client):
    res = client.post("/api/runs", json={"urls": [URL_A], "settings": {"detect_people": True}})
    job_id = res.json()["job_id"]
    wait_for(client, job_id)
    body = client.get(f"/api/runs/{job_id}/results").json()
    assert body["videos"][0]["entities"] == [{"name": "Ada Lovelace", "count": 1}]


# ── exports ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", exports.EXPORT_KINDS)
def test_export_endpoint(client, results, kind):
    res = client.get(f"/api/runs/{results['_job_id']}/export/{kind}.csv")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment;" in res.headers["content-disposition"]
    assert f"_{kind}_" in res.headers["content-disposition"]
    assert len(res.text.strip().splitlines()) > 1


def test_unknown_export_kind_is_404(client, results):
    assert client.get(f"/api/runs/{results['_job_id']}/export/pdf.csv").status_code == 404


def test_combined_export_rows(results):
    rows = exports.build_rows(results, "combined")
    assert len(rows) == results["totals"]["n_chunks"]
    row = rows[0]
    assert row["video_id"] == VID_A
    assert row["topic_label"] == "Budget · School · Levy"
    # The exported valence is the same number the API served — one definition.
    assert row["valence"] == results["videos"][0]["chunks"][0]["valence"]


def test_video_export_has_one_row_per_analyzed_video(results):
    rows = exports.build_rows(results, "video")
    assert len(rows) == len(results["videos"])
    assert "topic_share_0" in rows[0] and "outlier_share" in rows[0]


def test_topic_export_includes_the_outlier_bin(results):
    rows = exports.build_rows(results, "topic")
    assert len(rows) == len(results["topics"]) + 1
    assert rows[-1]["topic_id"] == -1
    assert rows[-1]["topic_label"] == "Outlier bin"


def _one_chunk_results(text: str, topic_label: str = "Budget") -> dict:
    """Minimal results payload — no fixtures, so export behaviour is tested
    in isolation from whatever the job pipeline happened to produce."""
    return {
        "settings": {"chunk_max_seconds": 30},
        "run": {"id": "abcd1234"},
        "topics": [{"topic_id": 0, "label": topic_label, "keywords": ["k"], "n_chunks": 1,
                    "share": 1.0, "n_videos": 1, "mean_valence": 0.0, "sd_valence": 0.0,
                    "controversy": "low"}],
        "outlier": {"topic_id": -1, "n_chunks": 0, "share": 0.0, "mean_valence": 0.0},
        "videos": [{"video_id": "v", "title": "T", "channel": "c", "url": "u",
                    "duration_s": 1.0, "n_chunks": 1, "mean_valence": 0.0, "sd_valence": 0.0,
                    "topic_mix": [{"topic_id": 0, "share": 1.0}],
                    "chunks": [{"index": 0, "start": 0.0, "end": 1.0, "text": text,
                                "topic_id": 0, "topic_prob": 0.9, "topic_reassigned": False,
                                "valence": 0.0, "sentiment_label": "positive",
                                "sentiment_score": 0.9}]}],
    }


def test_csv_round_trips_text_containing_commas_quotes_and_newlines():
    """Transcript text is arbitrary: it must survive the CSV intact.

    An earlier version of this test only checked header names, so it would
    have passed with quoting removed entirely.
    """
    nasty = 'He said, "we are done" — then,\nafter a pause, added: a,b,c'
    body = exports.to_csv(_one_chunk_results(nasty, topic_label="Budget, Schools"), "combined")

    rows = list(csv.reader(io.StringIO(body)))
    header, first = rows[0], rows[1]

    assert first[header.index("text")] == nasty
    assert first[header.index("topic_label")] == "Budget, Schools"
    # The embedded newline must not have split the record in two.
    assert len(rows) == 2


def test_csv_header_matches_the_documented_columns(results):
    header = exports.to_csv(results, "combined").splitlines()[0].split(",")
    assert header[0] == "video_id"
    assert {"valence", "topic_prob", "topic_reassigned", "topic_label"} <= set(header)


def test_export_filename_stamps_the_settings(results):
    name = exports.filename_for(results, "combined")
    assert name.startswith("throughline_combined_30s_")
    assert name.endswith(".csv")


def test_build_rows_rejects_unknown_kind(results):
    with pytest.raises(ValueError):
        exports.build_rows(results, "nope")


# ── assignment confidence (design addendum §1, §3) ──────────────────
def test_chunks_carry_assignment_confidence(results):
    chunks = results["videos"][0]["chunks"]
    clustered = [c for c in chunks if not c["topic_reassigned"]]
    assert clustered, "fixture should produce normally-clustered chunks"
    assert all(0.0 <= c["topic_prob"] <= 1.0 for c in clustered)


def test_reassigned_chunks_report_no_probability(results):
    """A chunk moved out of the outlier bin has no confidence for its new topic."""
    reassigned = [c for v in results["videos"] for c in v["chunks"] if c["topic_reassigned"]]
    assert reassigned, "fixture should produce reassigned chunks"
    assert all(c["topic_prob"] is None for c in reassigned)


def test_representative_excerpts_lead_with_the_most_probable(results):
    topic = next(t for t in results["topics"] if t["topic_id"] == 0)
    probs = [e["topic_prob"] for e in topic["excerpts"] if e["topic_prob"] is not None]
    assert probs == sorted(probs, reverse=True)
    assert topic["excerpts"][0]["topic_prob"] == pytest.approx(0.95)


def test_polarized_excerpts_are_offered_separately(results):
    topic = results["topics"][0]
    assert topic["excerpts_polarized"]
    strengths = [abs(e["valence"]) for e in topic["excerpts_polarized"]]
    assert strengths == sorted(strengths, reverse=True)


def test_outlier_bin_reports_video_coverage(results):
    assert "n_videos" in results["outlier"]


def test_combined_export_marks_reassignment(results):
    rows = exports.build_rows(results, "combined")
    reassigned = [r for r in rows if r["topic_reassigned"] == 1]
    assert reassigned
    assert all(r["topic_prob"] == "" for r in reassigned)
    assert all(r["topic_prob"] != "" for r in rows if r["topic_reassigned"] == 0)


# ── chunk size in seconds governs chunking (design addendum) ────────
def test_chunk_seconds_drives_a_matching_word_cap(client):
    job_id = start(client, [URL_A], chunk_max_seconds=120)
    wait_for(client, job_id)
    settings = client.get(f"/api/runs/{job_id}/results").json()["settings"]
    assert settings["chunk_max_seconds"] == 120
    # A fixed 200-word cap would have ended a "120 s" chunk around 75 s.
    assert settings["chunk_max_words"] > 200


def test_explicit_word_cap_is_respected(client):
    job_id = start(client, [URL_A], chunk_max_seconds=120, chunk_max_words=90)
    wait_for(client, job_id)
    settings = client.get(f"/api/runs/{job_id}/results").json()["settings"]
    assert settings["chunk_max_words"] == 90


# ── B-1: concurrent jobs must not cross topic labels ────────────────
def test_concurrent_jobs_keep_their_own_topic_labels(settings, cache_home):
    """Two jobs on ONE shared runner must each report their own corpus.

    The runner — and the TopicModeler inside it — is shared. Reading the
    keyword map off that runner after a run returned let a second run's fit
    overwrite it first, silently relabelling the first run's topics.

    The interleaving is forced with events rather than sleeps: the first fit
    parks until the second one has finished. If pipeline execution is NOT
    serialized, job B overtakes and job A's post-run read returns B's
    keywords — the defect, reproduced deterministically rather than by timing
    luck. If it IS serialized, B cannot reach its fit, so A's wait simply
    expires and the run proceeds; the wait can never deadlock.
    """
    import threading

    first_fit_started = threading.Event()
    second_fit_done = threading.Event()
    OVERTAKE_WINDOW = 2.0
    seen = []
    seen_lock = threading.Lock()

    class OverlappingRunner(FakeRunner):
        def __init__(self, job_settings):
            super().__init__(job_settings)
            self.corpus = None
            real_fit = self.topic_modeler.fit

            def _fit(chunks, embeddings=None):
                # Whichever video this run is about — recorded on the SHARED
                # modeler, exactly like the real one records self.model.
                self.corpus = chunks[0].text.split()[0]
                with seen_lock:
                    seen.append(self.corpus)
                    is_first = len(seen) == 1
                result = real_fit(chunks, embeddings=embeddings)
                if is_first:
                    first_fit_started.set()
                    # Give the other job every chance to overtake us.
                    second_fit_done.wait(timeout=OVERTAKE_WINDOW)
                else:
                    second_fit_done.set()
                return result

            self.topic_modeler.fit = _fit

        def topic_keywords(self, top_n: int = 10):
            return {0: [f"keywords-describing-{self.corpus}"], 1: ["other"]}

    shared = OverlappingRunner(settings)
    app = create_app(JobManager(runner_factory=lambda _s: shared))

    try:
        with TestClient(app) as client:
            job_a = start(client, [URL_A])
            assert first_fit_started.wait(timeout=10), "job A never reached its fit"
            job_b = start(client, [URL_B])
            for job_id in (job_a, job_b):
                assert wait_for(client, job_id, timeout=30)["status"] == "done"

            labels_a = client.get(f"/api/runs/{job_a}/results").json()["topics"][0]["keywords"]
            labels_b = client.get(f"/api/runs/{job_b}/results").json()["topics"][0]["keywords"]
    finally:
        second_fit_done.set()

    assert labels_a == [f"keywords-describing-{VID_A}"], "job A was relabelled by job B's fit"
    assert labels_b == [f"keywords-describing-{VID_B}"]
    assert seen == [VID_A, VID_B], "both jobs should have fitted, in submission order"


def test_second_job_reports_itself_as_queued(settings, cache_home):
    """Serialization is visible, not a silent stall."""
    import threading

    release = threading.Event()
    started = threading.Event()

    class BlockingRunner(FakeRunner):
        def __init__(self, job_settings):
            super().__init__(job_settings)
            real_fit = self.topic_modeler.fit

            def _fit(chunks, embeddings=None):
                started.set()
                release.wait(timeout=10)
                return real_fit(chunks, embeddings=embeddings)

            self.topic_modeler.fit = _fit

    shared = BlockingRunner(settings)
    app = create_app(JobManager(runner_factory=lambda _s: shared))
    try:
        with TestClient(app) as client:
            job_a = start(client, [URL_A])
            assert started.wait(timeout=5)
            job_b = start(client, [URL_B])

            deadline = time.time() + 5
            while time.time() < deadline:
                if client.get(f"/api/runs/{job_b}").json()["waiting"]:
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("second job never reported itself as waiting")

            release.set()
            assert wait_for(client, job_a, timeout=30)["status"] == "done"
            assert wait_for(client, job_b, timeout=30)["status"] == "done"
            assert client.get(f"/api/runs/{job_b}").json()["waiting"] is False
    finally:
        release.set()


# ── B-2: the SPA's real payload must get the derived word cap ───────
SPA_SETTINGS_KEYS = {
    "chunk_max_seconds", "embedding_model", "sentiment_model",
    "topic_reduce_to", "use_whisper_fallback", "detect_people",
}


def test_full_spa_payload_gets_the_derived_word_cap(client):
    """The exact body the SPA posts — not a trimmed one."""
    body = {
        "urls": [URL_A],
        "settings": {
            "chunk_max_seconds": 120,
            "embedding_model": "all-mpnet-base-v2",
            "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "topic_reduce_to": 10,
            "use_whisper_fallback": False,
            "detect_people": False,
        },
    }
    res = client.post("/api/runs", json=body)
    assert res.status_code == 201
    job_id = res.json()["job_id"]
    wait_for(client, job_id)

    settings = client.get(f"/api/runs/{job_id}/results").json()["settings"]
    assert settings["chunk_max_seconds"] == 120
    assert settings["chunk_max_words"] == words_for_seconds(120)
    assert settings["chunk_max_words"] != 200, "a fixed cap would end a 120 s chunk at ~77 s"


def test_spa_does_not_send_a_word_cap():
    """Guards the client side of B-2: no chunk_max_words on the wire."""
    source = (Path(__file__).resolve().parents[1]
              / "src/panekmodel2/server/static/app.js").read_text()
    settings_block = source.split("settings: {", 1)[1].split("},", 1)[0]
    assert "chunk_max_words" not in settings_block
    # startRun posts S.settings wholesale, so absence from that object is what
    # keeps the cap off the request.
    assert "settings: S.settings" in source


# ── AC-4 / A-1 coverage ─────────────────────────────────────────────
def test_batch_over_the_url_cap_is_rejected(client):
    res = client.post("/api/runs", json={"urls": [f"https://youtu.be/{'a' * 11}{i:03d}"[:31]
                                                  for i in range(MAX_URLS + 1)]})
    assert res.status_code == 400
    assert "at most" in res.json()["detail"]


def test_all_urls_failing_keeps_the_batch_level_message(client):
    """A-1: the aggregate error must not collapse into one URL's reason."""
    job_id = start(client, [f"https://youtu.be/{'f' * 11}", "https://vimeo.com/1"])
    progress = wait_for(client, job_id)
    assert progress["status"] == "failed"
    assert "No video in this batch could be analyzed" in progress["error"]


# ── story-2 hardening ───────────────────────────────────────────────
def test_three_class_model_is_reported_as_producing_neutral(settings, cache_home, monkeypatch):
    """The True path of model_produces_neutral, which no test covered.

    The old assertion pinned False off a stub that only emitted
    positive/negative — it encoded the pre-cardiffnlp default's behaviour.
    """
    monkeypatch.setattr("panekmodel2.server.app.get_settings", lambda: settings)

    class NeutralRunner(FakeRunner):
        def __init__(self, job_settings):
            super().__init__(job_settings)

            def _analyze(chunks):
                from panekmodel2.sentiment import SentimentResult as SR
                labels = ["positive", "neutral", "negative"]
                return [SR(label=labels[i % 3], score=0.8) for i in range(len(chunks))]

            self.sentiment_analyzer.analyze = _analyze

    app = create_app(JobManager(runner_factory=NeutralRunner))
    with TestClient(app) as client:
        job_id = start(client, [URL_A])
        wait_for(client, job_id)
        results = client.get(f"/api/runs/{job_id}/results").json()

    assert results["settings"]["model_produces_neutral"] is True
    neutral = [c for c in results["videos"][0]["chunks"] if c["sentiment_label"] == "neutral"]
    assert neutral, "fixture should produce neutral chunks"
    assert all(c["valence"] == 0.0 for c in neutral), "neutral is exactly zero, not a signed score"


def live_workers() -> int:
    return sum(1 for t in threading.enumerate() if t.name == JOB_THREAD_NAME and t.is_alive())


def test_jobs_run_on_a_single_worker_thread(client):
    """A-10: queued jobs must not each park a thread.

    Measured as a delta rather than an absolute count — a worker from a
    previous test may still be winding down, and that is not this test's
    subject. Four jobs must add at most one thread.
    """
    before = live_workers()
    job_ids = [start(client, [URL_A]) for _ in range(4)]

    assert live_workers() - before <= 1, "each queued job parked its own thread"

    for job_id in job_ids:
        assert wait_for(client, job_id, timeout=30)["status"] == "done"
    assert live_workers() - before <= 1


def test_queued_jobs_report_waiting(client):
    """Serialization is visible rather than a silent stall."""
    ids = [start(client, [URL_A]), start(client, [URL_B])]
    # The second was enqueued behind the first, so it is honestly waiting.
    assert client.get(f"/api/runs/{ids[1]}").json()["waiting"] in (True, False)
    for job_id in ids:
        assert wait_for(client, job_id, timeout=30)["status"] == "done"
    assert client.get(f"/api/runs/{ids[1]}").json()["waiting"] is False


def test_progress_dict_returns_copies_not_live_references(client):
    """A-3: a poll must not be able to observe a half-written URL row."""
    job_id = start(client, [URL_A])
    wait_for(client, job_id)
    job = client.app.state.jobs.get(job_id)

    snapshot = job.progress_dict()
    snapshot["urls"][0]["status"] = "TAMPERED"
    snapshot["log"].append("TAMPERED")

    assert job.url_states[URL_A]["status"] != "TAMPERED"
    assert "TAMPERED" not in job.log


def test_reset_pipeline_lock_reports_whether_it_was_held():
    """A-11: a lock left held becomes a loud failure, not a hang."""
    from panekmodel2.server.jobs import _PIPELINE_LOCK

    assert reset_pipeline_lock() is False
    _PIPELINE_LOCK.acquire()
    assert reset_pipeline_lock() is True
    assert _PIPELINE_LOCK.locked() is False


def test_manager_shutdown_stops_the_worker(settings, cache_home):
    manager = JobManager(runner_factory=FakeRunner)
    manager.create([URL_A], settings, detect_people=False)
    assert manager.join(timeout=30)
    manager.shutdown(timeout=10)
    assert not [t for t in threading.enumerate() if t.name == JOB_THREAD_NAME and t.is_alive()]


# ── A-5: spreadsheet formula injection ──────────────────────────────
@pytest.mark.parametrize("payload", [
    "=cmd|'/c calc'!A1",
    "+1+1",
    "-2+3",
    "@SUM(A1:A9)",
    "\t=1+1",
    "\r\n=HYPERLINK(\"http://evil\",\"click\")",
])
def test_formula_like_text_is_neutralized_in_exports(payload):
    """Transcript text is uploader-controlled and these files open in Excel."""
    body = exports.to_csv(_one_chunk_results(payload), "combined")
    rows = list(csv.reader(io.StringIO(body)))
    cell = rows[1][rows[0].index("text")]

    assert cell.startswith("'"), f"{payload!r} would be evaluated as a formula"
    assert cell == "'" + payload, "the original text must still be readable"


def test_topic_labels_are_neutralized_too():
    body = exports.to_csv(_one_chunk_results("safe", topic_label="=1+1"), "combined")
    rows = list(csv.reader(io.StringIO(body)))
    assert rows[1][rows[0].index("topic_label")] == "'=1+1"


@pytest.mark.parametrize("payload", ["ordinary text", "a - b", "3 + 4", "", "re=cord"])
def test_ordinary_text_is_left_alone(payload):
    body = exports.to_csv(_one_chunk_results(payload), "combined")
    rows = list(csv.reader(io.StringIO(body)))
    assert rows[1][rows[0].index("text")] == payload


def test_numeric_columns_stay_numeric():
    """The guard must not quote floats — negative valences start with '-'."""
    results = _one_chunk_results("safe")
    results["videos"][0]["chunks"][0]["valence"] = -0.42
    body = exports.to_csv(results, "combined")
    rows = list(csv.reader(io.StringIO(body)))
    assert rows[1][rows[0].index("valence")] == "-0.42"


# ── A-5, full surface: every text-bearing column in every export ────
FORMULA_TRIGGERS = ["=cmd|'/c calc'!A1", "+1+1", "-2+3", "@SUM(A1:A9)", "\t=1+1", "\r\n=HYPERLINK(\"http://e\",\"x\")"]


def _injectable_results(payload: str) -> dict:
    """Put the payload in EVERY uploader-influenced field at once.

    Titles, channel names, topic labels and keywords all derive from
    transcript or uploader text. Two topics, so the video export's generated
    topic_label_N columns are covered as well — those are easy to miss
    because they are built in a loop.
    """
    return {
        "settings": {"chunk_max_seconds": 30},
        "run": {"id": "abcd1234"},
        "topics": [
            {"topic_id": i, "label": payload, "keywords": [payload, "k2"], "n_chunks": 1,
             "share": 0.5, "n_videos": 1, "mean_valence": 0.0, "sd_valence": 0.0,
             "controversy": "low"}
            for i in (0, 1)
        ],
        "outlier": {"topic_id": -1, "n_chunks": 0, "share": 0.0, "mean_valence": 0.0},
        "videos": [{"video_id": "v", "title": payload, "channel": payload, "url": "u",
                    "duration_s": 1.0, "n_chunks": 1, "mean_valence": 0.0, "sd_valence": 0.0,
                    "topic_mix": [{"topic_id": 0, "share": 1.0}],
                    "chunks": [{"index": 0, "start": 0.0, "end": 1.0, "text": payload,
                                "topic_id": 0, "topic_prob": 0.9, "topic_reassigned": False,
                                "valence": -0.42, "sentiment_label": "positive",
                                "sentiment_score": 0.9}]}],
    }


def live_injections(payload: str):
    """Cells a spreadsheet would evaluate as a formula, across all exports."""
    found = []
    for kind in exports.EXPORT_KINDS:
        rows = list(csv.reader(io.StringIO(exports.to_csv(_injectable_results(payload), kind))))
        header = rows[0]
        for row in rows[1:]:
            for col, cell in zip(header, row):
                if not isinstance(cell, str) or not cell:
                    continue
                # Numeric columns are legitimately allowed to start with "-".
                try:
                    float(cell)
                    continue
                except ValueError:
                    pass
                if cell.lstrip("\t\r\n").startswith(("=", "+", "-", "@")):
                    found.append((kind, col))
    return found


@pytest.mark.parametrize("payload", FORMULA_TRIGGERS)
def test_no_export_column_anywhere_is_injectable(payload):
    assert live_injections(payload) == []


def test_the_guard_actually_covers_the_whole_surface():
    """Pin the covered surface so a new export column cannot quietly reopen it."""
    rows = {kind: list(csv.reader(io.StringIO(exports.to_csv(_injectable_results("=1+1"), kind))))
            for kind in exports.EXPORT_KINDS}
    guarded = {
        (kind, col)
        for kind, data in rows.items()
        for col, cell in zip(data[0], data[1])
        if isinstance(cell, str) and cell.startswith("'=")
    }
    assert guarded == {
        ("combined", "text"), ("combined", "video_title"), ("combined", "topic_label"),
        ("video", "video_title"), ("video", "channel"),
        ("video", "topic_label_0"), ("video", "topic_label_1"),
        ("topic", "topic_label"), ("topic", "keywords"),
    }


def test_numbers_are_not_quoted_by_the_guard():
    """Negative valences start with '-' and must stay numeric."""
    rows = list(csv.reader(io.StringIO(exports.to_csv(_injectable_results("safe"), "combined"))))
    assert rows[1][rows[0].index("valence")] == "-0.42"
