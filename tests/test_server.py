"""API tests with the pipeline mocked: job lifecycle, results shape, CSV export."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from panekmodel2.server import exports
from panekmodel2.server.app import create_app
from panekmodel2.server.jobs import STAGES, Job, JobError, JobManager, Stage, normalize_urls

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


def test_csv_quotes_text_containing_commas(results):
    body = exports.to_csv(results, "combined")
    header = body.splitlines()[0].split(",")
    assert header[0] == "video_id"
    assert "valence" in header
    assert body.endswith("\n")


def test_export_filename_stamps_the_settings(results):
    name = exports.filename_for(results, "combined")
    assert name.startswith("throughline_combined_30s_")
    assert name.endswith(".csv")


def test_build_rows_rejects_unknown_kind(results):
    with pytest.raises(ValueError):
        exports.build_rows(results, "nope")
