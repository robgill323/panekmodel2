"""Turn PipelineOutputs into the JSON shape the Throughline frontend renders.

Everything numeric here goes through :func:`panekmodel2.sentiment.normalize_sentiment`
so the API, the CSV exports and the Streamlit UI cannot disagree about what a
sentiment score means.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

from ..pipeline import PipelineOutputs, URLOutcome
from ..sentiment import normalize_sentiment, produces_neutral

# Paul Tol derived categorical palette, deuteranopia-checked, as specified by
# the Throughline design. Topic identity only — never used for sentiment.
TOPIC_PALETTE: List[str] = [
    "#332288", "#4477AA", "#88CCEE", "#117733",
    "#999933", "#DDCC77", "#CC6677", "#882255",
    "#AA4499", "#EE8866", "#8A5A2B", "#6E6E6E",
]

# Chunks whose |valence| falls below this read as neutral/procedural in the UI.
NEUTRAL_BAND = 0.08


def topic_color(rank: int) -> str:
    return TOPIC_PALETTE[rank % len(TOPIC_PALETTE)]


def label_from_keywords(topic_id: int, keywords: Sequence[str]) -> str:
    """Human-readable topic label built from the topic's own top terms.

    Deliberately not an invented theme name: the keyword chips are the ground
    truth, and the label must not claim more than they support.
    """
    if topic_id == -1:
        return "Outlier bin"
    picked = [k.replace("_", "-") for k in keywords[:3] if k]
    if not picked:
        return f"Topic {topic_id}"
    return " · ".join(w.capitalize() if w.islower() else w for w in picked)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _controversy(sd: float) -> str:
    if sd > 0.55:
        return "high"
    if sd > 0.42:
        return "moderate"
    return "low"


def valence_histogram(values: Sequence[float], bins: int = 11) -> List[dict]:
    """Bucket valences over −1 … +1 for the batch histogram."""
    counts = [0] * bins
    for v in values:
        idx = int(round((max(-1.0, min(1.0, v)) + 1) / 2 * (bins - 1)))
        counts[idx] += 1
    step = 2.0 / (bins - 1)
    return [{"value": round(-1 + i * step, 2), "count": c} for i, c in enumerate(counts)]


def _entities(outputs: PipelineOutputs, limit: int = 12) -> List[dict]:
    counts: Dict[str, int] = {}
    for names in outputs.people.values():
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [{"name": n, "count": c} for n, c in ranked]


def _video_chunks(outputs: PipelineOutputs) -> List[dict]:
    """Per-chunk records, joined to topic assignment by chunk_index."""
    topic_by_index: Dict[int, tuple] = {}
    if not outputs.topics_df.empty:
        for row in outputs.topics_df.itertuples():
            topic_by_index[int(row.chunk_index)] = (int(row.topic), float(row.prob or 0.0))

    chunks = []
    for i, (chunk, sent) in enumerate(zip(outputs.chunks, outputs.sentiments)):
        topic_id, prob = topic_by_index.get(i, (-1, 0.0))
        valence = normalize_sentiment(sent.label, sent.score)
        chunks.append(
            {
                "index": i,
                "start": round(float(chunk.start), 2),
                "end": round(float(chunk.end), 2),
                "text": chunk.text,
                "topic_id": topic_id,
                "topic_prob": round(prob, 4),
                "valence": round(valence, 4),
                "sentiment_label": sent.label,
                "sentiment_score": round(float(sent.score), 4),
            }
        )
    return chunks


def build_results(
    outputs: List[PipelineOutputs],
    outcomes: List[URLOutcome],
    keywords: Dict[int, List[str]],
    settings_summary: dict,
    run: dict,
) -> dict:
    """Assemble the full results payload for one run."""
    url_by_video = {
        oc.video_id: oc.url for oc in outcomes if oc.status == "analyzed" and oc.video_id
    }

    videos: List[dict] = []
    all_valences: List[float] = []
    all_labels: List[str] = []
    # topic_id → {"valences": [...], "videos": set()}
    topic_acc: Dict[int, dict] = {}
    # topic_id → list of (abs valence, excerpt dict) for representative quotes
    excerpt_pool: Dict[int, List[tuple]] = {}

    for out in outputs:
        chunks = _video_chunks(out)
        valences = [c["valence"] for c in chunks]
        all_valences.extend(valences)
        all_labels.extend(s.label for s in out.sentiments)

        counts: Dict[int, List[float]] = {}
        for c in chunks:
            counts.setdefault(c["topic_id"], []).append(c["valence"])
            acc = topic_acc.setdefault(c["topic_id"], {"valences": [], "videos": set()})
            acc["valences"].append(c["valence"])
            acc["videos"].add(out.video_id)

        n_chunks = len(chunks) or 1
        mix = sorted(
            (
                {
                    "topic_id": tid,
                    "share": len(vals) / n_chunks,
                    "n_chunks": len(vals),
                    "mean_valence": round(_mean(vals), 4),
                    "sd_valence": round(_sd(vals), 4),
                }
                for tid, vals in counts.items()
            ),
            key=lambda m: -m["share"],
        )

        duration = max((c["end"] for c in chunks), default=0.0)
        meta = out.metadata or {}
        video = {
            "video_id": out.video_id,
            "url": url_by_video.get(out.video_id, f"https://www.youtube.com/watch?v={out.video_id}"),
            "title": meta.get("title") or out.video_id,
            "channel": meta.get("channel", ""),
            "published": meta.get("published", ""),
            "duration_s": round(duration, 2),
            "n_chunks": len(chunks),
            "mean_valence": round(_mean(valences), 4),
            "sd_valence": round(_sd(valences), 4),
            "topic_mix": mix,
            "chunks": chunks,
            "entities": _entities(out),
        }
        videos.append(video)

        for c in chunks:
            excerpt_pool.setdefault(c["topic_id"], []).append(
                (
                    abs(c["valence"]),
                    {
                        "text": c["text"][:400],
                        "video_id": out.video_id,
                        "video_title": video["title"],
                        "channel": video["channel"],
                        "start": c["start"],
                        "valence": c["valence"],
                    },
                )
            )

    total_chunks = sum(v["n_chunks"] for v in videos) or 1

    # Rank real topics by prevalence so palette assignment is stable and the
    # most-present topic always gets the first hue.
    real_ids = sorted(
        (tid for tid in topic_acc if tid != -1),
        key=lambda t: (-len(topic_acc[t]["valences"]), t),
    )
    topics: List[dict] = []
    for rank, tid in enumerate(real_ids):
        vals = topic_acc[tid]["valences"]
        sd = _sd(vals)
        kws = keywords.get(tid, [])
        pool = sorted(excerpt_pool.get(tid, []), key=lambda p: -p[0])[:6]
        topics.append(
            {
                "topic_id": tid,
                "label": label_from_keywords(tid, kws),
                "keywords": kws,
                "color": topic_color(rank),
                "n_chunks": len(vals),
                "share": len(vals) / total_chunks,
                "n_videos": len(topic_acc[tid]["videos"]),
                "mean_valence": round(_mean(vals), 4),
                "sd_valence": round(sd, 4),
                "controversy": _controversy(sd),
                "excerpts": [e for _, e in pool],
            }
        )

    outlier_vals = topic_acc.get(-1, {}).get("valences", [])
    outlier = {
        "topic_id": -1,
        "n_chunks": len(outlier_vals),
        "share": len(outlier_vals) / total_chunks,
        "mean_valence": round(_mean(outlier_vals), 4),
    }

    skipped = [
        {
            "url": oc.url,
            "video_id": oc.video_id,
            "reason": oc.reason,
        }
        for oc in outcomes
        if oc.status == "skipped"
    ]

    totals = {
        "n_submitted": len(outcomes),
        "n_analyzed": len(videos),
        "n_skipped": len(skipped),
        "n_chunks": sum(v["n_chunks"] for v in videos),
        "duration_s": round(sum(v["duration_s"] for v in videos), 2),
        "n_topics": len(topics),
        "mean_valence": round(_mean(all_valences), 4),
        "sd_valence": round(_sd(all_valences), 4),
        "pct_negative": (
            round(sum(1 for v in all_valences if v < -NEUTRAL_BAND) / len(all_valences), 4)
            if all_valences else 0.0
        ),
        "pct_positive": (
            round(sum(1 for v in all_valences if v > NEUTRAL_BAND) / len(all_valences), 4)
            if all_valences else 0.0
        ),
        "histogram": valence_histogram(all_valences),
    }

    return {
        "run": run,
        "settings": {
            **settings_summary,
            # The default sentiment model is binary: it never emits a neutral
            # class. Say so, rather than let the UI imply a neutral reading.
            "model_produces_neutral": produces_neutral(all_labels),
        },
        "videos": videos,
        "topics": topics,
        "outlier": outlier,
        "skipped": skipped,
        "totals": totals,
    }
