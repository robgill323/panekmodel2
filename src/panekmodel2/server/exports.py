"""CSV exports for a completed run.

Column sets mirror the schema documented on the Export screen, so what a
researcher previews is exactly what lands in the file.
"""

from __future__ import annotations

import csv
import io
from typing import Dict, List

EXPORT_KINDS = ("combined", "video", "topic")

# Excel, Sheets and LibreOffice treat a cell starting with any of these as a
# formula. Transcript text and topic labels are uploader-controlled, and this
# tool's entire output is meant to be opened in a spreadsheet, so a chunk that
# happens to start "=" must not become executable on open.
_FORMULA_TRIGGERS = ("=", "+", "-", "@")
# Leading control characters are stripped by spreadsheets before the trigger
# check, so they can smuggle a formula past a naive first-character test.
_FORMULA_STRIPPED = ("\t", "\r", "\n")


def sanitize_cell(value):
    """Neutralize spreadsheet formula injection, leaving the text readable.

    Prefixes a single quote, which spreadsheets consume as "treat as text".
    Non-strings pass through untouched so numeric columns stay numeric.
    """
    if not isinstance(value, str) or not value:
        return value
    probe = value.lstrip("".join(_FORMULA_STRIPPED))
    if probe.startswith(_FORMULA_TRIGGERS):
        return "'" + value
    return value


def _topic_lookup(results: dict) -> Dict[int, dict]:
    return {t["topic_id"]: t for t in results["topics"]}


def combined_rows(results: dict) -> List[dict]:
    topics = _topic_lookup(results)
    rows = []
    for video in results["videos"]:
        for chunk in video["chunks"]:
            topic = topics.get(chunk["topic_id"])
            rows.append(
                {
                    "video_id": video["video_id"],
                    "video_title": video["title"],
                    "chunk_index": chunk["index"],
                    "start_s": chunk["start"],
                    "end_s": chunk["end"],
                    "text": chunk["text"],
                    "topic_id": chunk["topic_id"],
                    "topic_label": topic["label"] if topic else "Outlier bin",
                    # Blank when the chunk was reassigned out of the outlier
                    # bin: no probability for its current topic exists.
                    "topic_prob": "" if chunk["topic_prob"] is None else chunk["topic_prob"],
                    "topic_reassigned": int(chunk["topic_reassigned"]),
                    "valence": chunk["valence"],
                    "sentiment_label": chunk["sentiment_label"],
                    "sentiment_score": chunk["sentiment_score"],
                }
            )
    return rows


def video_rows(results: dict) -> List[dict]:
    topics = _topic_lookup(results)
    ordered_ids = [t["topic_id"] for t in results["topics"]]
    rows = []
    for video in results["videos"]:
        share_by_topic = {m["topic_id"]: m["share"] for m in video["topic_mix"]}
        row = {
            "video_id": video["video_id"],
            "video_title": video["title"],
            "channel": video["channel"],
            "url": video["url"],
            "duration_s": video["duration_s"],
            "n_chunks": video["n_chunks"],
            "mean_valence": video["mean_valence"],
            "sd_valence": video["sd_valence"],
        }
        for tid in ordered_ids:
            label = topics[tid]["label"]
            row[f"topic_share_{tid}"] = round(share_by_topic.get(tid, 0.0), 4)
            row[f"topic_label_{tid}"] = label
        row["outlier_share"] = round(share_by_topic.get(-1, 0.0), 4)
        rows.append(row)
    return rows


def topic_rows(results: dict) -> List[dict]:
    rows = [
        {
            "topic_id": t["topic_id"],
            "topic_label": t["label"],
            "keywords": "|".join(t["keywords"]),
            "n_chunks": t["n_chunks"],
            "share": round(t["share"], 4),
            "n_videos": t["n_videos"],
            "mean_valence": t["mean_valence"],
            "sd_valence": t["sd_valence"],
            "controversy": t["controversy"],
        }
        for t in results["topics"]
    ]
    outlier = results["outlier"]
    rows.append(
        {
            "topic_id": -1,
            "topic_label": "Outlier bin",
            "keywords": "",
            "n_chunks": outlier["n_chunks"],
            "share": round(outlier["share"], 4),
            "n_videos": 0,
            "mean_valence": outlier["mean_valence"],
            "sd_valence": 0.0,
            "controversy": "",
        }
    )
    return rows


_BUILDERS = {"combined": combined_rows, "video": video_rows, "topic": topic_rows}


def build_rows(results: dict, kind: str) -> List[dict]:
    if kind not in _BUILDERS:
        raise ValueError(f"Unknown export kind {kind!r}; expected one of {list(EXPORT_KINDS)}")
    return _BUILDERS[kind](results)


def to_csv(results: dict, kind: str) -> str:
    """Render one export as a UTF-8 CSV string with a single header row."""
    rows = build_rows(results, kind)
    buf = io.StringIO()
    if not rows:
        return ""
    # Union of keys preserving first-seen order: video rows can differ in the
    # per-topic share columns when a run found no topics at all.
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {key: sanitize_cell(value) for key, value in row.items()} for row in rows
    )
    return buf.getvalue()


def filename_for(results: dict, kind: str) -> str:
    """Stamp the run settings into the filename so a figure is traceable."""
    settings = results.get("settings", {})
    stamp = str(results.get("run", {}).get("id", "run"))[:8]
    seconds = settings.get("chunk_max_seconds", "na")
    return f"throughline_{kind}_{seconds}s_{stamp}.csv"
