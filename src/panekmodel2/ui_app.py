import json
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from panekmodel2.config import Settings, get_settings
from panekmodel2.pipeline import PipelineRunner, extract_video_id

st.set_page_config(page_title="PanekModel2", layout="wide")

st.title("YouTube Transcript → Topics → Sentiment")


def fmt_ts(seconds: float) -> str:
    """Convert seconds to YouTube-style timestamp (MM:SS or H:MM:SS)."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def parse_urls(raw: str) -> List[str]:
    values = []
    for line in raw.replace(",", "\n").splitlines():
        cleaned = line.strip()
        if cleaned:
            values.append(cleaned)
    return values


def build_timeline_dataframe(outputs, max_time: float, topic_keywords: dict | None = None) -> pd.DataFrame:
    if outputs.topics_df.empty:
        rows = []
        for idx, (chunk, sent) in enumerate(zip(outputs.chunks, outputs.sentiments)):
            rows.append(
                {
                    "chunk_index": idx,
                    "start": float(chunk.start),
                    "end": float(chunk.end),
                    "text": chunk.text,
                    "topic": -1,
                    "sentiment_label": sent.label,
                    "sentiment_score": sent.score,
                }
            )
        df = pd.DataFrame(rows)
    else:
        df = outputs.topics_df.copy().sort_values("chunk_index").reset_index(drop=True)
        df["sentiment_label"] = [s.label for s in outputs.sentiments]
        df["sentiment_score"] = [s.score for s in outputs.sentiments]

    sentiment_rows = df[["chunk_index", "start", "end", "text", "topic", "sentiment_label", "sentiment_score"]].copy()
    sentiment_rows["type"] = "Sentiment"
    sentiment_rows["value"] = sentiment_rows["sentiment_label"]

    topic_rows = df[["chunk_index", "start", "end", "text", "topic"]].copy()
    topic_rows["type"] = "Topic"
    topic_rows["value"] = "Topic " + topic_rows["topic"].astype(str)
    if topic_keywords:
        topic_rows["hover_label"] = topic_rows["topic"].map(
            lambda t: f"Topic {t}: {', '.join(topic_keywords.get(t, []))}"
            if topic_keywords.get(t) else f"Topic {t}"
        )
    else:
        topic_rows["hover_label"] = topic_rows["value"]
    sentiment_rows["hover_label"] = sentiment_rows["value"]

    timeline = pd.concat([sentiment_rows, topic_rows], ignore_index=True)

    # Guard against non-finite or non-numeric values that can break rendering.
    timeline = timeline.replace([np.inf, -np.inf], np.nan)
    timeline["start"] = pd.to_numeric(timeline["start"], errors="coerce")
    timeline["end"] = pd.to_numeric(timeline["end"], errors="coerce")
    timeline = timeline.dropna(subset=["start", "end"])
    if timeline.empty:
        return timeline
    timeline = timeline[np.isfinite(timeline["start"]) & np.isfinite(timeline["end"])]
    if timeline.empty:
        return timeline
    # Clamp to valid bounds
    timeline["start"] = timeline["start"].clip(lower=0.0, upper=max_time)
    timeline["end"] = timeline["end"].clip(lower=0.0, upper=max_time)
    # Ensure start <= end
    swapped = timeline["start"] > timeline["end"]
    if swapped.any():
        timeline.loc[swapped, ["start", "end"]] = timeline.loc[swapped, ["end", "start"]].values
    # Drop zero-length rows that Altair can choke on
    timeline = timeline[timeline["end"] >= timeline["start"]]
    # Final finite guard
    timeline = timeline[np.isfinite(timeline["start"]) & np.isfinite(timeline["end"])]
    return timeline


with st.form("run-form"):
    urls_raw = st.text_area(
        "YouTube URLs or IDs (one per line)",
        placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
        key="urls",
    )

    col1, col2, col3 = st.columns(3)
    base = get_settings()
    with col1:
        chunk_max_words = st.number_input(
            "Chunk max words", min_value=50, max_value=2000, value=base.chunk_max_words, step=50
        )
        use_whisper = st.checkbox("Enable Whisper fallback", value=base.use_whisper_fallback)
    with col2:
        chunk_max_seconds = st.number_input(
            "Chunk max seconds", min_value=15, max_value=600, value=base.chunk_max_seconds, step=15
        )
        topic_reduce_to = st.number_input(
            "Reduce topics to", min_value=0, max_value=50, value=base.topic_reduce_to, step=1,
            help="0 disables reduction"
        )
    with col3:
        sentiment_model = st.text_input("Sentiment model", value=base.sentiment_model)
        embedding_model = st.text_input("Embedding model", value=base.embedding_model)

    custom_stopwords_raw = st.text_area(
        "Additional stopwords (comma or newline separated)",
        placeholder="e.g. joe, rogan, bro, dude",
        help="Words to remove from topic keyword labels on every run. Added on top of the built-in spoken-word stoplist.",
        height=68,
    )
    detect_people = st.checkbox(
        "Detect people (NER)",
        value=True,
        help="Run NLTK named-entity recognition to find person names in each chunk. "
             "Uncheck to skip this step and keep names in topic keywords.",
    )

    submitted = st.form_submit_button("Run pipeline", type="primary")

if submitted:
    urls = parse_urls(urls_raw)
    if not urls:
        st.error("Please provide at least one YouTube URL or ID.")
    else:
        status = st.empty()
        status.info(f"Running pipeline for {len(urls)} video(s)… this may take a bit while models download.")

        overrides = {
            "chunk_max_words": int(chunk_max_words),
            "chunk_max_seconds": int(chunk_max_seconds),
            "use_whisper_fallback": use_whisper,
            "topic_reduce_to": int(topic_reduce_to),
            "sentiment_model": sentiment_model,
            "embedding_model": embedding_model,
            "custom_stopwords": [
                w.strip().lower()
                for w in custom_stopwords_raw.replace(",", "\n").splitlines()
                if w.strip()
            ],
        }
        data = base.model_dump()
        data.update(overrides)
        custom_settings = Settings(**data)

        runner = PipelineRunner(custom_settings)
        results = []
        failures = []
        try:
            with st.status("Running pipeline…", expanded=True) as run_status:
                def _progress(msg: str) -> None:
                    run_status.write(msg)

                if len(urls) > 1:
                    all_outputs = runner.run_multi(urls, progress=_progress, detect_people=detect_people)
                    results = list(zip(urls, all_outputs))
                else:
                    _progress(f"Fetching transcript: {urls[0]}")
                    outputs = runner.run(urls[0], detect_people=detect_people)
                    _progress("Done.")
                    results = [(urls[0], outputs)]

                run_status.update(label="Pipeline complete!", state="complete")
        except Exception as exc:  # noqa: BLE001
            failures = [(urls, exc)]
            results = []

        if failures:
            st.error({"failed": [(str(u), str(e)) for u, e in failures]})

        if not results:
            st.stop()

        status.success("Done")

        video_choices = []
        for raw_url, outputs in results:
            title = outputs.metadata.get("title") or extract_video_id(raw_url)
            yt_url = f"https://youtube.com/watch?v={outputs.video_id}"
            label = f"{title} ({yt_url})"
            video_choices.append({"label": label, "title": title, "raw_url": raw_url, "outputs": outputs})

        # Persist results so reruns triggered by widget interactions don't lose them.
        st.session_state["video_choices"] = video_choices
        st.session_state["runner"] = runner


if st.session_state.get("video_choices"):
    video_choices = st.session_state["video_choices"]
    runner = st.session_state["runner"]

    # ── Cross-video Topic Map (only shown when 2+ videos) ──────────────────────
    if len(video_choices) > 1:
        with st.expander("Topic Map — cross-video relationships", expanded=True):
            # Collect all topic ids across all videos (excluding outlier -1)
            all_topic_ids = sorted({
                int(tid)
                for vc in video_choices
                for tid in vc["outputs"].topics_df["topic"].unique()
                if int(tid) != -1
            })

            # Build keyword labels for each topic from the shared model
            topic_labels = {}
            _cv_noise = {
                "__", "_", "",
                "laughter", "laughing", "laughs",
                "applause", "clapping",
                "clears throat", "throat",
                "crosstalk", "inaudible",
                "sighs", "sigh", "music", "beep",
            } | set(runner.topic_modeler.extra_stop_words)
            for tid in all_topic_ids:
                kws = [
                    kw for kw, _ in (runner.topic_modeler.model.get_topic(tid) or [])
                    if kw.strip("_") != "" and "__" not in kw and kw not in _cv_noise
                ][:4]
                topic_labels[tid] = f"T{tid}: {', '.join(kws)}" if kws else f"T{tid}"

            # Build matrix: rows = videos, cols = topics, values = chunk counts
            video_names = [vc["title"][:35] for vc in video_choices]
            z = []  # chunk counts
            z_pct = []  # percentage of video's chunks
            for vc in video_choices:
                tdf = vc["outputs"].topics_df
                total_chunks = len(vc["outputs"].chunks)
                row = []
                row_pct = []
                for tid in all_topic_ids:
                    count = int((tdf["topic"] == tid).sum())
                    row.append(count)
                    row_pct.append(round(count / total_chunks * 100, 1) if total_chunks else 0.0)
                z.append(row)
                z_pct.append(row_pct)

            col_labels = [topic_labels[tid] for tid in all_topic_ids]

            # Custom text: show count + % on each cell
            text_vals = [
                [f"{z[r][c]}<br>{z_pct[r][c]}%" if z[r][c] > 0 else "" for c in range(len(all_topic_ids))]
                for r in range(len(video_choices))
            ]

            fig_hm = go.Figure(go.Heatmap(
                z=z,
                x=col_labels,
                y=video_names,
                text=text_vals,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Chunks"),
                hoverongaps=False,
            ))
            fig_hm.update_layout(
                height=max(200, 80 + len(video_choices) * 55),
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(tickangle=-35, title=""),
                yaxis=dict(title=""),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            # Shared-topic summary table with navigation buttons
            shared = [
                {
                    "topic_label": topic_labels[tid],
                    "topic_id": tid,
                    "col_idx": c,
                    "videos": [
                        vc for vc in video_choices
                        if (vc["outputs"].topics_df["topic"] == tid).any()
                    ],
                    "total_chunks": sum(z[r][c] for r, c in enumerate([all_topic_ids.index(tid)] * len(video_choices))),
                }
                for c, tid in enumerate(all_topic_ids)
                if sum(1 for vc in video_choices if (vc["outputs"].topics_df["topic"] == tid).any()) > 1
            ]
            if shared:
                st.markdown("**Topics shared across multiple videos — click a video to jump to its Preview:**")
                for row in shared:
                    cols = st.columns([2] + [1] * len(row["videos"]))
                    cols[0].markdown(f"**{row['topic_label']}** ({row['total_chunks']} chunks)")
                    for ci, vc in enumerate(row["videos"]):
                        tid = row["topic_id"]
                        # Find earliest chunk timestamp for this topic in this video
                        tdf = vc["outputs"].topics_df
                        topic_chunks = tdf[tdf["topic"] == tid].sort_values("start")
                        jump_t = float(topic_chunks.iloc[0]["start"]) if not topic_chunks.empty else 0.0
                        btn_label = f"▶ {vc['title'][:22]} @ {fmt_ts(jump_t)}"
                        if cols[ci + 1].button(btn_label, key=f"nav_{tid}_{vc['outputs'].video_id}"):
                            st.session_state["_nav_video_label"] = vc["label"]
                            st.session_state["_nav_preview_time"] = jump_t
                            st.session_state["last_video_id"] = None  # force preview reset
                            st.rerun()
            else:
                st.info("No topics are shared across multiple videos in this run.")

    st.divider()

    # Apply any topic-click navigation before rendering the selectbox
    if st.session_state.get("_nav_video_label"):
        st.session_state["video_select"] = st.session_state.pop("_nav_video_label")
        st.session_state["_nav_toast"] = True
    if st.session_state.get("_nav_preview_time") is not None:
        st.session_state["preview_time"] = st.session_state.pop("_nav_preview_time")

    selection_label = st.selectbox(
        "Select video",
        [c["label"] for c in video_choices],
        key="video_select",
    )

    if st.session_state.pop("_nav_toast", False):
        st.toast("Topic navigation applied — click the **Preview** tab to watch", icon="▶")
    selected = next(c for c in video_choices if c["label"] == selection_label)
    outputs = selected["outputs"]
    video_title = selected["title"]

    # Reset preview state when switching videos to avoid stale timestamps from prior selections.
    if st.session_state.get("last_video_id") != outputs.video_id:
        st.session_state["preview_time"] = 0.0
        st.session_state["last_video_id"] = outputs.video_id

    st.markdown(f"### {video_title}")

    tabs = st.tabs(["Overview", "Topics", "Topic Sentiment", "Sentiment", "Transcript", "Preview", "Analysis"])

    # ── shared data used by multiple tabs ────────────────────────────────────
    _topic_summary = runner.summarize_topics(outputs)
    _by_topic      = runner.sentiment_by_topic(outputs)
    _kw_map        = {t["topic_id"]: t["keywords"] for t in _topic_summary}

    # Build topic_id → set of person names (if detection ran)
    _topic_people: dict = {}
    if outputs.people:
        for _ci, _names in outputs.people.items():
            _tr = outputs.topics_df[outputs.topics_df["chunk_index"] == _ci]
            _tid = int(_tr.iloc[0]["topic"]) if not _tr.empty else -1
            _topic_people.setdefault(_tid, set()).update(_names)

    # ── Tab 0 · Overview ─────────────────────────────────────────────────────
    with tabs[0]:
        meta = outputs.metadata or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Segments", len(outputs.segments))
        c2.metric("Chunks", len(outputs.chunks))
        c3.metric("Topics", len(_topic_summary))

        if meta:
            st.divider()
            mc1, mc2 = st.columns([2, 1])
            mc1.markdown(f"**Title:** {meta.get('title', '—')}")
            mc1.markdown(f"**Channel:** {meta.get('channel', '—')}")
            mc2.markdown(f"**Published:** {meta.get('published', '—')}")
            mc2.markdown(f"**Video ID:** `{outputs.video_id}`")

        if outputs.people:
            st.divider()
            all_people = sorted({n for ns in outputs.people.values() for n in ns})
            st.markdown(f"**People mentioned ({len(all_people)}):** {', '.join(all_people)}")

    # ── Tab 1 · Topics ───────────────────────────────────────────────────────
    with tabs[1]:
        if _topic_summary:
            for t in _topic_summary:
                kw_str  = ", ".join(t["keywords"]) or "—"
                ts_str  = f"{fmt_ts(t['representative']['start'])} – {fmt_ts(t['representative']['end'])}"
                people_str = ", ".join(sorted(_topic_people.get(t["topic_id"], set()))) or None

                with st.expander(f"**Topic {t['topic_id']}** — {kw_str}   ·   {ts_str}", expanded=False):
                    st.markdown(f"**Keywords:** {kw_str}")
                    st.markdown(f"**Representative chunk:** {ts_str}")
                    st.caption(t["representative"]["text"])
                    if people_str:
                        st.markdown(f"**People:** {people_str}")
        else:
            st.info("No topics found. Try reducing chunk size or disabling topic reduction.")

    # ── Tab 2 · Topic Sentiment ──────────────────────────────────────────────
    with tabs[2]:
        import statistics as _statistics  # noqa: PLC0415
        _sent_scores = [
            s.score * (1 if s.label.lower().startswith("pos") else -1)
            for s in outputs.sentiments
        ]
        _ts_rows = []
        for _row in _by_topic:
            _tid  = _row["topic_id"]
            _kws  = _kw_map.get(_tid, [])
            _frac = _row["sentiment"].get("fractions", {})
            _mask = outputs.topics_df["topic"] == _tid
            _scores = [
                _sent_scores[i]
                for i in outputs.topics_df[_mask]["chunk_index"].tolist()
                if i < len(_sent_scores)
            ]
            _std = round(_statistics.stdev(_scores), 3) if len(_scores) > 1 else 0.0
            _people = ", ".join(sorted(_topic_people.get(_tid, set()))) if _topic_people.get(_tid) else "—"
            _ts_rows.append({
                "Topic":        f"T{_tid}: {', '.join(_kws[:4])}" if _kws else f"T{_tid}",
                "topic_id":     _tid,
                "Keywords":     ", ".join(_kws) or "—",
                "Chunks":       _row["count"],
                "Mean sentiment": round(_row["sentiment"]["mean"], 3),
                "Median":       round(_row["sentiment"]["median"], 3),
                "% Positive":   f"{_frac.get('positive', 0)*100:.0f}%",
                "% Negative":   f"{_frac.get('negative', 0)*100:.0f}%",
                "Controversy ↕": _std,
                "People":       _people,
            })

        if _ts_rows:
            _tsdf = pd.DataFrame(_ts_rows)

            # Summary metrics row
            _overall = outputs.sentiment_rollup
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Overall mean", f"{_overall['mean']:.3f}")
            sc2.metric("Overall median", f"{_overall['median']:.3f}")
            frac = _overall.get("fractions", {})
            sc3.metric("% Positive", f"{frac.get('positive', 0)*100:.0f}%")
            sc4.metric("% Negative", f"{frac.get('negative', 0)*100:.0f}%")
            st.divider()

            # Bar chart: mean sentiment per topic
            _chart_df = _tsdf[_tsdf["topic_id"] != -1].sort_values("Mean sentiment")
            _bar_colors = [
                "#22c55e" if v >= 0.05 else "#ef4444" if v <= -0.05 else "#eab308"
                for v in _chart_df["Mean sentiment"]
            ]
            fig_bar = go.Figure(go.Bar(
                x=_chart_df["Mean sentiment"].tolist(),
                y=_chart_df["Topic"].tolist(),
                orientation="h",
                marker_color=_bar_colors,
                hovertemplate="<b>%{y}</b><br>Mean sentiment: %{x:.3f}<extra></extra>",
            ))
            fig_bar.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
            fig_bar.update_layout(
                height=max(200, len(_chart_df) * 38),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Mean Sentiment Score",
                yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Detail table (drop internal topic_id column)
            st.dataframe(
                _tsdf.drop(columns=["topic_id"]).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No sentiment data available.")

    # ── Tab 3 · Sentiment (overall + timeline) ───────────────────────────────
    with tabs[3]:
        roll = outputs.sentiment_rollup
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("Mean", f"{roll['mean']:.3f}")
        oc2.metric("Median", f"{roll['median']:.3f}")
        _frac2 = roll.get("fractions", {})
        oc3.metric("% Positive", f"{_frac2.get('positive', 0)*100:.0f}%")
        oc4.metric("% Negative", f"{_frac2.get('negative', 0)*100:.0f}%")

        st.divider()

        # Chunk-level sentiment over time
        if outputs.sentiments and outputs.chunks:
            _sent_df = pd.DataFrame([
                {
                    "start": c.start,
                    "mid":   (c.start + c.end) / 2,
                    "score": s.score * (1 if s.label.lower().startswith("pos") else -1),
                    "label": s.label,
                    "text":  c.text[:120],
                }
                for c, s in zip(outputs.chunks, outputs.sentiments)
                if np.isfinite(c.start) and np.isfinite(c.end)
            ])
            if not _sent_df.empty:
                _colors = [
                    "#22c55e" if v >= 0.05 else "#ef4444" if v <= -0.05 else "#eab308"
                    for v in _sent_df["score"]
                ]
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=_sent_df["mid"].tolist(),
                    y=_sent_df["score"].tolist(),
                    mode="markers+lines",
                    marker=dict(color=_colors, size=7),
                    line=dict(color="rgba(100,100,100,0.3)", width=1),
                    hovertemplate="<b>%{x:.0f}s</b>  score: %{y:.3f}<br>%{customdata}<extra></extra>",
                    customdata=_sent_df["text"].tolist(),
                    showlegend=False,
                ))
                fig_line.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                fig_line.update_layout(
                    xaxis_title="Time (seconds)",
                    yaxis_title="Sentiment score",
                    yaxis=dict(range=[-1.05, 1.05]),
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.caption("Sentiment score over time — green = positive, red = negative, amber = neutral.")
                st.plotly_chart(fig_line, use_container_width=True)

    # ── Tab 4 · Transcript ───────────────────────────────────────────────────
    with tabs[4]:
        _search = st.text_input("Search transcript", placeholder="keyword…", key="transcript_search")

        st.markdown("##### Chunks")
        for idx, (chunk, sent) in enumerate(zip(outputs.chunks, outputs.sentiments)):
            _tr2 = outputs.topics_df[outputs.topics_df["chunk_index"] == idx]
            _t_id = int(_tr2.iloc[0]["topic"]) if not _tr2.empty else None
            _kws_inline = ", ".join(_kw_map.get(_t_id, [])[:3]) if _t_id is not None else ""
            _label_color = "🟢" if sent.label.lower().startswith("pos") else "🔴" if sent.label.lower().startswith("neg") else "🟡"

            if _search and _search.lower() not in chunk.text.lower():
                continue

            header = (
                f"{_label_color} **Chunk {idx}** · {fmt_ts(chunk.start)} – {fmt_ts(chunk.end)}"
                + (f" · T{_t_id}: {_kws_inline}" if _t_id is not None else "")
                + f" · {sent.label} ({sent.score:.2f})"
            )
            with st.expander(header, expanded=False):
                st.write(chunk.text)

        st.divider()
        with st.expander("Raw segments"):
            st.dataframe(pd.DataFrame([s.__dict__ for s in outputs.segments]), use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Download segments JSON",
            json.dumps([s.__dict__ for s in outputs.segments], indent=2),
            file_name=f"{outputs.video_id}_segments.json",
        )

    # ── Tab 5 · Preview ──────────────────────────────────────────────────────
    with tabs[5]:
        if not outputs.chunks:
            st.info("No chunks to preview.")
        else:
            valid_chunks = [c for c in outputs.chunks if np.isfinite(c.start) and np.isfinite(c.end)]
            if not valid_chunks:
                st.info("No valid timestamps to preview.")
            else:
                max_time = max(c.end for c in valid_chunks)
                if not np.isfinite(max_time) or max_time <= 0:
                    st.info("No valid timestamps to preview.")
                else:
                    max_time = float(np.clip(max_time, 0.0, 86400.0))
                    topic_jump = (
                        outputs.topics_df.groupby("topic")["start"].min().reset_index()
                        if not outputs.topics_df.empty
                        else pd.DataFrame({"topic": [-1], "start": [0.0]})
                    )
                    topic_jump = topic_jump.replace([np.inf, -np.inf], np.nan).dropna(subset=["start"])
                    if topic_jump.empty:
                        topic_jump = pd.DataFrame({"topic": [-1], "start": [0.0]})
                    topic_jump["start"] = topic_jump["start"].clip(lower=0.0, upper=max_time)

                    topic_options = [f"Topic {int(row.topic)} @ {fmt_ts(row.start)}" for _, row in topic_jump.iterrows()]
                    default_option = topic_options[0] if topic_options else None

                    col_jump, col_slider = st.columns([1, 2])
                    with col_jump:
                        choice = st.selectbox("Jump to topic", topic_options, index=0 if default_option else None)
                        if choice:
                            chosen_idx = topic_options.index(choice)
                            jump_time = float(topic_jump.iloc[chosen_idx].start)
                            jump_time = max(0.0, min(jump_time, float(max_time)))
                            st.session_state["preview_time"] = jump_time
                    with col_slider:
                        selected_time = float(st.session_state.get("preview_time", 0.0))
                        selected_time = max(0.0, min(selected_time, float(max_time)))
                        step_val = 1.0
                        selected_time = round(selected_time / step_val) * step_val
                        selected_time = st.slider(
                            "Timestamp (seconds)",
                            min_value=0.0,
                            max_value=float(max_time),
                            value=selected_time,
                            step=step_val,
                            key="preview_time",
                        )

                    if not np.isfinite(selected_time):
                        selected_time = 0.0

                    def chunk_for_time(time_val: float) -> int:
                        for idx, chk in enumerate(outputs.chunks):
                            if chk.start <= time_val <= chk.end:
                                return idx
                        return len(outputs.chunks) - 1

                    chunk_idx = chunk_for_time(selected_time)
                    chunk = outputs.chunks[chunk_idx]
                    sentiment = outputs.sentiments[chunk_idx]
                    topic_row = outputs.topics_df.loc[outputs.topics_df["chunk_index"] == chunk_idx]
                    topic_id: Optional[int] = None
                    if not topic_row.empty:
                        topic_id = int(topic_row.iloc[0]["topic"])

                    embed_url = (
                        f"https://www.youtube.com/embed/{outputs.video_id}?start={int(selected_time)}&controls=1"
                        "&modestbranding=1"
                    )
                    st.components.v1.html(
                        f"<iframe width='100%' height='360' src='{embed_url}' "
                        "title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; "
                        "encrypted-media; gyroscope; picture-in-picture' allowfullscreen></iframe>",
                        height=380,
                    )

                    st.write(
                        {
                            "timestamp": fmt_ts(selected_time),
                            "chunk": f"{fmt_ts(chunk.start)} – {fmt_ts(chunk.end)}",
                            "sentiment": sentiment.label,
                            "topic": topic_id,
                        }
                    )

                    topic_kw_dict = {
                        t["topic_id"]: t["keywords"]
                        for t in runner.summarize_topics(outputs)
                    }
                    timeline_df = build_timeline_dataframe(outputs, max_time, topic_keywords=topic_kw_dict)
                    timeline_df["start"] = pd.to_numeric(timeline_df["start"], errors="coerce")
                    timeline_df["end"] = pd.to_numeric(timeline_df["end"], errors="coerce")
                    timeline_df = timeline_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["start", "end"])
                    timeline_df = timeline_df[np.isfinite(timeline_df["start"]) & np.isfinite(timeline_df["end"])]
                    timeline_df["start"] = timeline_df["start"].clip(0.0, max_time)
                    timeline_df["end"] = timeline_df["end"].clip(0.0, max_time)
                    if timeline_df.empty or not np.isfinite(max_time):
                        st.info("No timeline data to display.")
                    else:
                        safe_time = float(np.clip(selected_time, 0.0, max_time))
                        row_order = ["Sentiment", "Topic"]
                        # Fixed semantic colors for sentiment labels; topics get
                        # the rotating palette below.
                        color_map: dict = {
                            "positive": "#22c55e",   # vivid green
                            "pos": "#22c55e",
                            "neutral": "#eab308",    # vivid amber
                            "neu": "#eab308",
                            "negative": "#ef4444",   # vivid red
                            "neg": "#ef4444",
                        }
                        palette = [
                            "#3B82F6",  # blue
                            "#8B5CF6",  # violet
                            "#06B6D4",  # cyan
                            "#F97316",  # orange
                            "#EC4899",  # pink
                            "#6366F1",  # indigo
                            "#14B8A6",  # teal
                            "#D946EF",  # fuchsia
                            "#0EA5E9",  # sky
                            "#A855F7",  # purple
                        ]
                        traces = []
                        for row_type in row_order:
                            subset = timeline_df[timeline_df["type"] == row_type]
                            for _, rec in subset.iterrows():
                                val = str(rec["value"])
                                if val not in color_map:
                                    color_map[val] = palette[len(color_map) % len(palette)]
                                hover_label = str(rec.get("hover_label", val))
                                traces.append(
                                    go.Bar(
                                        x=[float(rec["end"]) - float(rec["start"])],
                                        y=[row_type],
                                        base=[float(rec["start"])],
                                        orientation="h",
                                        marker_color=color_map[val],
                                        name=val,
                                        hovertemplate=(
                                            f"<b>{hover_label}</b><br>"
                                            f"{fmt_ts(float(rec['start']))} \u2013 {fmt_ts(float(rec['end']))}<br>"
                                            f"{str(rec.get('text',''))[:120]}<extra></extra>"
                                        ),
                                        showlegend=val not in [t.name for t in traces],
                                    )
                                )
                        fig = go.Figure(traces)
                        fig.add_vline(x=safe_time, line_color="black", line_width=2)
                        fig.update_layout(
                            barmode="overlay",
                            height=140,
                            margin=dict(l=0, r=0, t=0, b=30),
                            xaxis=dict(title="Seconds", range=[0.0, float(max_time)]),
                            yaxis=dict(title=""),
                            legend=dict(orientation="h", y=1.3),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 6 · Analysis ─────────────────────────────────────────────────────
    with tabs[6]:
        model = runner.topic_modeler.model
        if model is None:
            st.info("No topic model available.")
        else:
            # ── 1. Intertopic Distance Map ─────────────────────────────────────
            st.subheader("Intertopic Distance Map")
            st.caption(
                "Topics are reduced to 2D via UMAP. Bubble size reflects the number of chunks "
                "assigned to that topic; proximity indicates semantic similarity."
            )
            try:
                fig_itd = model.visualize_topics()
                fig_itd.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig_itd, use_container_width=True)
            except Exception as _e:
                st.info(f"Intertopic distance map unavailable (need ≥ 2 topics): {_e}")

            st.divider()

            # ── 2. Topic Hierarchy Dendrogram ──────────────────────────────────
            st.subheader("Topic Hierarchy")
            st.caption(
                "Hierarchical clustering of topics by their c-TF-IDF representations. "
                "Topics that merge early (low on the y-axis) are most semantically similar."
            )
            try:
                fig_hier = model.visualize_hierarchy()
                fig_hier.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=max(400, len(model.get_topics()) * 22),
                )
                st.plotly_chart(fig_hier, use_container_width=True)
            except Exception as _e:
                st.info(f"Topic hierarchy unavailable (need ≥ 2 topics): {_e}")


