import json
from typing import List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from panekmodel2.config import Settings, get_settings
from panekmodel2.pipeline import PipelineRunner, extract_video_id

st.set_page_config(page_title="PanekModel2", layout="wide")

st.title("YouTube Transcript → Topics → Sentiment")


def parse_urls(raw: str) -> List[str]:
    values = []
    for line in raw.replace(",", "\n").splitlines():
        cleaned = line.strip()
        if cleaned:
            values.append(cleaned)
    return values


def build_timeline_dataframe(outputs, max_time: float) -> pd.DataFrame:
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

    timeline = pd.concat([sentiment_rows, topic_rows], ignore_index=True)

    # Guard against non-finite values that can break rendering.
    timeline = timeline.replace([np.inf, -np.inf], np.nan)
    timeline = timeline.dropna(subset=["start", "end"])
    if timeline.empty:
        return timeline
    timeline["start"] = timeline["start"].astype(float)
    timeline["end"] = timeline["end"].astype(float)
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
        }
        data = base.model_dump()
        data.update(overrides)
        custom_settings = Settings(**data)

        runner = PipelineRunner(custom_settings)
        results = []
        failures = []
        for raw_url in urls:
            with st.spinner(f"Processing {raw_url}"):
                try:
                    outputs = runner.run(raw_url)
                    results.append((raw_url, outputs))
                except Exception as exc:  # noqa: BLE001
                    failures.append((raw_url, exc))

        if failures:
            st.error({"failed": [(u, str(e)) for u, e in failures]})

        if not results:
            st.stop()

        status.success("Done")

        video_choices = []
        for raw_url, outputs in results:
            title = outputs.metadata.get("title") or extract_video_id(raw_url)
            label = f"{title} ({outputs.video_id})"
            video_choices.append({"label": label, "title": title, "raw_url": raw_url, "outputs": outputs})

        selection_label = st.selectbox("Select video", [c["label"] for c in video_choices])
        selected = next(c for c in video_choices if c["label"] == selection_label)
        outputs = selected["outputs"]
        video_title = selected["title"]

        st.markdown(f"### {video_title}")

        tabs = st.tabs(["Overview", "Topics", "Sentiment", "Transcript", "Preview"])

        with tabs[0]:
            st.write(
                {
                    "video_id": outputs.video_id,
                    "segments": len(outputs.segments),
                    "chunks": len(outputs.chunks),
                }
            )
            if outputs.metadata:
                st.write(outputs.metadata)

        with tabs[1]:
            topic_summary = runner.summarize_topics(outputs)
            if topic_summary:
                topic_rows = [
                    {
                        "topic_id": t["topic_id"],
                        "keywords": ", ".join(t["keywords"]),
                        "start": t["representative"]["start"],
                        "end": t["representative"]["end"],
                        "text": t["representative"]["text"],
                    }
                    for t in topic_summary
                ]
                st.dataframe(pd.DataFrame(topic_rows))
            else:
                st.info("No topics found (possibly too few chunks or all outliers). Try reducing chunk sizes.")

        with tabs[2]:
            roll = outputs.sentiment_rollup
            st.metric("Mean", f"{roll['mean']:.3f}")
            st.metric("Median", f"{roll['median']:.3f}")
            st.write({"fractions": roll.get("fractions", {})})

            by_topic = runner.sentiment_by_topic(outputs)
            if by_topic:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "topic": row["topic_id"],
                                "mean": row["sentiment"]["mean"],
                                "median": row["sentiment"]["median"],
                                "frac_positive": row["sentiment"].get("fractions", {}).get("positive", 0.0),
                                "frac_negative": row["sentiment"].get("fractions", {}).get("negative", 0.0),
                            }
                            for row in by_topic
                        ]
                    )
                )

        with tabs[3]:
            with st.expander("Chunks"):
                for idx, chunk in enumerate(outputs.chunks):
                    st.markdown(f"**Chunk {idx}**: {chunk.start:.1f}-{chunk.end:.1f}s")
                    st.write(chunk.text)

            with st.expander("Segments (raw transcript)"):
                st.write(pd.DataFrame([s.__dict__ for s in outputs.segments]))

            st.download_button(
                "Download segments JSON",
                json.dumps([s.__dict__ for s in outputs.segments], indent=2),
                file_name=f"{outputs.video_id}_segments.json",
            )

        with tabs[4]:
            if not outputs.chunks:
                st.info("No chunks to preview.")
            else:
                valid_chunks = [c for c in outputs.chunks if np.isfinite(c.start) and np.isfinite(c.end)]
                if not valid_chunks:
                    st.info("No valid timestamps to preview.")
                    st.stop()
                max_time = max(c.end for c in valid_chunks)
                if not np.isfinite(max_time) or max_time <= 0:
                    st.info("No valid timestamps to preview.")
                else:
                    # Topic jump: pick the earliest chunk per topic to seek quickly.
                    topic_jump = (
                        outputs.topics_df.groupby("topic")["start"].min().reset_index()
                        if not outputs.topics_df.empty
                        else pd.DataFrame({"topic": [-1], "start": [0.0]})
                    )
                    topic_jump = topic_jump.replace([np.inf, -np.inf], np.nan).dropna(subset=["start"])
                    if topic_jump.empty:
                        topic_jump = pd.DataFrame({"topic": [-1], "start": [0.0]})
                    topic_jump["start"] = topic_jump["start"].clip(lower=0.0, upper=max_time)

                    topic_options = [f"Topic {int(row.topic)} @ {row.start:.1f}s" for _, row in topic_jump.iterrows()]
                    default_option = topic_options[0] if topic_options else None

                    col_jump, col_slider = st.columns([1, 2])
                    with col_jump:
                        choice = st.selectbox("Jump to topic", topic_options, index=0 if default_option else None)
                        if choice:
                            chosen_idx = topic_options.index(choice)
                            selected_time = float(topic_jump.iloc[chosen_idx].start)
                            selected_time = max(0.0, min(selected_time, float(max_time)))
                            st.session_state["preview_time"] = selected_time
                    with col_slider:
                        selected_time = float(st.session_state.get("preview_time", 0.0))
                        selected_time = max(0.0, min(selected_time, float(max_time)))
                        step_val = 1.0
                        # Align value to step grid to avoid slider conflicts.
                        selected_time = round(selected_time / step_val) * step_val
                        selected_time = st.slider(
                            "Timestamp (seconds)",
                            min_value=0.0,
                            max_value=float(max_time),
                            value=selected_time,
                            step=step_val if step_val > 0 else 0.1,
                            key="preview_time",
                        )

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
                        "timestamp": f"{selected_time:.1f}s",
                        "chunk": f"{chunk.start:.1f}-{chunk.end:.1f}s",
                        "sentiment": sentiment.label,
                        "topic": topic_id,
                    }
                )

                timeline_df = build_timeline_dataframe(outputs, max_time)
                if timeline_df.empty:
                    st.info("No timeline data to display.")
                else:
                    timeline_chart = (
                        alt.Chart(timeline_df)
                        .mark_rect(height=24)
                        .encode(
                            x=alt.X("start:Q", title="Seconds"),
                            x2="end:Q",
                            y=alt.Y("type:N", title="", sort=["Sentiment", "Topic"]),
                            color=alt.Color("value:N", title=""),
                            tooltip=["type", "value", "start", "end", "text"],
                        )
                        .properties(height=120, width="container")
                    )
                    selected_rule = alt.Chart(pd.DataFrame({"time": [selected_time]})).mark_rule(color="black").encode(
                        x="time:Q"
                    )
                    # Stretch to the full tab width to match the video embed above.
                    st.altair_chart(timeline_chart + selected_rule, use_container_width=True)
