from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd
from googleapiclient.discovery import build

try:
    import yt_dlp as _yt_dlp  # type: ignore
except Exception:
    _yt_dlp = None

from .chunker import Chunk, chunk_segments
from .config import Settings, get_settings
from .sentiment import SentimentAnalyzer, SentimentResult
from .topic_model import TopicModeler
from .transcript_fetcher import TranscriptFetcher, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutputs:
    video_id: str
    metadata: Dict[str, str]
    segments: List[TranscriptSegment]
    chunks: List[Chunk]
    topics_df: pd.DataFrame
    sentiments: List[SentimentResult]
    sentiment_rollup: Dict


def extract_video_id(url_or_id: str) -> str:
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)
    match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video id")


class PipelineRunner:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.fetcher = TranscriptFetcher(self.settings)
        self.topic_modeler = TopicModeler(
            embedding_model=self.settings.embedding_model, reduce_to=self.settings.topic_reduce_to
        )
        self.sentiment_analyzer = SentimentAnalyzer(
            model_name=self.settings.sentiment_model,
            batch_size=self.settings.sentiment_batch_size,
            use_cuda=self.settings.cuda,
        )

    def fetch_metadata(self, video_id: str) -> Dict[str, str]:
        # Try YouTube Data API first
        if self.settings.youtube_api_key:
            try:
                yt = build("youtube", "v3", developerKey=self.settings.youtube_api_key, cache_discovery=False)
                resp = (
                    yt.videos()
                    .list(part="snippet,contentDetails", id=video_id)
                    .execute()
                    .get("items", [])
                )
                if resp:
                    item = resp[0]["snippet"]
                    meta = {
                        "title": item.get("title", ""),
                        "channel": item.get("channelTitle", ""),
                        "published": item.get("publishedAt", ""),
                    }
                    if meta["title"]:
                        return meta
            except Exception as exc:  # noqa: BLE001
                logger.warning("YouTube API metadata fetch failed: %s", exc)

        # Fallback: use yt-dlp (no API key required)
        if _yt_dlp is not None:
            try:
                url = f"https://www.youtube.com/watch?v={video_id}"
                ydl_opts = {"quiet": True, "skip_download": True, "no_warnings": True}
                with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title", ""),
                    "channel": info.get("uploader", ""),
                    "published": info.get("upload_date", ""),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("yt-dlp metadata fetch failed: %s", exc)

        return {}

    def run(self, url_or_id: str) -> PipelineOutputs:
        video_id = extract_video_id(url_or_id)
        metadata = self.fetch_metadata(video_id)
        segments = self.fetcher.fetch(video_id)
        chunks = chunk_segments(
            segments,
            max_words=self.settings.chunk_max_words,
            max_seconds=self.settings.chunk_max_seconds,
        )

        topic_model, topics, probs = self.topic_modeler.fit(chunks)
        topics_df = self.topic_modeler.topic_dataframe(chunks, topics, probs)

        sentiments = self.sentiment_analyzer.analyze(chunks)
        sentiment_rollup = self.sentiment_analyzer.aggregate(sentiments)

        return PipelineOutputs(
            video_id=video_id,
            metadata=metadata,
            segments=segments,
            chunks=chunks,
            topics_df=topics_df,
            sentiments=sentiments,
            sentiment_rollup=sentiment_rollup,
        )

    def run_multi(self, urls: List[str], progress: Callable[[str], None] | None = None) -> List["PipelineOutputs"]:
        """Run pipeline across multiple videos with a single shared topic model."""
        def _prog(msg: str) -> None:
            if progress:
                progress(msg)
            else:
                logger.info(msg)

        # --- fetch & chunk each video independently ---
        per_video: List[dict] = []
        for url in urls:
            video_id = extract_video_id(url)
            _prog(f"Fetching transcript: {video_id}")
            metadata = self.fetch_metadata(video_id)
            segments = self.fetcher.fetch(video_id)
            chunks = chunk_segments(
                segments,
                max_words=self.settings.chunk_max_words,
                max_seconds=self.settings.chunk_max_seconds,
            )
            per_video.append(
                {"video_id": video_id, "metadata": metadata, "segments": segments, "chunks": chunks}
            )
            _prog(f"{video_id}: {len(segments)} segments → {len(chunks)} chunks")

        # --- combine all chunks and fit ONE shared topic model ---
        all_chunks: List[Chunk] = []
        slices: List[tuple[int, int]] = []
        for v in per_video:
            start = len(all_chunks)
            all_chunks.extend(v["chunks"])
            slices.append((start, len(all_chunks)))

        _prog(f"Fitting topic model on {len(all_chunks)} combined chunks…")
        topic_model, topics, probs = self.topic_modeler.fit(all_chunks)
        topics_df_combined = self.topic_modeler.topic_dataframe(all_chunks, topics, probs)

        _prog(f"Running sentiment on {len(all_chunks)} chunks…")
        sentiments_combined = self.sentiment_analyzer.analyze(all_chunks)

        # --- split back per video ---
        outputs: List[PipelineOutputs] = []
        for v, (s, e) in zip(per_video, slices):
            v_chunks = all_chunks[s:e]
            v_topics_df = topics_df_combined.iloc[s:e].reset_index(drop=True)
            v_sentiments = sentiments_combined[s:e]
            v_rollup = self.sentiment_analyzer.aggregate(v_sentiments)
            outputs.append(
                PipelineOutputs(
                    video_id=v["video_id"],
                    metadata=v["metadata"],
                    segments=v["segments"],
                    chunks=v_chunks,
                    topics_df=v_topics_df,
                    sentiments=v_sentiments,
                    sentiment_rollup=v_rollup,
                )
            )
        _prog("Done.")
        return outputs

    # Tokens that carry no semantic meaning and should never appear in topic
    # keyword lists.  Includes SentencePiece underscore artifacts, transcript
    # stage-directions, and common audience-reaction words.
    _NOISE_TOKENS: frozenset = frozenset({
        "__", "_", "",
        # transcript stage-directions / audience reactions
        "laughter", "laughing", "laughs",
        "applause", "clapping",
        "clears throat", "throat",
        "crosstalk", "inaudible",
        "sighs", "sigh",
        "music", "beep",
    })

    @staticmethod
    def _clean_kws(raw: list, top_n: int) -> list:
        """Strip noise tokens and return up to *top_n* clean keywords."""
        return [
            kw for kw, _ in raw
            if (
                kw.strip("_") != ""
                and "__" not in kw           # catches 'laughter __', 'word__word'
                and kw not in PipelineRunner._NOISE_TOKENS
            )
        ][:top_n]

    def summarize_topics(self, outputs: PipelineOutputs, top_n: int = 5) -> List[dict]:
        summary = []
        for topic_id, group in outputs.topics_df.groupby("topic"):
            if topic_id == -1:
                continue
            # Fetch extra candidates so noise removal doesn't shrink the list.
            raw = self.topic_modeler.model.get_topic(topic_id) or []
            keywords = self._clean_kws(raw, top_n)
            sample_chunk = group.sort_values("prob", ascending=False).iloc[0]
            summary.append(
                {
                    "topic_id": topic_id,
                    "keywords": keywords,
                    "representative": {
                        "start": float(sample_chunk["start"]),
                        "end": float(sample_chunk["end"]),
                        "text": sample_chunk["text"][:400],
                    },
                }
            )
        summary.sort(key=lambda x: x["topic_id"])
        return summary

    def sentiment_by_topic(self, outputs: PipelineOutputs) -> List[dict]:
        results = []
        df = outputs.topics_df.copy()
        df["sentiment"] = outputs.sentiments
        grouped = df.groupby("topic")
        for topic_id, group in grouped:
            sentiments = [s for s in group["sentiment"].tolist()]
            agg = self.sentiment_analyzer.aggregate(sentiments)
            results.append({"topic_id": topic_id, "sentiment": agg, "count": len(group)})
        return results
