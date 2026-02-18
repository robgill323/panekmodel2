from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from googleapiclient.discovery import build

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
        if not self.settings.youtube_api_key:
            return {}
        yt = build("youtube", "v3", developerKey=self.settings.youtube_api_key, cache_discovery=False)
        resp = (
            yt.videos()
            .list(part="snippet,contentDetails", id=video_id)
            .execute()
            .get("items", [])
        )
        if not resp:
            return {}
        item = resp[0]["snippet"]
        return {
            "title": item.get("title", ""),
            "channel": item.get("channelTitle", ""),
            "published": item.get("publishedAt", ""),
        }

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

    def summarize_topics(self, outputs: PipelineOutputs, top_n: int = 5) -> List[dict]:
        summary = []
        for topic_id, group in outputs.topics_df.groupby("topic"):
            if topic_id == -1:
                continue
            keywords = [kw for kw, _ in (self.topic_modeler.model.get_topic(topic_id) or [])][:top_n]
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
