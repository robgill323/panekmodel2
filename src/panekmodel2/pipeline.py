from __future__ import annotations

import hashlib
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
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


class VideoCache:
    """Disk cache for per-video transcript, embeddings, and sentiment.

    Cache key covers chunk settings + model names so changing any of those
    automatically invalidates the cache for all videos.

    Topics are intentionally NOT cached because BERTopic fits a shared model
    across the full batch — adding one new video changes everyone's topics.
    Only the inputs to topic modelling (embeddings) and the independent
    per-chunk outputs (sentiments) are safe to cache.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (Path.home() / ".panekmodel2_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, video_id: str, key: str) -> Path:
        return self.cache_dir / f"{video_id}_{key}.pkl"

    def load(self, video_id: str, key: str) -> dict | None:
        p = self._path(video_id, key)
        if p.exists():
            try:
                with open(p, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def save(self, video_id: str, key: str, data: dict) -> None:
        p = self._path(video_id, key)
        with open(p, "wb") as fh:
            pickle.dump(data, fh)

    @staticmethod
    def make_key(
        chunk_max_words: int,
        chunk_max_seconds: int,
        embedding_model: str,
        sentiment_model: str,
    ) -> str:
        raw = f"{chunk_max_words}|{chunk_max_seconds}|{embedding_model}|{sentiment_model}"
        return hashlib.md5(raw.encode()).hexdigest()[:10]


@dataclass
class PipelineOutputs:
    video_id: str
    metadata: Dict[str, str]
    segments: List[TranscriptSegment]
    chunks: List[Chunk]
    topics_df: pd.DataFrame
    sentiments: List[SentimentResult]
    sentiment_rollup: Dict
    # chunk_index → list of PERSON names detected in that chunk
    people: Dict[int, List[str]] = field(default_factory=dict)


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
            embedding_model=self.settings.embedding_model,
            reduce_to=self.settings.topic_reduce_to,
            extra_stop_words=list(self.settings.custom_stopwords),
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

    @staticmethod
    def _detect_people(chunks: List[Chunk]) -> Dict[int, List[str]]:
        """Use NLTK NER to find PERSON entities in each chunk.

        Returns a mapping of chunk_index → [name, ...].  Quietly degrades
        if NLTK data is missing rather than crashing the pipeline.
        """
        try:
            import nltk  # noqa: PLC0415
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("taggers/averaged_perceptron_tagger")
            nltk.data.find("chunkers/maxent_ne_chunker")
            nltk.data.find("corpora/words")
        except LookupError:
            logger.info("NLTK data not fully available; downloading for people detection…")
            import nltk  # noqa: PLC0415
            for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
                        "averaged_perceptron_tagger_eng",
                        "maxent_ne_chunker", "maxent_ne_chunker_tab", "words"):
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

        result: Dict[int, List[str]] = {}
        try:
            import nltk  # noqa: PLC0415
            for idx, chunk in enumerate(chunks):
                names: List[str] = []
                try:
                    tokens = nltk.word_tokenize(chunk.text)
                    tagged = nltk.pos_tag(tokens)
                    tree = nltk.ne_chunk(tagged)
                    seen: set = set()
                    for subtree in tree:
                        if hasattr(subtree, "label") and subtree.label() == "PERSON":
                            name = " ".join(leaf[0] for leaf in subtree.leaves())
                            if name and name not in seen:
                                seen.add(name)
                                names.append(name)
                except Exception:  # noqa: BLE001
                    pass
                if names:
                    result[idx] = names
        except Exception as exc:  # noqa: BLE001
            logger.warning("People detection failed (non-fatal): %s", exc)
        return result

    def run(self, url_or_id: str, detect_people: bool = True) -> PipelineOutputs:
        video_id = extract_video_id(url_or_id)
        cache = VideoCache()
        cache_key = VideoCache.make_key(
            self.settings.chunk_max_words,
            self.settings.chunk_max_seconds,
            self.settings.embedding_model,
            self.settings.sentiment_model,
        )
        cached = cache.load(video_id, cache_key)
        if cached:
            logger.info("%s: loaded transcript + embeddings + sentiments from cache", video_id)
            metadata = cached["metadata"]
            segments = cached["segments"]
            chunks   = cached["chunks"]
            embeddings = cached["embeddings"]
            sentiments = cached["sentiments"]
        else:
            metadata = self.fetch_metadata(video_id)
            segments = self.fetcher.fetch(video_id)
            chunks = chunk_segments(
                segments,
                max_words=self.settings.chunk_max_words,
                max_seconds=self.settings.chunk_max_seconds,
            )
            embeddings = self.topic_modeler.embed_chunks(chunks)
            sentiments = self.sentiment_analyzer.analyze(chunks)
            cache.save(video_id, cache_key, {
                "metadata": metadata,
                "segments": segments,
                "chunks": chunks,
                "embeddings": embeddings,
                "sentiments": sentiments,
            })

        topic_model, topics, probs = self.topic_modeler.fit(chunks, embeddings=embeddings)
        topics_df = self.topic_modeler.topic_dataframe(chunks, topics, probs)

        sentiment_rollup = self.sentiment_analyzer.aggregate(sentiments)
        people = self._detect_people(chunks) if detect_people else {}

        return PipelineOutputs(
            video_id=video_id,
            metadata=metadata,
            segments=segments,
            chunks=chunks,
            topics_df=topics_df,
            sentiments=sentiments,
            sentiment_rollup=sentiment_rollup,
            people=people,
        )

    def run_multi(
        self,
        urls: List[str],
        progress: Callable[[str], None] | None = None,
        detect_people: bool = True,
    ) -> tuple[List["PipelineOutputs"], List[tuple[str, Exception]]]:
        """Run pipeline across multiple videos with a single shared topic model.

        Returns ``(outputs, failures)`` where ``failures`` is a list of
        ``(url, exception)`` pairs for videos that could not be processed.
        Successfully processed videos are always returned even when some fail.
        """
        def _prog(msg: str) -> None:
            if progress:
                progress(msg)
            else:
                logger.info(msg)

        cache = VideoCache()
        cache_key = VideoCache.make_key(
            self.settings.chunk_max_words,
            self.settings.chunk_max_seconds,
            self.settings.embedding_model,
            self.settings.sentiment_model,
        )

        # --- fetch, chunk, embed, and score sentiment per video (cached) ---
        per_video: List[dict] = []
        failures: List[tuple[str, Exception]] = []
        for url in urls:
            try:
                video_id = extract_video_id(url)
            except Exception as exc:  # noqa: BLE001
                _prog(f"⚠ Skipping {url!r}: {exc}")
                failures.append((url, exc))
                continue
            try:
                cached = cache.load(video_id, cache_key)
                if cached:
                    _prog(f"{video_id}: loaded from cache ({len(cached['chunks'])} chunks)")
                    per_video.append(cached)
                    continue

                _prog(f"Fetching transcript: {video_id}")
                metadata = self.fetch_metadata(video_id)
                segments = self.fetcher.fetch(video_id)
                chunks = chunk_segments(
                    segments,
                    max_words=self.settings.chunk_max_words,
                    max_seconds=self.settings.chunk_max_seconds,
                )
                _prog(f"{video_id}: {len(segments)} segments → {len(chunks)} chunks — embedding…")
                embeddings = self.topic_modeler.embed_chunks(chunks)
                _prog(f"{video_id}: running sentiment…")
                sentiments = self.sentiment_analyzer.analyze(chunks)
                entry = {
                    "video_id": video_id,
                    "metadata": metadata,
                    "segments": segments,
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "sentiments": sentiments,
                }
                cache.save(video_id, cache_key, entry)
                per_video.append(entry)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s due to error: %s", video_id, exc)
                _prog(f"⚠ Skipping {video_id}: {exc}")
                failures.append((url, exc))

        if not per_video:
            return [], failures

        # --- combine all chunks + embeddings and fit ONE shared topic model ---
        all_chunks: List[Chunk] = []
        all_embeddings_list = []
        slices: List[tuple[int, int]] = []
        for v in per_video:
            start = len(all_chunks)
            all_chunks.extend(v["chunks"])
            all_embeddings_list.append(v["embeddings"])
            slices.append((start, len(all_chunks)))

        all_embeddings = np.vstack(all_embeddings_list)

        _prog(f"Fitting topic model on {len(all_chunks)} combined chunks…")
        topic_model, topics, probs = self.topic_modeler.fit(all_chunks, embeddings=all_embeddings)
        topics_df_combined = self.topic_modeler.topic_dataframe(all_chunks, topics, probs)

        # sentiments are already computed (cached per video) — just concatenate
        sentiments_combined = [s for v in per_video for s in v["sentiments"]]

        if detect_people:
            _prog("Detecting people…")
            people_combined = self._detect_people(all_chunks)
        else:
            people_combined = {}

        # --- split back per video ---
        outputs: List[PipelineOutputs] = []
        for v, (s, e) in zip(per_video, slices):
            v_chunks = all_chunks[s:e]
            v_topics_df = topics_df_combined.iloc[s:e].reset_index(drop=True)
            v_sentiments = sentiments_combined[s:e]
            v_rollup = self.sentiment_analyzer.aggregate(v_sentiments)
            # Re-index people dict to per-video chunk indices (0-based)
            v_people = {
                i - s: names
                for i, names in people_combined.items()
                if s <= i < e
            }
            outputs.append(
                PipelineOutputs(
                    video_id=v["video_id"],
                    metadata=v["metadata"],
                    segments=v["segments"],
                    chunks=v_chunks,
                    topics_df=v_topics_df,
                    sentiments=v_sentiments,
                    sentiment_rollup=v_rollup,
                    people=v_people,
                )
            )
        _prog("Done.")
        return outputs, failures

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
    def _clean_kws(raw: list, top_n: int, extra_noise: frozenset | None = None) -> list:
        """Strip noise tokens and return up to *top_n* clean keywords."""
        noise = PipelineRunner._NOISE_TOKENS | (extra_noise or frozenset())
        return [
            kw for kw, _ in raw
            if (
                kw.strip("_") != ""
                and "__" not in kw           # catches 'laughter __', 'word__word'
                and kw not in noise
            )
        ][:top_n]

    def summarize_topics(self, outputs: PipelineOutputs, top_n: int = 5) -> List[dict]:
        summary = []
        _extra = frozenset(self.topic_modeler.extra_stop_words)
        for topic_id, group in outputs.topics_df.groupby("topic"):
            if topic_id == -1:
                continue
            # Fetch extra candidates so noise removal doesn't shrink the list.
            raw = self.topic_modeler.model.get_topic(topic_id) or []
            keywords = self._clean_kws(raw, top_n, extra_noise=_extra)
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
