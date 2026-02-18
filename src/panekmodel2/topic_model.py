import logging
import re
from typing import List, Tuple

import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .chunker import Chunk

logger = logging.getLogger(__name__)


class TopicModeler:
    def __init__(self, embedding_model: str = "all-mpnet-base-v2", reduce_to: int = 10):
        self.embedding_model_name = embedding_model
        self.reduce_to = reduce_to
        self.model: BERTopic | None = None

    @staticmethod
    def _clean_text(text: str) -> str:
        # Light normalization to help topic quality without harming meaning
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, chunks: List[Chunk]) -> Tuple[BERTopic, List[int], List[float]]:
        texts = [self._clean_text(c.text) for c in chunks]
        logger.info("Loading sentence embedding model: %s", self.embedding_model_name)
        embedder = SentenceTransformer(self.embedding_model_name)

        logger.info("Fitting BERTopic on %d chunks", len(texts))
        n_samples = len(texts)

        if n_samples == 0:
            raise ValueError("No chunks to model topics for")

        # Avoid UMAP spectral layout errors on very small corpora (k >= N)
        # by constraining neighbors/components to be < number of samples.
        n_neighbors = max(2, min(15, n_samples - 1)) if n_samples > 1 else 2
        n_components = 1 if n_samples <= 3 else min(5, n_samples - 2)
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
        )

        # Make clustering tolerant to tiny datasets.
        min_cluster_size = max(2, min(10, n_samples))
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, prediction_data=True)

        # Vectorizer with stopword removal and bi-grams to reduce stopword-only topics
        # Keep thresholds generous for tiny corpora to avoid max_df/min_df clashes.
        # Keep vectorizer robust on tiny corpora and avoid max_df < min_df errors.
        if n_samples <= 2:
            # Degenerate corpora: allow every token, avoid any filtering.
            min_df = 1
            max_df = 1.0
        elif n_samples <= 5:
            min_df = 1
            max_df = 1.0
        elif n_samples <= 20:
            min_df = 1
            max_df = 0.95
        else:
            min_df = 2
            max_df = 0.9

        # Clamp to valid relationships for the given corpus size.
        # Translate float max_df into counts to ensure validity, clamp aggressively for tiny corpora.
        if isinstance(max_df, float) and max_df * n_samples < float(min_df):
            max_df = 1.0
            min_df = 1
        if isinstance(max_df, int) and max_df < min_df:
            max_df = min_df
        # min_df cannot exceed the number of documents.
        if isinstance(min_df, int) and min_df > n_samples:
            min_df = n_samples
        # Final guard: if after all adjustments max_df < min_df, relax both to 1.
        if (isinstance(max_df, (int, float)) and isinstance(min_df, (int, float)) and max_df < min_df):
            max_df = 1.0
            min_df = 1
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
        )

        topic_model = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language="english",
            verbose=False,
        )
        topics, probs = topic_model.fit_transform(texts)

        reduce_target = self.reduce_to if (self.reduce_to and self.reduce_to < n_samples) else None
        if reduce_target and len(set(t for t in topics if t != -1)) > reduce_target:
            topic_model = topic_model.reduce_topics(documents=texts, probabilities=probs, nr_topics=reduce_target)
            topics, probs = topic_model.transform(texts)

        self.model = topic_model
        return topic_model, topics, probs

    def describe_topics(self, top_n: int = 10) -> List[dict]:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        topics = []
        for topic_id in self.model.get_topics():
            words = self.model.get_topic(topic_id)
            if words is None:
                continue
            topics.append(
                {
                    "topic_id": topic_id,
                    "keywords": [w for w, _ in words[:top_n]],
                }
            )
        return topics

    def topic_dataframe(self, chunks: List[Chunk], topics: List[int], probs: List[float]) -> pd.DataFrame:
        data = []
        for idx, (chunk, topic, prob) in enumerate(zip(chunks, topics, probs)):
            data.append(
                {
                    "chunk_index": idx,
                    "topic": topic,
                    "prob": prob,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                }
            )
        return pd.DataFrame(data)
