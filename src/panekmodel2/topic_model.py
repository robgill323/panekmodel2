import logging
import re
from typing import List, Tuple

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

from .chunker import Chunk

logger = logging.getLogger(__name__)

# Spoken/conversational filler words that swamp podcast/interview topic labels.
# These are absent from sklearn's written-text stopword list.
_SPOKEN_STOPWORDS: frozenset = frozenset({
    # Discourse fillers
    "like", "yeah", "yep", "yup", "nope", "okay", "ok", "right", "alright",
    "actually", "basically", "literally", "obviously", "certainly", "definitely",
    "absolutely", "exactly", "totally", "honestly", "seriously", "clearly",
    "simply", "generally", "probably", "essentially", "certainly", "apparently",
    # Hedges & softeners
    "just", "kind", "sort", "bit", "little", "lot", "pretty", "quite",
    "really", "very", "sure", "maybe", "perhaps", "guess", "suppose",
    "good", "great", "big", "better", "best", "bad", "different",
    # Conversation mechanics
    "know", "mean", "think", "say", "said", "saying", "look", "see",
    "going", "gonna", "wanna", "gotta", "got", "get", "come",
    "way", "thing", "things", "stuff", "something", "anything", "everything",
    "someone", "anybody", "everybody", "nobody", "people", "person", "guys", "guy",
    "man", "dude", "bro", "hey", "oh", "ah", "um", "uh", "hmm",
    # Contraction fragments produced after stripping apostrophes:
    # don't→don, didn't→didn, doesn't→doesn, wasn't→wasn, isn't→isn,
    # aren't→aren, couldn't→couldn, wouldn't→wouldn, shouldn't→shouldn,
    # I've→ve, I'll→ll, we're→re, that's→thats handled by _clean_text
    "don", "didn", "doesn", "wasn", "isn", "aren", "couldn", "wouldn", "shouldn",
    "ve", "ll", "re", "blah",
    # Common verbs that carry no topical signal
    "make", "made", "making", "use", "used", "using", "want", "wanted",
    "need", "needed", "try", "tried", "trying", "feel", "felt", "feeling",
    "talk", "talking", "talked", "tell", "told", "telling", "ask", "asked",
    "happen", "happened", "happening", "start", "started", "end", "ended",
    "put", "take", "taken", "taking", "give", "given", "giving",
    "go", "goes", "went", "came", "coming",
    "do", "doing", "does", "did", "done",
    # Time/quantity words that add noise
    "time", "times", "year", "years", "day", "days", "week", "weeks",
    "number", "lot", "lots", "bunch", "ago",
    # Interview/podcast specifics
    "interview", "podcast", "episode", "show", "video", "watch",
    "subscribe", "channel", "guest", "host",
})

STOP_WORDS = ENGLISH_STOP_WORDS.union(_SPOKEN_STOPWORDS)


class TopicModeler:
    def __init__(
        self,
        embedding_model: str = "all-mpnet-base-v2",
        reduce_to: int = 10,
        extra_stop_words: List[str] | None = None,
    ):
        self.embedding_model_name = embedding_model
        self.reduce_to = reduce_to
        self.extra_stop_words: List[str] = [w.lower().strip() for w in (extra_stop_words or []) if w.strip()]
        self.model: BERTopic | None = None
        self._embedder: SentenceTransformer | None = None

    def _get_embedder(self) -> SentenceTransformer:
        """Return a cached SentenceTransformer, loading it once on first call."""
        if self._embedder is None:
            logger.info("Loading sentence embedding model: %s", self.embedding_model_name)
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @staticmethod
    def _clean_text(text: str) -> str:
        # Light normalization to help topic quality without harming meaning
        text = text.lower()
        # Preserve hyphenated compound terms (e.g. "anti-inflammatory") as a
        # single underscored token so the vectorizer treats them as one unit
        # rather than two noisy unigrams.
        text = re.sub(r"-", "_", text)
        text = re.sub(r"[^a-z0-9_\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, chunks: List[Chunk]) -> Tuple[BERTopic, List[int], List[float]]:
        texts = [self._clean_text(c.text) for c in chunks]
        embedder = self._get_embedder()

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
            random_state=42,
        )

        # Scale min_cluster_size with corpus: small corpora need tiny clusters,
        # large corpora (~200 chunks) work well with ~5 to avoid too many outliers.
        if n_samples <= 10:
            min_cluster_size = 2
        elif n_samples <= 50:
            min_cluster_size = 3
        else:
            min_cluster_size = max(3, min(5, n_samples // 40))
        # min_samples controls noise sensitivity; scale it with min_cluster_size
        # rather than fixing at 1, which makes clusters too sensitive to noise.
        min_samples = max(1, min_cluster_size - 1)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True)

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
            min_df = 2
            max_df = 0.95
        elif n_samples <= 60:
            min_df = 3
            max_df = 0.85
        else:
            # Scale with corpus size: term must appear in ~3% of chunks, min 3.
            # Filters rare single-video proper nouns while preserving sub-topic vocabulary.
            min_df = max(3, n_samples // 35)
            max_df = 0.85

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
        effective_stop_words = STOP_WORDS.union(self.extra_stop_words)
        vectorizer_model = CountVectorizer(
            stop_words=list(effective_stop_words),
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
        )

        # KeyBERTInspired selects keywords that are maximally similar to the
        # topic embedding rather than just the most frequent c-TF-IDF terms,
        # producing more discriminative and human-readable topic labels.
        representation_model = KeyBERTInspired()
        topic_model = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            language="english",
            verbose=False,
        )
        topics, probs = topic_model.fit_transform(texts)

        reduce_target = self.reduce_to if (self.reduce_to and self.reduce_to < n_samples) else None
        if reduce_target and len(set(t for t in topics if t != -1)) > reduce_target:
            topic_model.reduce_topics(docs=texts, nr_topics=reduce_target)
            # Use stored attributes rather than re-running transform(), which
            # can produce assignments inconsistent with the merged model state.
            topics = topic_model.topics_
            probs = topic_model.probabilities_

        # Reassign outlier chunks (-1) to their nearest topic by embedding
        # similarity. Only worthwhile when there are actual outliers and at
        # least one real topic exists.
        n_outliers = sum(1 for t in topics if t == -1)
        n_real_topics = len(set(t for t in topics if t != -1))
        if n_outliers > 0 and n_real_topics > 0:
            try:
                topics = topic_model.reduce_outliers(
                    texts, topics, strategy="embeddings", threshold=0.4
                )
                topic_model.update_topics(
                    texts, topics=topics, vectorizer_model=vectorizer_model
                )
                logger.info(
                    "Reduced %d outlier chunks into existing topics", n_outliers
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Outlier reduction failed (non-fatal): %s", exc)

        # Refresh topic representations with the effective stop-word list so
        # that any user-supplied words are stripped from final keyword labels.
        try:
            topic_model.update_topics(
                texts,
                topics=topic_model.topics_,
                vectorizer_model=vectorizer_model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("update_topics after stopword refresh failed (non-fatal): %s", exc)

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
            _noise = {
                "__", "_", "",
                "laughter", "laughing", "laughs",
                "applause", "clapping",
                "clears throat", "throat",
                "crosstalk", "inaudible",
                "sighs", "sigh",
                "music", "beep",
            } | set(self.extra_stop_words)
            clean = [
                w for w, _ in words
                if w.strip("_") != "" and "__" not in w and w not in _noise
            ][:top_n]
            topics.append(
                {
                    "topic_id": topic_id,
                    "keywords": clean,
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
