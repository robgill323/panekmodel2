import logging
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    youtube_api_key: Optional[str] = Field(
        default=None,
        description=(
            "YouTube Data API key. Used ONLY to fetch video metadata; the API "
            "cannot serve captions for videos you do not own."
        ),
    )
    whisper_model: str = Field(default="small", description="Whisper model size for ASR fallback.")
    use_whisper_fallback: bool = Field(
        default=False, description="Enable Whisper transcription when no captions/transcripts are found."
    )
    embedding_model: str = Field(
        default="all-mpnet-base-v2", description="Sentence embedding model for BERTopic."
    )
    chunk_max_words: int = Field(default=200, description="Maximum words per chunk before splitting.")
    chunk_max_seconds: int = Field(default=60, description="Maximum seconds per chunk before splitting.")
    topic_reduce_to: int = Field(default=10, description="Reduce topics to roughly this count for display.")
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description=(
            "HF model for sentiment-analysis pipeline. The default is 3-class "
            "(positive/neutral/negative) and tuned on speech-like text, so a "
            "procedural passage can actually be scored neutral. "
            "siebert/sentiment-roberta-large-english is binary — it has no "
            "neutral class at all."
        ),
    )
    sentiment_batch_size: int = Field(default=16, description="Batch size for sentiment inference.")
    cuda: bool = Field(default=False, description="Force CUDA usage when available.")
    hf_token: Optional[str] = Field(default=None, description="Hugging Face token for HF Hub downloads.")
    custom_stopwords: List[str] = Field(
        default_factory=list,
        description="Extra words to strip from topic keyword lists (additive to built-in spoken stopwords).",
    )

    class Config:
        env_prefix = ""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Settings that used to do something and now do not. Silently ignoring an env
# var that was previously load-bearing is the kind of quiet behaviour change
# that costs someone an afternoon later, so say so once at startup.
RETIRED_SETTINGS = {
    "GOOGLE_CREDENTIALS_FILE": "the OAuth captions tier was removed in 0.2.0",
    "GOOGLE_TOKEN_FILE": "the OAuth captions tier was removed in 0.2.0",
}


def warn_about_retired_settings() -> List[str]:
    """Log any retired env var that is still set. Returns the names found."""
    found = []
    for name, why in RETIRED_SETTINGS.items():
        if os.environ.get(name):
            found.append(name)
            logger.info("%s is set but %s; the setting is ignored.", name, why)
    return found


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    warn_about_retired_settings()
    # Propagate HF token to env for libraries that read os.environ directly.
    if settings.hf_token:
        # Cover common env var names used by transformers/huggingface_hub/langchain.
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", settings.hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", settings.hf_token)
    return settings
