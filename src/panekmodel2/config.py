import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    youtube_api_key: Optional[str] = Field(
        default=None, description="YouTube Data API key (used for metadata and captions where applicable)."
    )
    google_credentials_file: Optional[str] = Field(
        default=None, description="Path to OAuth client credentials JSON for official captions."
    )
    google_token_file: str = Field(
        default=".youtube_token.json", description="Path to store OAuth token for offline reuse."
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
        default="siebert/sentiment-roberta-large-english",
        description="HF model for sentiment-analysis pipeline.",
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # Propagate HF token to env for libraries that read os.environ directly.
    if settings.hf_token:
        # Cover common env var names used by transformers/huggingface_hub/langchain.
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", settings.hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", settings.hf_token)
    return settings
