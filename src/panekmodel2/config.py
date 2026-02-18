from functools import lru_cache
from typing import Optional

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
    chunk_max_words: int = Field(default=400, description="Maximum words per chunk before splitting.")
    chunk_max_seconds: int = Field(default=90, description="Maximum seconds per chunk before splitting.")
    topic_reduce_to: int = Field(default=10, description="Reduce topics to roughly this count for display.")
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="HF model for sentiment-analysis pipeline.",
    )
    sentiment_batch_size: int = Field(default=16, description="Batch size for sentiment inference.")
    cuda: bool = Field(default=False, description="Force CUDA usage when available.")

    class Config:
        env_prefix = ""
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
