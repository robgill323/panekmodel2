import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import certifi

try:
    import yt_dlp  # type: ignore
except Exception:  # noqa: BLE001
    yt_dlp = None
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


class TranscriptFetcher:
    """Fetch transcripts from public transcript tracks, or Whisper audio fallback.

    The YouTube Data API caption-download tier was removed: it requires OAuth
    *ownership* of the video, so it could never serve third-party analysis,
    which is this tool's entire use case.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch(self, video_id: str) -> List[TranscriptSegment]:
        errors = []

        try:
            public = self._fetch_public_transcript(video_id)
            logger.info("Using public transcript")
            return public
        except Exception as exc:  # noqa: BLE001
            logger.warning("Public transcript failed: %s", exc)
            errors.append(exc)

        if self.settings.use_whisper_fallback:
            try:
                whisper_segments = self._fetch_whisper(video_id)
                logger.info("Using Whisper fallback")
                return whisper_segments
            except Exception as exc:  # noqa: BLE001
                logger.error("Whisper fallback failed: %s", exc)
                errors.append(exc)

        raise RuntimeError(f"No transcript available. Errors: {errors}")

    def _fetch_public_transcript(self, video_id: str) -> List[TranscriptSegment]:
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
        except TranscriptsDisabled as exc:  # noqa: BLE001
            raise RuntimeError("Transcripts are disabled for this video") from exc
        except NoTranscriptFound as exc:  # noqa: BLE001
            raise RuntimeError("No transcript found") from exc

        # Prefer manually created English, then generated English, else first available.
        manual_en = next((t for t in transcript_list if not t.is_generated and t.language_code.startswith("en")), None)
        gen_en = next((t for t in transcript_list if t.is_generated and t.language_code.startswith("en")), None)
        fallback = next(iter(transcript_list), None)
        transcript = manual_en or gen_en or fallback
        if transcript is None:
            raise RuntimeError("No usable transcript track")

        entries = transcript.fetch()
        segments: List[TranscriptSegment] = []
        for e in entries:
            if isinstance(e, dict):
                segments.append(
                    TranscriptSegment(text=e.get("text", ""), start=float(e.get("start", 0.0)), duration=float(e.get("duration", 0.0)))
                )
            else:
                text = getattr(e, "text", "") or ""
                start = float(getattr(e, "start", 0.0))
                duration = float(getattr(e, "duration", 0.0)) if hasattr(e, "duration") else float(getattr(e, "end", 0.0)) - start
                segments.append(TranscriptSegment(text=text, start=start, duration=duration))
        return segments

    def _fetch_whisper(self, video_id: str) -> List[TranscriptSegment]:
        import whisper  # local import to avoid heavy load if unused

        yt_url = f"https://www.youtube.com/watch?v={video_id}"
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._download_audio(yt_url, Path(tmp))
            model = whisper.load_model(self.settings.whisper_model)
            result = model.transcribe(str(audio_path), verbose=False)

        segments = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptSegment(
                    text=seg.get("text", "").strip(),
                    start=float(seg.get("start", 0.0)),
                    duration=float(seg.get("end", 0.0) - float(seg.get("start", 0.0))),
                )
            )
        if not segments and "text" in result:
            segments.append(TranscriptSegment(text=result["text"], start=0.0, duration=0.0))
        return segments

    def _download_audio(self, url: str, tmpdir: Path) -> Path:
        if yt_dlp is not None:
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(tmpdir / "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[attr-defined]
                info = ydl.extract_info(url, download=True)
                downloaded = Path(ydl.prepare_filename(info))
                if downloaded.exists():
                    return downloaded
            raise RuntimeError(f"yt-dlp downloaded no audio file for {url}")
        raise RuntimeError(
            "yt-dlp is required for the Whisper audio fallback but is not installed."
        )
