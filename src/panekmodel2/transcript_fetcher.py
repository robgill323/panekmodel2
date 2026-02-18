import io
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import srt
import certifi
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from pytube import YouTube

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
    """Fetch transcripts via official captions, public transcripts, or Whisper fallback."""

    scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch(self, video_id: str, prefer_official: bool = True) -> List[TranscriptSegment]:
        errors = []

        if prefer_official and self.settings.google_credentials_file:
            try:
                captions = self._fetch_official_captions(video_id)
                if captions:
                    logger.info("Using official captions")
                    return captions
            except Exception as exc:  # noqa: BLE001
                logger.warning("Official captions failed: %s", exc)
                errors.append(exc)

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

    def _fetch_official_captions(self, video_id: str) -> List[TranscriptSegment]:
        youtube = self._youtube_client()
        if youtube is None:
            raise RuntimeError("Google credentials not configured")

        try:
            response = youtube.captions().list(part="id, snippet", videoId=video_id).execute()
        except HttpError as exc:  # noqa: BLE001
            raise RuntimeError(f"Captions list failed: {exc}") from exc

        items = response.get("items", [])
        if not items:
            raise RuntimeError("No caption tracks found")

        # Prefer English
        caption_id = None
        for item in items:
            lang = item.get("snippet", {}).get("language")
            if lang and lang.startswith("en"):
                caption_id = item.get("id")
                break
        if caption_id is None:
            caption_id = items[0].get("id")

        if caption_id is None:
            raise RuntimeError("Caption id missing")

        request = youtube.captions().download(id=caption_id, tfmt="srt")
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        srt_body = fh.read().decode("utf-8", errors="replace")

        subtitles = list(srt.parse(srt_body))
        segments = [
            TranscriptSegment(text=sub.content.strip(), start=sub.start.total_seconds(), duration=sub.duration.total_seconds())
            for sub in subtitles
        ]
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
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        if stream is None:
            raise RuntimeError("No audio stream available")
        out_file = stream.download(output_path=str(tmpdir))
        return Path(out_file)

    def _youtube_client(self):
        cred_path = self.settings.google_credentials_file
        if not cred_path:
            return None
        cred_file = Path(cred_path)
        if not cred_file.exists():
            raise RuntimeError(f"Credentials file not found: {cred_file}")

        creds: Optional[Credentials] = None
        token_path = Path(self.settings.google_token_file)
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), self.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(cred_file), self.scopes)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())

        return build("youtube", "v3", credentials=creds, cache_discovery=False)
