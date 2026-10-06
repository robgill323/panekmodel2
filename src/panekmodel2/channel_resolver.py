"""List a YouTube channel's newest uploads, metadata only, for the run picker.

The channel URL is the one value in this app that a user types and the server
hands to yt-dlp as a URL to *resolve*, so it is treated as untrusted input:
it is parsed down to one of four channel forms and a fresh URL is rebuilt from
the validated parts. The raw string never reaches yt-dlp, and the extractor
allow-list stops yt-dlp from taking anything but a channel tab even if it did.

Going the other way, every video ID yt-dlp reports is held to the pipeline's
strict ID check before it is emitted, and the watch and thumbnail URLs are
rebuilt from that ID rather than echoed from the payload.

Extraction is flat: one listing request per page of the tab, no per-video
pages, nothing downloaded. Flat mode reports title and usually duration, but
not upload dates — the tab's own newest-first order stands in for them.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
import threading
from typing import Optional
from urllib.parse import urlsplit

try:
    import yt_dlp as _yt_dlp  # type: ignore
except Exception:  # noqa: BLE001
    _yt_dlp = None

from .pipeline import extract_video_id

logger = logging.getLogger(__name__)

DEFAULT_COUNT = 25
MAX_COUNT = 100
# The whole listing — every page request included — must finish inside this.
FETCH_TIMEOUT_SECONDS = 30.0
# Per-socket bound, so a stalled worker that outlives the request still ends.
SOCKET_TIMEOUT_SECONDS = 15
MAX_INPUT_LENGTH = 300
FETCH_THREAD_NAME = "panekmodel2-channel-list"

_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com"})
_NAME = r"[A-Za-z0-9_][A-Za-z0-9._-]{0,99}"
_CHANNEL_PATH = re.compile(
    rf"/(?P<base>@{_NAME}|channel/UC[A-Za-z0-9_-]{{22}}|c/{_NAME}|user/{_NAME})(?:/videos)?/?"
)
_BARE_HANDLE = re.compile(rf"@{_NAME}")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")

_MESSAGES = {
    "invalid": (
        "That doesn't look like a YouTube channel address. Use a link like "
        "https://www.youtube.com/@handle, or a /channel/UC…, /c/… or /user/… channel link."
    ),
    "not_found": "YouTube has no channel at that address. Check the handle or channel ID for typos.",
    "private_or_empty": (
        "That channel has no public videos to list — it may be private, terminated, "
        "or have no uploads yet."
    ),
    "unreachable": (
        "Couldn't reach YouTube to list that channel. Check this machine's internet "
        "connection and try again."
    ),
    "timeout": (
        f"YouTube took longer than {int(FETCH_TIMEOUT_SECONDS)} seconds to list that channel. "
        "Try again, or ask for fewer videos."
    ),
    "failed": "YouTube didn't return a video list for that channel. Try again in a minute.",
}
_STATUS = {
    "invalid": 400,
    "not_found": 404,
    "private_or_empty": 404,
    "unreachable": 502,
    "timeout": 504,
    "failed": 502,
}


class ChannelError(Exception):
    """A channel listing failure, carrying a plain-language message for the UI."""

    def __init__(self, kind: str, message: Optional[str] = None):
        self.kind = kind
        self.message = message or _MESSAGES[kind]
        self.status = _STATUS[kind]
        super().__init__(self.message)


def normalize_channel_url(raw) -> str:
    """Rebuild a channel's /videos tab URL from user input, or raise.

    Accepts @handle, /channel/UC…, /c/name and /user/name, each with or
    without /videos, on youtube.com, www. or m., with or without a scheme —
    and a bare "@handle". Query strings and fragments are dropped. Anything
    else, including local paths, other hosts, userinfo and ports, is refused.
    """
    if not isinstance(raw, str):
        raise ChannelError("invalid")
    text = raw.strip()
    if not text or len(text) > MAX_INPUT_LENGTH or not text.isascii():
        raise ChannelError("invalid")
    if any(ch.isspace() or ord(ch) < 0x20 or ch in "\\\x7f" for ch in text):
        raise ChannelError("invalid")

    if _BARE_HANDLE.fullmatch(text):
        return f"https://www.youtube.com/{text}/videos"

    if "://" not in text:
        if not text.lower().startswith(tuple(f"{host}/" for host in _HOSTS)):
            raise ChannelError("invalid")
        text = "https://" + text

    parts = urlsplit(text)
    # netloc is compared whole, so userinfo ("x@youtube.com") and ports fail too.
    if parts.scheme not in ("http", "https") or parts.netloc.lower() not in _HOSTS:
        raise ChannelError("invalid")
    match = _CHANNEL_PATH.fullmatch(parts.path)
    if not match:
        raise ChannelError("invalid")
    return f"https://www.youtube.com/{match.group('base')}/videos"


def clamp_count(count: int) -> int:
    return max(1, min(int(count), MAX_COUNT))


def is_strict_video_id(value) -> bool:
    """The pipeline's ID rule, as a full match.

    extract_video_id() alone is not enough here: its first branch is
    ``re.match(r"^…{11}$")``, and "$" also matches before a trailing newline.
    """
    if not isinstance(value, str) or not _VIDEO_ID.fullmatch(value):
        return False
    try:
        return extract_video_id(value) == value
    except ValueError:
        return False


def thumbnail_url(video_id: str) -> str:
    if not is_strict_video_id(video_id):
        raise ValueError(f"Not a video ID: {video_id!r}")
    return f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"


def _duration(raw) -> Optional[int]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if not math.isfinite(raw) or raw < 0:
        return None
    return int(round(raw))


def video_from_entry(entry) -> Optional[dict]:
    """One flat-mode entry as the picker's video record, or None to skip it."""
    if not isinstance(entry, dict):
        return None
    if entry.get("_type") not in (None, "url", "url_transparent"):
        return None
    if entry.get("ie_key") not in (None, "Youtube"):
        return None
    video_id = entry.get("id")
    if not is_strict_video_id(video_id):
        return None
    title = entry.get("title")
    return {
        "id": video_id,
        "title": title.strip() if isinstance(title, str) else "",
        "duration_seconds": _duration(entry.get("duration")),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail_url": thumbnail_url(video_id),
    }


def ydl_options(count: int) -> dict:
    return {
        "extract_flat": True,
        "playlistend": count,
        "skip_download": True,
        "allowed_extractors": ["youtube:tab"],
        "socket_timeout": SOCKET_TIMEOUT_SECONDS,
        "quiet": True,
        "no_warnings": True,
        "logger": _QuietLogger(),
    }


class _QuietLogger:
    """yt-dlp prints ERROR lines to stderr even when quiet; route them to debug."""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        logger.debug("yt-dlp: %s", msg)

    def error(self, msg):
        logger.debug("yt-dlp: %s", msg)


def classify_error(exc: BaseException) -> ChannelError:
    """Map a yt-dlp failure onto one of the plain-language kinds."""
    if isinstance(exc, ChannelError):
        return exc
    low = str(exc).lower()
    if "http error 404" in low or "does not exist" in low:
        return ChannelError("not_found")
    if any(s in low for s in (
        "private", "terminated", "not available", "unavailable",
        "does not have a videos tab", "has no videos",
    )):
        return ChannelError("private_or_empty")
    if any(s in low for s in (
        "unable to download", "urlopen error", "timed out", "connection",
        "transporterror", "errno", "name resolution", "nodename",
    )):
        return ChannelError("unreachable")
    return ChannelError("failed")


def _extract(url: str, count: int):
    with _yt_dlp.YoutubeDL(ydl_options(count)) as ydl:
        info = ydl.extract_info(url, download=False)
        if not isinstance(info, dict):
            return {}, []
        # Drained here, inside the timeout: tab entries are lazy, and each
        # further page is another request.
        entries = list(itertools.islice(info.get("entries") or [], count))
    return info, entries


def _extract_with_timeout(url: str, count: int, timeout: float):
    box: dict = {}

    def work() -> None:
        try:
            box["result"] = _extract(url, count)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            box["error"] = exc

    # A daemon thread rather than an executor: a fetch that overruns is
    # abandoned (socket_timeout ends it soon after) and must not hold up exit.
    worker = threading.Thread(target=work, name=FETCH_THREAD_NAME, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.warning("Channel listing for %s exceeded %.0f s", url, timeout)
        raise ChannelError("timeout")
    if "error" in box:
        logger.info("Channel listing for %s failed: %s", url, box["error"])
        raise classify_error(box["error"])
    return box["result"]


def _channel_name(info: dict) -> str:
    for key in ("channel", "uploader"):
        if isinstance(info.get(key), str) and info[key].strip():
            return info[key].strip()
    title = info.get("title")
    if isinstance(title, str):
        return re.sub(r"\s+-\s+Videos$", "", title.strip())
    return ""


def list_channel_videos(
    channel_url, count: int = DEFAULT_COUNT, *, timeout: Optional[float] = None
) -> dict:
    """The channel's newest ``count`` uploads (capped at MAX_COUNT), newest first.

    Raises ChannelError with a plain-language message on every failure path.
    """
    url = normalize_channel_url(channel_url)
    n = clamp_count(count)
    if _yt_dlp is None:
        raise ChannelError("failed", "yt-dlp is not installed, so channels can't be listed.")
    info, entries = _extract_with_timeout(
        url, n, FETCH_TIMEOUT_SECONDS if timeout is None else timeout
    )

    videos: list[dict] = []
    seen: set[str] = set()
    skipped = 0
    for raw in entries:
        video = video_from_entry(raw)
        if video is None:
            skipped += 1
            continue
        if video["id"] in seen:
            continue
        seen.add(video["id"])
        videos.append(video)
    if not videos:
        raise ChannelError("private_or_empty")
    return {
        "channel": _channel_name(info),
        "channel_url": url,
        "requested": n,
        "skipped": skipped,
        "videos": videos,
    }
