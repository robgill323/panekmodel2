import math
from dataclasses import dataclass
from typing import List, Sequence

from .transcript_fetcher import TranscriptSegment

# Conversational speech runs around 2.6 words/second, the rate the UI quotes
# when it says "30 s ≈ 78 words".
WORDS_PER_SECOND = 2.6


def words_for_seconds(seconds: int) -> int:
    """Word cap to pair with a chunk length expressed in seconds.

    The UI picks chunk size in seconds, but :func:`chunk_segments` splits on
    whichever bound trips first. A fixed word cap would silently override the
    chosen duration on longer settings — at 200 words a "120 s" chunk really
    ends around 75 s. Scaling the cap well above normal speech rate keeps the
    seconds setting authoritative while still capping a pathologically dense
    segment.
    """
    return max(40, math.ceil(seconds * WORDS_PER_SECOND * 1.6))


@dataclass
class Chunk:
    text: str
    start: float
    end: float
    source_indices: List[int]


def chunk_segments(
    segments: Sequence[TranscriptSegment],
    max_words: int = 400,
    max_seconds: int = 90,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    buffer: List[str] = []
    indices: List[int] = []
    start_time: float = 0.0
    end_time: float = 0.0

    def flush():
        nonlocal buffer, indices, start_time, end_time
        if not buffer:
            return
        chunks.append(
            Chunk(
                text=" ".join(buffer).strip(),
                start=start_time,
                end=end_time,
                source_indices=indices.copy(),
            )
        )
        buffer = []
        indices = []

    for idx, seg in enumerate(segments):
        words = seg.text.split()
        if not buffer:
            start_time = seg.start
        end_time = seg.end

        # Split overly long individual segments; iterate in max_words strides
        # so the tail of the segment is never silently discarded.
        if len(words) > max_words:
            flush()
            for word_start in range(0, len(words), max_words):
                chunk_text = " ".join(words[word_start:word_start + max_words])
                chunks.append(Chunk(text=chunk_text, start=seg.start, end=seg.end, source_indices=[idx]))
            continue

        prospective_word_count = len(" ".join(buffer + [seg.text]).split())
        duration = end_time - start_time

        if prospective_word_count > max_words or duration > max_seconds:
            flush()
            start_time = seg.start
            end_time = seg.end

        buffer.append(seg.text)
        indices.append(idx)

    flush()
    return chunks
