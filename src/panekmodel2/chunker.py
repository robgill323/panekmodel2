from dataclasses import dataclass
from typing import List, Sequence

from .transcript_fetcher import TranscriptSegment


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
