import json
import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings, get_settings
from .pipeline import PipelineRunner, extract_video_id

logging.basicConfig(level=logging.INFO)
console = Console()
app = typer.Typer(help="YouTube transcript → topics → sentiment pipeline")


def _load_settings(use_whisper: Optional[bool]) -> Settings:
    base = get_settings()
    data = base.model_dump()
    if use_whisper is not None:
        data["use_whisper_fallback"] = use_whisper
    return Settings(**data)


@app.command()
def fetch(url_or_id: str, outfile: Path = typer.Option(Path("transcript.json"))):
    """Fetch transcript and write JSON with timestamped segments."""
    settings = _load_settings(use_whisper=False)
    runner = PipelineRunner(settings)
    video_id = extract_video_id(url_or_id)
    segments = runner.fetcher.fetch(video_id)
    payload = [seg.__dict__ for seg in segments]
    outfile.write_text(json.dumps(payload, indent=2))
    console.print(f"Saved {len(segments)} segments to {outfile}")


@app.command()
def run(
    urls: List[str] = typer.Argument(..., help="One or more YouTube URLs or video IDs"),
    whisper: bool = typer.Option(False, help="Enable Whisper fallback"),
    chunk_max_words: Optional[int] = typer.Option(None, help="Max words per chunk (overrides config)"),
    chunk_max_seconds: Optional[int] = typer.Option(None, help="Max seconds per chunk (overrides config)"),
    topic_reduce_to: Optional[int] = typer.Option(None, help="Reduce topics to this count (overrides config)"),
):
    """Run full pipeline on one or more videos (shared topic model when multiple)."""
    settings = _load_settings(use_whisper=whisper)
    if chunk_max_words is not None:
        settings = Settings(**{**settings.model_dump(), "chunk_max_words": chunk_max_words})
    if chunk_max_seconds is not None:
        settings = Settings(**{**settings.model_dump(), "chunk_max_seconds": chunk_max_seconds})
    if topic_reduce_to is not None:
        settings = Settings(**{**settings.model_dump(), "topic_reduce_to": topic_reduce_to})
    runner = PipelineRunner(settings)

    all_outputs = runner.run_multi(urls) if len(urls) > 1 else [runner.run(urls[0])]

    if len(all_outputs) > 1:
        console.rule("[bold]Shared Topic Model")
        # Summarize topics from the combined corpus (use first output's model)
        # Print combined topic keywords using the shared model directly
        topic_kws: dict = {}
        for tid in runner.topic_modeler.model.get_topics():
            if tid == -1:
                continue
            kws = [kw for kw, _ in (runner.topic_modeler.model.get_topic(tid) or [])][:5]
            topic_kws[tid] = kws
        shared_table = Table("Topic", "Keywords")
        for tid in sorted(topic_kws):
            shared_table.add_row(str(tid), ", ".join(topic_kws[tid]))
        console.print(shared_table)

    for outputs in all_outputs:
        console.rule(f"[bold]{outputs.video_id}")
        if outputs.metadata:
            console.print(outputs.metadata)
        console.print(f"Segments: {len(outputs.segments)} | Chunks: {len(outputs.chunks)}")

        console.rule("Topics")
        topic_summary = runner.summarize_topics(outputs)
        table = Table("Topic", "Keywords", "Representative (start-end)")
        for t in topic_summary:
            rep = t["representative"]
            table.add_row(str(t["topic_id"]), ", ".join(t["keywords"]), f"{rep['start']:.1f}-{rep['end']:.1f}: {rep['text']}")
        console.print(table)

        console.rule("Sentiment")
        roll = outputs.sentiment_rollup
        console.print(f"Mean: {roll['mean']:.3f} | Median: {roll['median']:.3f}")
        console.print(f"Fractions: {roll['fractions']}")

        by_topic = runner.sentiment_by_topic(outputs)
        if by_topic:
            console.print("Sentiment by topic:")
            ttable = Table("Topic", "Mean", "Median", "% Positive", "% Negative")
            for row in by_topic:
                frac = row["sentiment"].get("fractions", {})
                pos = frac.get("positive", 0.0) * 100
                neg = frac.get("negative", 0.0) * 100
                ttable.add_row(
                    str(row["topic_id"]),
                    f"{row['sentiment']['mean']:.3f}",
                    f"{row['sentiment']['median']:.3f}",
                    f"{pos:.1f}%",
                    f"{neg:.1f}%",
                )
            console.print(ttable)


if __name__ == "__main__":
    app()
