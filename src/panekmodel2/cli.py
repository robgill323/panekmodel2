import json
import logging
from pathlib import Path
from typing import Optional

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
def run(url_or_id: str, whisper: bool = typer.Option(False, help="Enable Whisper fallback")):
    """Run full pipeline: fetch transcript, chunk, topics, sentiment."""
    settings = _load_settings(use_whisper=whisper)
    runner = PipelineRunner(settings)
    outputs = runner.run(url_or_id)

    console.rule("Transcript & Chunks")
    console.print(f"Video: {outputs.video_id}")
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
