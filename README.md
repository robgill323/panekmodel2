# panekmodel2

End-to-end YouTube transcript ingestion, topic modeling (BERTopic), and sentiment pipeline. Provides a tiered transcript strategy (official captions, public transcripts, Whisper fallback), chunked processing, BERTopic topics, and transformer-based sentiment with rollups.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m nltk.downloader stopwords
```

Run the full pipeline on a YouTube URL or ID:

```bash
panekmodel2 run https://www.youtube.com/watch?v=VIDEO_ID
```

This will:

1. Fetch transcript (official captions if credentials are configured, else public transcript, else Whisper fallback if enabled).
2. Chunk text into timestamped blocks.
3. Fit BERTopic on chunk texts and reduce to a manageable number of topics.
4. Compute transformer sentiment per chunk and aggregate per video and per topic.
5. Print a concise report to the console.

## Configuration

Environment variables (or .env) via pydantic BaseSettings:

- `YOUTUBE_API_KEY` or `GOOGLE_CREDENTIALS_FILE` for official captions (OAuth).
- `WHISPER_MODEL` (e.g., `small`, `base`, `medium`, `large-v3`) if using ASR fallback.
- `EMBEDDING_MODEL` for BERTopic (default `all-mpnet-base-v2`).
- `CHUNK_MAX_WORDS` (default 400) and `CHUNK_MAX_SECONDS` (default 90).

## Notes

- Whisper requires `ffmpeg` installed on your system. On macOS: `brew install ffmpeg`.
- Official caption download via the YouTube Data API needs OAuth; provide a credentials JSON and token store. The code includes a minimal client; you may need to adapt scopes/consent for your project.
- Topic modeling and Whisper depend on torch; ensure you have a compatible build for your hardware.

## CLI commands

- `panekmodel2 run <video_url_or_id>`: run ingestion → topics → sentiment and print a report.
- `panekmodel2 fetch <video_url_or_id>`: fetch transcript only and save as JSON.

## Outputs

Console report includes:

- Video metadata, transcript coverage, and chunk counts.
- Top topics with c-TF-IDF keywords and representative chunks (timestamps).
- Sentiment aggregates: mean, median, and % positive/neutral/negative; sentiment timeline bins.

## Extending

- Swap in Top2Vec or CTM by adding a new model class in `topic_model.py`.
- Add UI for human-in-the-loop topic curation; the pipeline surfaces stable IDs and chunk references.
