# panekmodel2

End-to-end YouTube transcript ingestion, topic modeling (BERTopic), and sentiment
analysis, with a local web UI. Fetches public transcript tracks (with an optional
local Whisper fallback for videos that have none), chunks them on a timeline, fits
one BERTopic model across the whole batch so topic IDs are comparable between
videos, and scores sentiment per chunk.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

NLTK data for entity detection is downloaded automatically on first use; no
separate download step is needed.

### Throughline — the web UI

The primary interface. Paste a batch of URLs, watch per-stage progress, and
explore topics, sentiment and the per-video Throughline:

```bash
panekmodel2 ui              # serves http://127.0.0.1:8000 and opens a browser
panekmodel2 ui --port 9000 --no-open
```

The server binds `127.0.0.1` by default. It has no authentication and runs
unbounded compute on request, so do not bind a public interface.

Results are session-scoped: they live in the server process and are discarded
when it stops. Export before quitting. Runs are executed one at a time — the
topic model is shared, so overlapping runs would corrupt each other's results;
a queued run says so on the progress screen.

### Command line

Run the full pipeline on a YouTube URL or ID:

```bash
panekmodel2 run https://www.youtube.com/watch?v=VIDEO_ID
```

This will:

1. Fetch the transcript (public transcript track, else the Whisper audio fallback if enabled).
2. Chunk text into timestamped blocks.
3. Fit BERTopic on chunk texts and reduce to a manageable number of topics.
4. Compute transformer sentiment per chunk and aggregate per video and per topic.
5. Print a concise report to the console.

## Configuration

Environment variables (or .env) via pydantic BaseSettings:

- `YOUTUBE_API_KEY` — optional, and used **only** to fetch video metadata
  (title, channel, publish date). It cannot fetch captions. Without it, metadata
  falls back to yt-dlp; transcripts are unaffected either way.
- `USE_WHISPER_FALLBACK` (default `false`) and `WHISPER_MODEL` (default `small`;
  also `base`, `medium`, `large-v3`) for videos with no caption track.
- `EMBEDDING_MODEL` for BERTopic (default `all-mpnet-base-v2`).
- `SENTIMENT_MODEL` (default `cardiffnlp/twitter-roberta-base-sentiment-latest`).
- `CHUNK_MAX_SECONDS` (default 60) and `CHUNK_MAX_WORDS` (default 200) — see
  Chunk size below.
- `TOPIC_REDUCE_TO` (default 10; `0` disables topic reduction).
- `TOPIC_GRANULARITY` — `coarse`, `standard` (default) or `fine`. See below.

Retired in 0.2.0: `GOOGLE_CREDENTIALS_FILE` and `GOOGLE_TOKEN_FILE`, which
configured the OAuth captions tier. Leaving them in a `.env` is harmless — they
are ignored — and the app logs one INFO line at startup saying so rather than
letting a formerly load-bearing variable vanish silently.

## Notes

- Whisper requires `ffmpeg` and `yt-dlp`. On macOS: `brew install ffmpeg`.
- There is no OAuth caption tier. The YouTube Data API only serves caption
  downloads to the *owner* of a video, so it could never work for third-party
  analysis; the code for it was removed rather than left as dead weight.
- Topic modeling and Whisper depend on torch; ensure you have a compatible build for your hardware.
- A batch runs at most 200 URLs, and runs execute one at a time.
- Credentials never reach the logs: records from this package are scrubbed of
  `key=`, `access_token=` and similar query parameters before any handler
  formats them, because Google API errors stringify to the full request URL.

## CLI commands

- `panekmodel2 ui`: serve the Throughline web UI and its API on localhost.
- `panekmodel2 run <video_url_or_id>`: run ingestion → topics → sentiment and print a report.
- `panekmodel2 fetch <video_url_or_id>`: fetch transcript only and save as JSON.

## Sentiment scale

Every sentiment number in the app, the API and the CSV exports is a *valence*
in −1 … +1, produced by the single conversion in `sentiment.py`
(`normalize_sentiment`). A `neutral` label maps to exactly `0.0` regardless of
model confidence, and a label the project cannot interpret raises rather than
being silently treated as neutral.

The default model, `cardiffnlp/twitter-roberta-base-sentiment-latest`, is
three-class: a procedural passage can genuinely be scored neutral.
`siebert/sentiment-roberta-large-english` is offered as an option but is
**binary** — it has no neutral class at all, so every chunk is forced positive
or negative and values near zero mean low confidence, not neutrality. Whenever
a run produces no neutral chunks the UI says so, and it only calls a model
binary when that model actually is.

## Topic granularity

How finely a batch is split into topics, selectable per run in Advanced
settings and stamped into the run header, the export metadata and the export
filename — granularity changes the topic set, so a figure cited from one
export cannot be reproduced from another without it.

| Level | Effect |
|---|---|
| `coarse` | Fewer, broader topics. Good for asking what a batch is broadly about. |
| `standard` | The balanced default; identical to the behaviour before this knob existed. |
| `fine` | More, narrower topics. Use when one long video collapses into a single topic. |

The value is HDBSCAN's minimum cluster size — the fewest chunks that may form
a topic — and each level is its own curve over corpus size:

| chunks | coarse | standard | fine |
|---|---|---|---|
| ≤ 10 | 4 | 2 | 2 |
| 11–50 | 6 | 3 | 2 |
| 200 | 10 | 5 | 2 |
| ≥ 400 | 10 | 5 | 3 |

`standard` is byte-identical to the behaviour that predates the knob. The
result is always at least 2 and never more than the batch can support, so a
coarse setting on a very short video cannot make the clusterer raise. Below 11
chunks `standard` is already at the floor of 2, so `fine` cannot go finer and
the knob is inert there — stated rather than hidden.

## Chunk size

The UI picks chunk length in seconds (15/30/60/120). `chunk_segments` splits on
whichever bound trips first, so the API derives a matching word cap
(`chunker.words_for_seconds`) rather than letting a fixed `CHUNK_MAX_WORDS`
quietly cut a "120 s" chunk short at about 75 s. Set `chunk_max_words`
explicitly on a run to override that. Both values are stamped into the run
settings and every export.

## Tests

```bash
pip install -e '.[dev]'
pytest                  # fast suite; model weights are stubbed
pytest -m slow          # adds real BERTopic fits (downloads all-MiniLM-L6-v2)
```

## Outputs

The web UI is the primary output: batch overview, topics, per-topic sentiment,
the per-video Throughline, and three CSV exports (per chunk, per video, per
topic). CSV cells that would otherwise be read as spreadsheet formulas are
prefixed with `'` so opening an export cannot execute transcript text.

The `run` console report includes:

- Video metadata and chunk counts.
- Topics with their keywords and a representative chunk (with timestamps).
- Sentiment aggregates: mean and median valence, and the per-topic breakdown.

## Extending

- Swap in Top2Vec or CTM by adding a new model class in `topic_model.py`.
- Topic labels are generated from each topic's own top terms; the keyword lists
  are the ground truth and the labels claim nothing beyond them.
