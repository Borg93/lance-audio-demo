# raudio

A thin ingestion/search layer that turns
[`easytranscriber`](https://github.com/kb-labb/easytranscriber) transcript JSON
into a **self-contained** [Lance](https://lancedb.com) dataset — text rows
with a **Tantivy BM25 index** for retrieval, plus the original media files
stored inline as **Lance blob-v2** columns so a search hit has everything the
player needs in one place.

The Lance dataset is the **interchange format**: the downstream consumer is
meant to be a TypeScript search application using
[`@lancedb/lancedb`](https://lancedb.github.io/lancedb/) (Node native today,
browser WASM emerging). Python produces the dataset, TypeScript reads it — no
external storage tier, no audio-path resolution, no broken links.

Pipeline:

```
 ┌─────────────┐  easytranscriber   ┌─────────────┐  raudio   ┌─────────────┐
 │  audio/mp3  │ ─────────────────▶ │   *.json    │ ─────────────▶ │    Lance    │
 │  audio/wav  │   VAD + Whisper    │ AudioMetadata│   ingest + FTS │  + Tantivy  │
 │   …/mp4     │   + forced align   │   (msgspec) │                │   index     │
 └─────────────┘                    └─────────────┘                └──────┬──────┘
                                                                         │
                                                       TS search UI  ◀───┘
                                                     (@lancedb/lancedb)
```

- **Input**: `AudioMetadata` JSON files from
  `easytranscriber.pipelines.pipeline(...)` (one per audio file).
- **Output**: a Lance directory with **two tables**:
  - `chunks` — one row per `AudioChunk` (`[start, end)`, word-level alignments,
    document-level context denormalised on). The `text` column is indexed by
    Tantivy BM25.
  - `documents` — one row per source media file. The raw mp3/mp4/wav bytes
    live in a **blob-v2** column (`media_blob`) and are loaded lazily via
    `ds.take_blobs(...)`.
- **Retrieval**: BM25 FTS on `chunks.text`; lookup `documents` by `doc_id`
  when the hit needs playback. Phrase queries, boolean operators, prefix
  match, `NEAR(...)`, etc. are all supported.

Why two tables? A chunk is ~50 words, a source file is ~30 MB. Putting the
blob on every chunk row would duplicate each media file 100–200×. The chunks
table stays light (cheap FTS scans), the documents table holds one heavy
blob per file. Blob-v2 encoding means the bytes never touch the scanner
unless you explicitly request them.

No embedding model, no vector search — just FTS. If you later want hybrid
retrieval you can add a `list<float32>` column with `table.add_columns(...)`
(Lance supports schema evolution).

## Bootstrap / prereqs

### System dependencies

| Tool | Required for | Install |
|---|---|---|
| **`uv`** | everything | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **`ffmpeg`** | `transcribe`, `ingest-with-audio` | `sudo apt install ffmpeg` (Linux) · `brew install ffmpeg` (macOS) |
| **NVIDIA GPU + CUDA** | `transcribe` with `kb-whisper-large` | see [PyTorch install matrix](https://pytorch.org/get-started/locally/); skip for CPU-only smaller models |
| **`hf` CLI + HF token** | gated `pyannote` VAD | `pip install huggingface_hub[cli]` then `hf auth login` (or pass `VAD=silero` to skip) |

Run the built-in check at any time:

```bash
make check-deps
```

It prints a pass/fail for each of the above with the exact install command.

### One-shot bootstrap

```bash
make bootstrap
```

This runs `check-deps`, then `uv sync`, then `uv sync --extra transcribe`, and
prints the next steps. Safe to re-run — `uv sync` is idempotent.

If you only want to **use an existing Lance table** (ingest + search, no
transcription), drop the transcribe extra:

```bash
make install          # lighter install, no torch/whisper
```

---

## Try it — one command

```bash
make demo
```

This installs deps, wipes any previous Lance DB, ingests
`examples/taleoftwocities_01_dickens_64kb_trimmed.json`, and runs three
representative FTS queries so you can see the full pipeline end-to-end. No
audio extraction, no ffmpeg required.

### Producing the JSON files from your own audio

The JSONs that `make ingest` reads are produced by
[`easytranscriber`](https://github.com/kb-labb/easytranscriber) (KBLab's ASR
pipeline: VAD → Whisper → forced alignment). That dep is **optional** (heavy:
torch + transformers + pyannote + ctranslate2) and lives in the `[transcribe]`
extra:

```bash
make install-transcribe                         # uv sync --extra transcribe
# or: uv pip install 'raudio[transcribe]'

# Then run the pipeline on a directory of audio — writes output/alignments/*.json.
# Defaults to kb-whisper-large + Swedish; override via env:
make transcribe AUDIO_DIR=./my-audio LANGUAGE=sv        # GPU recommended
make transcribe AUDIO_DIR=./my-audio LANGUAGE=en \
    MODEL=onnx-community/whisper-base DEVICE=cpu         # CPU-friendly fallback

# Feed those JSONs into Lance:
make reingest SAMPLE=output/alignments/*.json
```

The transcription stage requires `ffmpeg` on PATH and (for gated `pyannote` VAD)
a Hugging Face auth token — run `hf auth login` first, or use `VAD=silero`.
Under the hood `make transcribe` just calls
`raudio transcribe --audio-dir …` (see `src/raudio/transcribe.py`).

### Individual steps

```bash
make install                          # uv sync
make ingest                           # appends the sample into ./transcripts.lance
make reingest                         # same, but wipes the DB first (safe to re-run)
make search Q='best of times'         # BM25 query
make search Q='"spring of hope"'      # exact phrase
make search Q='wisdom OR foolishness' # boolean

# Also embed the raw source media as a blob-v2 column in a `documents` table:
#   needs the source files under AUDIO_ROOT (no ffmpeg required — we store bytes verbatim)
make ingest-with-media AUDIO_ROOT=/path/to/media-dir

make shell                            # Python REPL with `la = raudio`
make clean                            # drop ./transcripts.lance + build artefacts
make reset                            # clean + drop .venv
```

Run `make` with no args for the annotated list of targets.

## Consuming the Lance dataset

### Python — pull a playable blob for a hit

```python
import lance
import lancedb

db = lancedb.connect("./transcripts.lance")
chunks = db.open_table("chunks")

hit = chunks.search("spring of hope", query_type="fts").limit(1).to_list()[0]
print(f"[{hit['start']:.2f}s] {hit['text']}")

# Fetch the source media blob for that chunk's parent document.
docs = lance.dataset("./transcripts.lance/documents.lance")
row = docs.to_table(
    filter=f"doc_id = '{hit['doc_id']}'",
    columns=[],           # <- empty: metadata-only, don't pull bytes
    with_row_id=True,
).to_pylist()[0]

blob = docs.take_blobs("media_blob", ids=[row["_rowid"]])[0]
with blob as f:
    media_bytes = f.read()        # full mp3/mp4/wav
# or stream just a range with f.seek()/f.read(n), or hand `blob` to PyAV.
```

### TypeScript — FTS + join on `doc_id`

```ts
import * as lancedb from '@lancedb/lancedb';

const db = await lancedb.connect('./transcripts.lance');
const chunks = await db.openTable('chunks');
const documents = await db.openTable('documents');

// BM25 full-text search (same Tantivy index `make search` uses).
const hits = await chunks.search('"spring of hope"', 'fts').limit(10).toArray();

for (const h of hits) {
    console.log(`[${h.start.toFixed(2)}s] ${h.text}`);
}

// Fetch the media blob(s) for those hits.
const docIds = [...new Set(hits.map(h => h.doc_id))];
const docs = await documents
    .query()
    .where(`doc_id IN (${docIds.map(d => `'${d}'`).join(',')})`)
    .select(['doc_id', 'media_mime', 'media_blob'])
    .toArray();

for (const d of docs) {
    // d.media_blob is a Uint8Array of the original file bytes.
    const url = URL.createObjectURL(new Blob([d.media_blob], { type: d.media_mime }));
    // <audio src={url} /> or seek with <audio>.currentTime = hit.start
}
```

- **Node/Bun** works today.
- **Browser** via `@lancedb/lancedb-wasm` is still experimental — if you need
  browser-side search today, wrap this Lance table in a tiny FastAPI/Node API
  and have the frontend call that.

`easytranscriber` itself is **not** a required dependency — we vendor a
minimal copy of its msgspec data model (`AudioMetadata` / `SpeechSegment` /
`AudioChunk` / `AlignmentSegment` / `WordSegment`) so ingest doesn't pull in
torch/transformers. Install it separately if you want to generate new JSONs:

```bash
uv pip install 'raudio[transcribe]'
```

## Usage

### CLI

```bash
# Ingest transcripts
raudio --db transcripts.lance ingest output/alignments/*.json

# Full-text search
raudio --db transcripts.lance search '"spring of hope"'
raudio --db transcripts.lance search 'age of wisdom' -n 5
raudio --db transcripts.lance search 'climate OR weather' \
    --where "language = 'en'"
```

### Python

```python
from pathlib import Path
from raudio import ingest_many, load_transcript, nearest_chunks

docs = [load_transcript(p) for p in Path("output/alignments").glob("*.json")]
ingest_many("transcripts.lance", docs)

hits = nearest_chunks("transcripts.lance", "the spring of hope", limit=5)
for h in hits:
    print(f"[{h['start']:.2f}s] {h['audio_path']}\n  {h['text']}")
```

## Schema

Two Lance datasets live under one directory (`transcripts.lance/`). They share
`doc_id` as the join key:

```
transcripts.lance/
├── chunks.lance       — search index, ~150 rows per source file
└── documents.lance    — one row per source file, holds video URI + thumbnail
```

### `chunks` — one row per `AudioChunk` (search target)

Lightweight, scan-friendly. The Tantivy FTS index lives here.

| Column            | Type                          | Notes                                                  |
|-------------------|-------------------------------|--------------------------------------------------------|
| `doc_id`          | string                        | `sha1(audio_path)[:16]` — join key to `documents`      |
| `speech_id`       | int32                         | index of the parent SpeechSegment                      |
| `chunk_id`        | int32                         | index within `speech.chunks`                           |
| `audio_path`      | string                        | original filename (denormalised for filter-free reads) |
| `sample_rate`     | int32                         |                                                        |
| `audio_duration`  | float32                       | whole-file duration                                    |
| `referenskod`     | string                        | Riksarkivet archival reference (from CSV)              |
| `namn`            | string                        | Riksarkivet event title                                |
| `bildid`          | string                        | Riksarkivet image id                                   |
| `extraid`         | string                        | Riksarkivet extra id                                   |
| `start` / `end`   | float32                       | absolute seconds in audio                              |
| `duration`        | float32                       |                                                        |
| `text`            | string                        | **indexed by Tantivy FTS** (Swedish stemmer)           |
| `audio_frames`    | int64                         |                                                        |
| `num_logits`      | int64                         |                                                        |
| `language`        | string                        | inherited from the document (e.g. `"sv"`)              |
| `language_prob`   | float32                       |                                                        |
| `alignments_json` | **JSONB** (`pa.json_()`)      | word-level alignments for this chunk, stored as JSONB  |
| `metadata`        | string (JSON)                 | per-speech metadata escape hatch                       |

Indexes:
- **FTS** on `text` (Tantivy, `with_position=True`, Swedish stemmer, stop words kept)
- **BTREE** on `doc_id` and `audio_path`

Why `alignments_json` is JSONB and not a nested struct: the TS Lance binding
can't decode nested word/struct lists, but every binding can read JSONB. The
backend parses it back to objects before returning hits.

### `documents` — one row per source file (Lance file format 2.2)

Heavy stuff lives here. Required for the viewer (videos + thumbnails).

| Column           | Type                          | Notes                                                                       |
|------------------|-------------------------------|-----------------------------------------------------------------------------|
| `doc_id`         | string                        | shared with `chunks`                                                        |
| `audio_path`     | string                        |                                                                             |
| `sample_rate`    | int32                         |                                                                             |
| `duration`       | float32                       |                                                                             |
| `referenskod`    | string                        |                                                                             |
| `namn`           | string                        |                                                                             |
| `bildid`         | string                        |                                                                             |
| `extraid`        | string                        |                                                                             |
| `language`       | string                        | ISO 639-1 code (`sv`, `en`, …)                                              |
| `media_mime`     | string                        | e.g. `"video/mp4"`                                                          |
| `media_blob`     | **Blob V2 External**          | stores a **URI** (`file:///abs/path.mp4`); read lazily via `ds.take_blobs`  |
| `thumbnail`      | **Blob V2 Inline**            | JPEG **bytes** stored in-line in the data file                              |
| `thumbnail_mime` | string                        | usually `"image/jpeg"`                                                      |
| `metadata`       | string (JSON)                 |                                                                             |

The Blob V2 split is intentional:
- **Videos** are GBs total → store URIs only, never copy bytes into Lance
- **Thumbnails** are KBs each → inline so a fetch is one Lance read

Full Arrow struct definitions live in [`schema.py`](src/raudio/schema.py).

## Search at query time

The backend builds a **structured Lance FTS query** (not a raw query string),
which gives us proper fuzziness and phrase support that the plain string API
doesn't expose.

### What happens for a single search

```
GET /api/search?q=betänkandet&fuzziness=0&phrase=false&language=sv&namn=…
```

1. **Build optional SQL where-clause** from the metadata params:
   ```sql
   language = 'sv' AND namn LIKE '%alkohol%' AND referenskod LIKE '%SE/RA%'
   ```
2. **Pick the FTS query type** based on toggles:
   - `phrase=true`    → `PhraseQuery(q, "text")` — exact word sequence (requires `with_position=True` index, which we have)
   - `fuzziness=N`    → `MatchQuery(q, "text", fuzziness=N)` — Levenshtein distance ≤ N (0, 1, or 2)
   - default          → `MatchQuery(q, "text", fuzziness=0)` — exact stemmed terms
3. **Run FTS, post-filter** on the where-clause, take top-N by BM25 score:
   ```python
   chunks.search(MatchQuery(q, "text", fuzziness=2))
         .where("language = 'sv' AND namn LIKE '%alkohol%'", prefilter=False)
         .limit(20)
         .to_list()
   ```
4. **Join via the frontend**: each hit carries `doc_id`. The viewer fetches
   `/api/thumbnail/{doc_id}` and `/api/media/{doc_id}` which call
   `documents.take_blobs("thumbnail" | "media_blob", indices=[idx])`. Media
   reads support HTTP Range — `BlobFile.seek(start) + read(length)` streams the
   exact byte range the browser asked for, so video scrubbing works without
   ever loading the whole MP4.

### Tokenization of `text`

The FTS index is built with:

| Setting             | Value      | Why                                                                  |
|---------------------|------------|----------------------------------------------------------------------|
| `language`          | Swedish    | Stemmer maps `betänkandet`/`betänkande`/`betänkanden` → same stem    |
| `with_position`     | true       | Required for `PhraseQuery`                                           |
| `remove_stop_words` | false      | Keeps "of"/"the"/"av" so quoted phrases match verbatim               |
| `lower_case`        | true (default) | Case-insensitive matching                                        |
| `ascii_folding`     | true (default) | `é`/`å` collapse to `e`/`a` for accent-tolerant matches          |

### What works in the search box

Lance's plain query string is **not** a Tantivy mini-language — `AND`/`OR`/`NOT`
typed in the box are treated as literal terms. To control matching, use the
toggles next to the search bar:

| Input                                                  | Hits                                                                      |
|--------------------------------------------------------|---------------------------------------------------------------------------|
| `betänkandet`                                          | Stemmed match: also finds *betänkande*, *betänkanden*                     |
| `minister regering`                                    | Either term, ranked by BM25 (more matched terms = higher score)           |
| `betänkadet` + ☑ **fuzzy**                             | Edit-distance ≤ 2: finds *betänkandet* despite the typo                   |
| `alkoholmonopolets framtid` + ☑ **phrase**             | Exact phrase, words adjacent in order                                     |
| filter: `language = sv`                                | Pre-narrows the result set; combines with FTS query via SQL `AND`         |

### Why typos return 0 hits without "fuzzy"

Tantivy looks up the exact stemmed token in its term dictionary. A misspelling
like `betänkadet` (missing `n`) doesn't exist in any inflection of any Swedish
word — the stemmer can't reconstruct the missing letter, so the lookup
returns nothing. Toggling **fuzzy** switches the query type to
`MatchQuery(..., fuzziness=2)` which permits up to 2 edits — at the cost of
some false positives on short queries.

## Relationship to `easytranscriber`

`easytranscriber` ships its own search UI (`easysearch`) backed by **SQLite
FTS5**. `raudio` is an alternative sink that targets **Lance + Tantivy**
instead — useful when you want columnar storage, ACID snapshots, the ability
to bolt on vector search later, or integration with the broader Lance
ecosystem (Rust / JS / LanceDB Cloud).

## Layout

```
raudio/
├── Makefile               # bootstrap + transcribe + ingest + search targets
├── README.md
├── pyproject.toml         # deps; `[transcribe]` extra adds easytranscriber
├── examples/
│   └── taleoftwocities_01_dickens_64kb_trimmed.json   # sample alignment JSON
└── src/raudio/
    ├── __init__.py        # public re-exports
    ├── cli.py             # `raudio {transcribe,ingest,search}` entrypoint
    ├── datamodel.py       # msgspec structs mirroring easytranscriber
    ├── schema.py          # CHUNK_SCHEMA + DOC_SCHEMA pyarrow schemas
    ├── ingest.py          # load_transcript / flatten_chunks / ingest_{document,many}
    ├── search.py          # nearest_chunks (Tantivy FTS) + timecode helper
    ├── audio.py           # ffmpeg PCM extraction for the audio_pcm column
    └── transcribe.py      # wrapper around easytranscriber.pipelines.pipeline
```
