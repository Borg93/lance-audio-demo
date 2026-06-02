# raudio — Architecture & Onboarding Guide

> A developer's map of `lance-audio`. The [README](README.md) is the quickstart
> (how to install and run); this guide is the **why and how** — the mental
> model, the data flow, the design decisions, and where to look for things.
> For the running task list see [TODO.md](TODO.md).

---

## 1. What this is, in one paragraph

`raudio` is a **searchable archive viewer for Swedish press-conference video
transcripts**. It ingests [`easytranscriber`](https://github.com/kb-labb/easytranscriber)
output (word-aligned transcript JSON + the source MP4) into a single, self-contained
[Lance](https://lancedb.com) dataset, then serves keyword / semantic / visual /
hybrid search and synchronized video playback through a typed HTTP API and a
SvelteKit UI. Everything — text, metadata, embeddings, thumbnails, per-chunk
video frames, and the media URIs — lives in **one Lance database**; there are no
sidecar JSON files or disk walks at query time.

**Deep-dive docs** (this guide is the overview; these go deep with diagrams):
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — the ASR pipeline & models (easytranscriber / easyaligner, KB-Whisper, wav2vec2, MMS-LID).
- [`docs/STORAGE.md`](docs/STORAGE.md) — how raudio uses Lance (Blob V2 tiers, JSONB, IVF_PQ / FTS).
- [`docs/EMBEDDINGS.md`](docs/EMBEDDINGS.md) — Qwen3-VL embeddings + reranking over vLLM.
- [`docs/INVESTIGATION.md`](docs/INVESTIGATION.md) — root-cause analysis of the Lance indexation + vLLM crash issues.

---

## 2. The mental model: one write side, one read side

```mermaid
flowchart LR
    subgraph WRITE["WRITE SIDE — offline, CLI (src/raudio/)"]
        direction TB
        CLI["raudio CLI<br/>transcribe → ingest → feature text_embedding<br/>→ extract-chunk-frames → feature frame_embedding"]
    end

    subgraph DB["transcripts.lance/ (single Lance dataset)"]
        direction TB
        T1["chunks.lance<br/>text + FTS + text_embedding"]
        T2["documents.lance<br/>media_blob + thumbnail"]
        T3["chunk_frames.lance<br/>frame_blob + frame_embedding"]
    end

    subgraph READ["READ SIDE — online, serving"]
        direction TB
        FE["SvelteKit frontend (frontend/)<br/>Bun static server + /api/* proxy"]
        BE["FastAPI backend (backend/app.py)<br/>/api/search /media /thumbnail /chunk-frame"]
        FE -->|"HTTP /api/* (proxied)"| BE
    end

    VLLM["vLLM servers (out-of-process, HTTP)<br/>Qwen3-VL Embedding-2B :8001 · Reranker-2B :8002"]

    CLI -->|writes| DB
    BE -->|"reads (lancedb + lance handles)"| DB
    CLI -.->|embed batch| VLLM
    BE -.->|"embed query / rerank"| VLLM

    classDef store fill:#1a1a1e,stroke:#818cf8,color:#e9e9ea;
    class T1,T2,T3 store;
```

> Both the CLI batch path and the backend serving path call the **same**
> `src/raudio/vllm/embedding.py` client (the "shared seam", §7) → the same
> out-of-process vLLM servers.

- **Write side** = `src/raudio/` (the `raudio` Typer CLI + the ingest/extract
  modules). Run offline, GPU-heavy, idempotent/resumable.
- **Read side** = `backend/` (FastAPI) + `frontend/` (SvelteKit). Run online,
  mostly CPU; only the embedding step needs a GPU and that lives in a **separate
  vLLM process** reached over HTTP, so FTS-only use works with no GPU at all.
- **`src/raudio/vllm/embedding.py` is the one module both sides share** — see §7.
- **`demo/` is unrelated** — a standalone in-browser (transformers.js/WebGPU)
  transcription playground. It shares zero code with raudio. See §9.

---

## 3. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Storage / search | **Lance / LanceDB** (file format 2.2) | Columnar store + Tantivy BM25 FTS + IVF_PQ vector index + Blob V2 |
| Transcript parsing | **Pydantic v2** | Typed model of easytranscriber JSON; `model_validate_json` at the ingest boundary |
| Arrow plumbing | **PyArrow** | Schemas, table materialization, blob/JSON extension columns |
| API | **FastAPI** + **Starlette** + **uvicorn** | API-only; HTTP Range streaming off Lance Blob V2 |
| Embeddings / rerank | **vLLM** serving **Qwen3-VL-Embedding-2B** / **-Reranker-2B** | Out-of-process, OpenAI-compatible HTTP; full 2048-d (no truncation) |
| Media | **ffmpeg** (subprocess) | Thumbnails + per-chunk frame extraction |
| ASR (upstream) | **easytranscriber** (Whisper/ct2 + wav2vec2 alignment) | torch pinned to `cu128` |
| Frontend | **SvelteKit 2 / Svelte 5** (runes), **TypeScript**, **Vite**, **Tailwind 4**, **Zod**, **bits-ui**, **Bun** | SPA (`adapter-static`, `ssr=false`); Bun server proxies `/api/*` |
| Python toolchain | **uv** + **ruff** + **ty** + **pytest** | `uv run` / `uvx`; lint + type config in `pyproject.toml` |
| Orchestration | **GNU Make** | `Makefile` is the single source of truth for how every piece runs together |

**Why vLLM is out-of-process (not a Python dep):** its torch pins conflict with
our `cu128` pin, and loading a Qwen3-VL model takes tens of seconds + several GB of VRAM. Running it
as a long-lived `uvx`/docker server means the model loads once and stays warm
across every CLI run and API restart, and gets continuous-batching throughput
for free. The CLI/backend are just HTTP clients of it.

---

## 4. The data model — three Lance tables

All three live inside one `transcripts.lance/` directory (gitignored — local
working data). Schemas: [`src/raudio/model/schema.py`](src/raudio/model/schema.py).

| Table | Grain | Carries | Key columns |
|---|---|---|---|
| **`chunks`** | one row per ~30 s transcript chunk | FTS text, metadata, alignments JSON, `text_embedding` (2048-d) | `(doc_id, speech_id, chunk_id)` |
| **`documents`** | one row per source media file | `media_blob` (Blob V2 *External* URI), `thumbnail` (Blob V2 *Inline* bytes), metadata | `doc_id` |
| **`chunk_frames`** | one row per extracted frame (a chunk can hold N frames, `frame_idx` 0..K-1; `frame_idx=0` is the single representative frame) | `frame_blob` (Blob V2 Inline JPEG), `frame_embedding` (2048-d, added later) | `(doc_id, speech_id, chunk_id, frame_idx)` |

`doc_id` is `sha1(audio_path)[:16]` — deterministic, so re-ingesting the same
file is stable.

```mermaid
erDiagram
    documents ||--o{ chunks : "doc_id"
    documents ||--o{ chunk_frames : "doc_id"
    chunks ||--o| chunk_frames : "(doc_id, speech_id, chunk_id)"

    documents {
        string doc_id PK "sha1(audio_path)[:16]"
        blob media_blob "Blob V2 External — URI"
        blob thumbnail "Blob V2 Inline — JPEG bytes"
        string namn "archival metadata: referenskod, namn, bildid, extraid"
    }
    chunks {
        string doc_id FK
        int speech_id
        int chunk_id
        string text "Tantivy FTS (BM25, Swedish)"
        json alignments_json "JSONB word timings"
        vector text_embedding "2048-d cosine IVF_PQ"
    }
    chunk_frames {
        string doc_id FK
        int speech_id
        int chunk_id
        int frame_idx "0..K-1; 0 = representative frame"
        blob frame_blob "Blob V2 Inline — JPEG"
        vector frame_embedding "2048-d cosine IVF_PQ"
    }
```

See [`docs/STORAGE.md`](docs/STORAGE.md) for the full Lance storage deep-dive.

**Why `chunk_frames` is a separate table** (the single most important schema
fact): Lance 4.0's `merge_insert` crashes its encoder when backfilling blob
columns post-hoc on the *wide* `chunks` schema (multiple extension types at
once). The Lance 2.2 docs recommend "append + `add_columns`" for data evolution,
so frames go into their own append-only table: `extract-chunk-frames` writes new
fragments, `feature frame_embedding` attaches `frame_embedding` via
`dataset.add_columns(...)`. No `merge_insert`. Visual / cross-modal search runs
the frame-vector query against `chunk_frames` and joins back to `chunks` for the
hit payload (`backend/search/service.py::_frame_search`); `chunks` carries **no**
`frame_*` columns.

**Blob V2 cheat-sheet** (load-bearing constraints):

- *External* = the column stores a **URI string** (`file://`, `hf://`, `s3://`)
  wrapped with `lance.blob_array([...])`; bytes live wherever the URI points.
- *Inline* = small bytes (<64 KB) wrapped with `blob_array([...])`, stored in the
  main data page.
- Read both via `ds.take_blobs(col, indices=[...])[0]` → a lazy, seekable
  `BlobFile`. This is why HTTP Range maps cleanly to `seek(start) + read(len)`.
- You **cannot** build a `blob_field` or `pa.json_()` column with
  `pa.array(values, type=...)`; you must wrap blob columns with `blob_array(...)`
  and build JSON columns per-declared-field-type. Both writers in `ingest/ingest.py`
  special-case these.
- Blob columns require `data_storage_version="2.2"`, which `lancedb.create_table`
  can't set — so `ingest/ingest.py` writes the first dataset via
  `lance.write_dataset(mode="create", data_storage_version="2.2", allow_external_blob_outside_bases=True)`
  then re-opens it through lancedb.

---

## 5. End-to-end information flow

### Build side (offline pipeline)

```mermaid
flowchart TD
    A["input/&lt;lang&gt;/*.mp4"] -->|"raudio detect-language<br/>(MMS-LID / Whisper)"| B["sorted into &lt;lang&gt;/"]
    B -->|"raudio transcribe<br/>(easytranscriber → KB-Whisper)"| C["output/&lt;lang&gt;/alignments/*.json"]
    C -->|"raudio thumbnail (ffmpeg)"| D["thumbnails/{stem}.jpg"]
    C --> E
    D --> E
    E["raudio ingest<br/>JSON → chunks + documents · FTS + scalar indexes"] --> F
    F["raudio feature text_embedding<br/>Qwen3-VL text → text_embedding · IVF_PQ"] --> G
    G["raudio extract-chunk-frames<br/>ffmpeg → chunk_frames.lance (append-only)"] --> H
    H["raudio feature frame_embedding<br/>Qwen3-VL image → frame_embedding (add_columns)"] --> DB["(transcripts.lance)<br/>self-contained dataset"]

    classDef done fill:#1a1a1e,stroke:#34d399,color:#e9e9ea;
    classDef blocked fill:#1a1a1e,stroke:#f87171,color:#e9e9ea;
    class C,D,E,F done;
    class G,H blocked;
```

> 🟢 = validated · 🔴 = the frame stages (`extract-chunk-frames` /
> `feature frame_embedding`) are
> blocked on the vLLM image-embed crash — see [`docs/INVESTIGATION.md`](docs/INVESTIGATION.md).
> Detailed pipeline + models: [`docs/PIPELINE.md`](docs/PIPELINE.md).

`ingest` is the heart: `load_transcript` (Pydantic `model_validate_json`) → `flatten_chunks`
(walks speeches→chunks, joins the `video_batcher` CSV metadata by
`bildid == audio_path stem`, embeds word alignments as JSON) → materializes
PyArrow tables → writes `chunks` (+ `documents` when `--audio-root`/`--thumbnail-dir`
is given) → builds the Tantivy FTS index (Swedish stemmer) + BTREE scalar indexes.

### Read side (a search request)

```mermaid
flowchart TD
    Q["browser → SvelteKit (Bun :3000)"] -->|"proxy /api/*"| BE["FastAPI :8000<br/>normalize → SearchSpec"]
    BE --> M{"mode?"}
    M -->|fts| F1["chunks.search() Tantivy BM25 · no GPU"]
    M -->|semantic| F2["vLLM embed text → nearest_to(text_embedding, cosine)"]
    M -->|visual| F3["vLLM embed image/text → nearest_to(frame_embedding)"]
    M -->|hybrid| F4["Lance native FTS + text-vector → RRF / LinearCombination / Qwen rerank"]
    M -->|all| F5["FTS + text-vector + frame-vector → _rrf_fuse → optional rerank"]
    F1 --> P["parse alignments_json → alignments<br/>(one hit shape for all modes)"]
    F2 --> P
    F3 --> P
    F4 --> P
    F5 --> P
    P -->|JSON| UI["HitList / HitCard → click → PlayerPane"]
```

Detailed embedding/serving flow: [`docs/EMBEDDINGS.md`](docs/EMBEDDINGS.md).

### Playback (Range streaming)

```
<video src="/api/media/{doc_id}">  (with Range: bytes=…)
   → FastAPI resolves doc_id → Lance _rowid → ds.take_blobs("media_blob")[0]
   → BlobFile.seek(start) + read(len) streamed back as 206 Partial Content
```

Per-chunk frames are fetched on demand from `/api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}?frame_idx=N`
(`frame_idx` defaults to 0, the representative frame);
the frontend stops asking after the first 404 (frames not extracted yet) via the
`feature-flags.svelte.ts` singleton.

---

## 6. Where things live (navigation map)

```
src/raudio/                Python pipeline — the WRITE side + search/embedding library
├── cli/                 Typer app: subcommands (the operator entry point)
├── __init__.py            library public API (re-exports model / ingest / retrieval)
├── model/                 DATA CONTRACTS
│   ├── datamodel.py        Pydantic v2 models mirroring easytranscriber JSON
│   └── schema.py           PyArrow/Lance schemas (incl. CHUNK_FRAMES_SCHEMA) + storage version
├── ingest/                WRITE PATH
│   ├── ingest.py           JSON → chunks + documents tables; FTS/scalar indexes
│   └── audio.py            source-path resolution + media URI / MIME composition
├── media/                 FFMPEG / DOWNLOAD side-steps
│   ├── frames.py           ffmpeg frame extraction + write_chunk_frames (thread-pool batcher)
│   ├── thumbnails.py       per-file thumbnail / waveform generation (ffmpeg)
│   └── download.py         bulk media download from a video_batcher CSV (httpx async)
├── asr/                   UPSTREAM ASR wrappers (in-process pipeline STAGE)
│   ├── transcribe.py       easytranscriber pipeline wrapper (Whisper/wav2vec2)
│   └── detect_language.py  MMS-LID / Whisper language detection + sort into <lang>/
├── vllm/                  CLIENTS to remote vLLM servers (shared by CLI + backend)
│   ├── base.py             VLLMTransport: shared httpx pool + concurrent fan-out
│   ├── embedding.py        VLLMEmbeddingClient + EmbeddingClient Protocol — SHARED SEAM
│   ├── reranker.py         cross-encoder: VLLMReranker + QwenVLReranker (Lance adapter)
│   ├── caption.py          image caption client
│   ├── summarize.py        text summary client
│   └── image.py            pure helpers: l2_normalize, image_to_data_url
├── features/             DATA-EVOLUTION engine (type-agnostic column upserts)
│   ├── engine.py           upsert_scan_column / upsert_blob_column / ensure_vector_index
│   └── columns.py          embed_text_column / embed_frame_column / summary_column /
│                           caption_column + the FEATURES registry
└── retrieval/             READ PATH
    ├── search.py           FTS query + pure parsing/formatting helpers (timecode, …)
    └── qwen3_vl_reranker.jinja  vLLM chat template for the reranker server

backend/                   FastAPI package (the read-side serving app)
├── app.py                 create_app() factory wiring the routers together
├── state.py               app state / Lance handles
├── deps.py                FastAPI dependencies
├── clients.py             vLLM client wiring for the backend
├── search/                /api/search router + search core
├── media/                 /api/media /thumbnail /chunk-frame router
└── system/                health / system router

frontend/                  primary SvelteKit 2 / Svelte 5 SPA (the viewer UI)
├── src/lib/api.ts         typed, Zod-validated client for the backend (the ONLY data boundary)
├── src/routes/+page.svelte single stateful orchestrator (search/browse/layout/pagination)
├── src/lib/components/     presentational components (SearchBar, HitCard, PlayerPane, …)
├── src/lib/feature-flags.svelte.ts  $state singleton (suppress chunk-frame 404 storms)
└── server.ts              Bun static server + /api/* reverse proxy (Range-aware)

demo/                      SEPARATE in-browser transformers.js/WebGPU demo (see §9)
Makefile                   end-to-end developer commands (source of truth for how it runs)
pyproject.toml             uv package, [multimodal] extra, ruff/ty/pytest config
tests/                     pytest: unit (pure logic) + dataset-gated backend smoke
```

---

## 7. The shared seam: `vllm/` clients

This is the most important coupling fact for onboarding. The `vllm/` clients
— `VLLMEmbeddingClient` (`embedding.py`, the bi-encoder) and `VLLMReranker`
(`reranker.py`, the cross-encoder) — are used by **two** consumers with different
concurrency needs:

- **CLI batch path** (`feature text_embedding` / `feature frame_embedding`): floods vLLM's
  continuous batcher with a `ThreadPoolExecutor` (`text_concurrency=32`,
  `image_concurrency=8`) for ~10–15× throughput over serial RTT.
- **Backend serving path** (`/api/search`): one query at a time, lazily connects,
  and lets `httpx.HTTPError` propagate so the API layer can translate it into a
  single structured **503** ("embedding service unavailable"). The error boundary
  lives in `backend/search/service.py` (`run_search` wraps the embed calls),
  **not** in the client — keep it that way.

A planned async-per-query path for the backend (see [TODO.md](TODO.md)) must
**preserve the sync ThreadPoolExecutor batch path** — the two coexist.

Two subtleties worth knowing:
- **Reranker double-scaffolding:** the model-card prefix/suffix framing is
  duplicated between the `_PREFIX`/`_SUFFIX` constants in `vllm/reranker.py`
  (which build `/v1/rerank` strings) and `retrieval/qwen3_vl_reranker.jinja` (the chat
  template the server applies). They must stay in sync; treat edits as risky.
- **Image pinning:** `_IMAGE_SIDE = 392` square center-crop (`vllm/image.py`) is a
  vLLM warmup-buffer workaround. vLLM sizes the Qwen3-VL deepstack buffer once at
  warmup, so every runtime image must yield the same vision-token count or the
  engine dies with `num_tokens > buffer`. Sending each image at exactly the area
  the server pins via `min_pixels == max_pixels` (392 × 392 = 153 664 px — see the
  `embed-server` / `embed-server-docker` Makefile targets) keeps the runtime token
  count at the warmup ceiling. (448 × 448 = 200 704 px overran it — the recurring
  crash.) Aspect ratio is sacrificed on purpose; if you change the side length,
  change the Makefile pixel pin to match (`side² == pin`).

---

## 8. Design decisions & gotchas (the load-bearing rationale)

These are choices that look odd until you know why. Don't "fix" them blindly.

- **Pydantic v2 in `model/datamodel.py`.** Typed decode of easytranscriber JSON
  at the ingest boundary (`AudioMetadata.model_validate_json`). A shared `_Base`
  sets `ConfigDict(extra="ignore")` so upstream adding fields never breaks ingest,
  and list fields use `Field(default_factory=list)` — never bare `= []`, which
  would share one mutable list across instances.
- **Lazy *module* imports from CLI commands / the backend.** The heavy modules
  (`lance`, `torch`-backed ASR, the `vllm/` clients) are imported inside
  the function body that needs them, so `raudio --help` stays instant and the
  optional `[multimodal]`/transcribe extras stay optional. Within those modules
  imports are normal top-level — the optionality comes from *the module not being
  imported at startup*, not from scattering imports inside methods. (Verified:
  `import raudio` pulls no `httpx`/`torch`/`PIL`.)
- **`_Ctx` class as CLI global state.** `--db` / `--table` are stashed on a
  module-level class by the root callback rather than threaded through every
  signature. It's process-global; the idiomatic Typer alternative is `ctx.obj`.
- **Eager DB open inside `create_app` (not a lifespan).** The backend holds
  read-only Lance handles — no pool to dispose, no async driver — so the skill's
  lifespan rule mostly doesn't apply. Cost: `create_app` needs a real dataset
  (it raises if `chunks` is missing), which is exactly what the TestClient smoke
  test exercises.
- **FTS language must be `"Swedish"`.** The default English stemmer can't reduce
  Swedish forms (`ministern`, `vägen`, `ansåg`), so those queries return zero
  hits. `with_position=True` enables phrase queries; `remove_stop_words=False`
  keeps stop words so quoted phrases match verbatim.
- **`ThreadPoolExecutor` (not Process) for ffmpeg in `frames.py`.** ffmpeg is a
  subprocess (GIL released during the wait) and threads dodge the "lance is not
  fork-safe" warning the fork start method triggers.
- **`ensure_vector_index` refuses to build with NULLs present.** Lance's IVF_PQ
  builder mishandles partially-NULL vector columns, so embeds run to completion
  before the index is built.
- **vLLM version/GPU pins live in the Makefile** with comments: the `cu128` torch
  pin (driver 570.x → CUDA 12.8 ceiling), vLLM run via `uvx` to avoid torch-pin
  conflicts, embed/rerank pinned to distinct GPUs to dodge a memory-profiling
  race, and `min==max` pixels to avoid the deepstack overflow bug. **Keep these
  comments** — they encode hours of debugging.
- **Frontend: `api.ts` is the only place untrusted JSON enters**, and every
  response is Zod-parsed (`asJson`). Backend schema drift surfaces as a clean
  `ApiError`, not a silent mis-render. Never add a fetch that returns un-parsed JSON.
- **Theme reactivity is local to `theme-toggle.svelte`** on purpose — an earlier
  `theme.svelte.ts` store didn't re-render reliably in production builds.

---

## 9. The `demo/` app — a separate subsystem

`demo/` is a **standalone, in-browser** audio-transcription playground built on
**transformers.js + ONNX Runtime Web** running Whisper/KB-Whisper on **WebGPU**
(WASM fallback), in a Web Worker. It supports realtime mic/tab capture and batch
file transcription, exports txt/srt/json/wav, and builds to static HTML for
Hugging Face Spaces. **It shares no code, types, or deps with raudio** and targets
a totally different runtime (browser, not server/GPU/vLLM).

Treat it as a quarantined playground — it should not be held to the core's
standards/urgency. Architecture notes if you do touch it:
- One Web Worker is shared by three consumers (`+page.svelte` lifecycle,
  `RealtimePanel` + `BatchPanel` inference) over one `postMessage` channel,
  disambiguated by a `jobId` convention — an implicit, untyped protocol (a
  candidate for a typed message contract; see [TODO.md](TODO.md)).
- The whole app only mounts if `navigator.gpu` exists; `webgpu.ts` does the
  auto/force/fallback logic.
- Run it: `cd demo && bun install && bun run dev` (needs Chrome/Edge 113+).

---

## 10. Developer workflow

### Toolchain (mandated by the `writing-python` / `writing-typescript` skills)

```bash
# Python — from the repo root
uv sync --group dev          # install runtime + dev deps (pytest, httpx)
uvx ruff check src backend tests        # lint  (config in pyproject.toml; lint-only — see note)
uvx ruff check --fix src backend tests  # autofix
uvx ty check                            # type-check
uv run pytest                           # tests (unit always; backend smoke if dataset present)
uv run raudio --help                    # CLI smoke

# Frontend (and demo) — from frontend/ (or demo/)
bun install
bun run check                # svelte-kit sync + svelte-check (the type gate)
bun run build                # static SPA into build/
```

> **Note on `ruff format`:** it is intentionally *not* run in CI. The codebase
> uses hand-aligned dict literals and box-drawing comment separators that the
> formatter would reflow into a large, low-signal diff. Run `uvx ruff format`
> manually only if you want to adopt it wholesale. Lint and types *are* enforced.

### The verification gates (what "green" means)

| Surface | Command | Bar |
|---|---|---|
| Python lint | `uvx ruff check src backend tests` | 0 issues |
| Python types | `uvx ty check` | 0 diagnostics |
| Python tests | `uv run pytest` | all pass (backend smoke auto-skips without a local dataset) |
| CLI | `uv run raudio --help` | exit 0 |
| Frontend types | `cd frontend && bun run check` | 0 errors / 0 warnings |
| Frontend build | `cd frontend && bun run build` | succeeds |

### Tests

- `tests/test_units.py` — pure logic, runs everywhere: `timecode`, `_parse_range`
  (HTTP Range), `_build_where_clause` (SQL predicate assembly + quote escaping),
  `_rrf_fuse`, query-term extraction.
- `tests/test_backend_smoke.py` — end-to-end against a real local
  `transcripts.lance`; **auto-skips** when the dataset isn't present (it's
  gitignored). Covers FTS, health, documents, thumbnail, Range streaming, and
  asserts GPU-only modes degrade to a clean **503** (never a 500).

### Running the whole thing

See the [README](README.md) quickstart and the `Makefile` (`make backend`,
`make frontend`, `make embed-server-docker`, `make pipeline`, …). The Makefile is
the authoritative, commented description of how every process fits together.

---

## 11. Quick "where do I look for…?" index

| I want to… | Look at |
|---|---|
| Change the table schema | `src/raudio/model/schema.py` |
| Change how transcripts become rows | `src/raudio/ingest/ingest.py` (`flatten_chunks`, `_document_row`) |
| Add/modify a CLI command | `src/raudio/cli/` |
| Add/modify a feature column (embed/summary/caption) | `src/raudio/features/columns.py` (+ `features/engine.py`) |
| Change search behavior / add a mode | `backend/search/service.py` (`run_search`, `_vector_search`, `_frame_search`, `_rrf_fuse`) |
| Add an API endpoint | the relevant router under `backend/` (`search/`, `media/`, `system/`) |
| Change the embedding/rerank wire format | `src/raudio/vllm/embedding.py` (+ `vllm/reranker.py`, `retrieval/qwen3_vl_reranker.jinja`) |
| Touch the search UI | `frontend/src/routes/+page.svelte` + `frontend/src/lib/components/` |
| Change the API client / response shapes | `frontend/src/lib/api.ts` (Zod schemas) |
| Change how processes are launched | `Makefile` |
| Understand a Lance/vLLM quirk | the comments in the relevant module (they're load-bearing) + §8 here |
