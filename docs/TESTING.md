# Testing strategy — what we test, what we don't, and why

Audience: a developer who opened `tests/` and wants to know what these tests are
protecting, where the seams are, and whether we're testing the right things.
This document answers that directly. It judges **value**, not coverage %.

All references use the current package layout — `src/raudio/{model,ingest,media,
asr,vllm,features,retrieval,cli}/…` and the split FastAPI backend
`backend/{app,state,deps,clients}.py` + `backend/{search,media,system}/`. Every
behaviour called "verified" below runs offline (no GPU, no network, no ffmpeg)
via deterministic fakes and tmp-dir Lance datasets.

---

## 1. Verdict — are we testing the right stuff?

**Yes.** The suite sits at the right altitude: branchy pure logic that regresses
silently, plus integration tests that build a *real* Lance dataset in a tmp dir
and exercise the genuine `LanceTable.search()` / `add_columns` paths with an
offline embedder — never a mocked Lance. The only thing we lean on the local
corpus for is the end-to-end `test_backend_smoke.py`, which skips cleanly when
the dataset is absent.

The guiding rule, applied throughout: **we never mock the world.** If a
function's only logic is assembling an ffmpeg argv, an httpx request body, or a
column list immediately handed to lancedb, a mocked test would just restate the
implementation. Those boundaries are covered by one opt-in integration test or
by testing the *pure* helper they delegate to (e.g. `_jpeg_dimensions`,
`l2_normalize`, `compose_media_uri`) — not by a change-detector.

Both halves of the system are now covered:

- **Read / query** — search-mode routing across all seven modes (`fts`,
  `semantic`, `visual`, `scene`, `scene_fts`, `hybrid`, `all`), RRF fusion, the
  cross-encoder rerank head, SQL-filter composition, HTTP Range streaming, and
  the lazy vLLM client / DI seam.
- **Write / ingest** — the transcriber-JSON → Lance transform (`flatten_chunks`,
  `_pick_alignments`), a full ingest round-trip, idempotent re-ingest, and the
  `add_columns` feature pipeline (text/frame embeddings, summary, caption).

Two facts worth knowing about the harness:

1. `backend` and `raudio` are installed (editable) packages, and `tests/` has no
   `__init__.py`, so pytest's default import mode puts `tests/` on `sys.path` —
   that's how `from fakes import …` resolves the shared test doubles in
   `tests/fakes.py`. There is no `conftest.py`.
2. `test_backend_smoke.py` is the only file gated on the local corpus
   (`transcripts.lance/chunks.lance`). Everything else runs in CI with no
   dataset, no GPU, and no network.

---

## 2. Philosophy — what we test vs what we deliberately DON'T

### We test (high value)

- **Branchy, deterministic pure logic** where a regression is silent and likely:
  timecode formatting, HTTP Range math, doc-id validation, SQL quote-escaping,
  RRF fusion, byte-level JPEG marker parsing, URI/MIME composition,
  alignment-JSON decoding, language-window planning, and the `SearchSpec`
  Pydantic clamps.
- **Real Lance integration, no GPU/network**: ingest round-trips, the
  `add_columns` embedding pipeline, and the backend search modes — all against a
  tmp-dir dataset with a deterministic fake embedder, so the genuine query
  builder and data-evolution paths actually run.
- **One dataset-gated end-to-end smoke** over the real corpus, asserting product
  contracts (FTS hits, 503-when-vLLM-down, 4xx-not-500, Range streaming).

### We deliberately DON'T test (and why it's the right call)

| Boundary | Examples | Why we skip |
|---|---|---|
| **vLLM HTTP transport** | `VLLMEmbeddingClient.embed_text`/`embed_image` (`vllm/embedding.py`), `VLLMReranker.rerank` (`vllm/reranker.py`), `caption`/`summarize` clients | Mocking httpx just restates our own request shape. The *pure* helpers are extracted to `vllm/image.py` (`l2_normalize`, `image_to_data_url`, `_square_crop`) and tested there; the lazy-construction + 503 seam is tested via fakes in `test_backend_clients.py`; the smoke test covers the 503-when-down contract end-to-end. |
| **ffmpeg subprocess** | `extract_chunk_frame`, `extract_chunk_frames_parallel`, `generate_thumbnails`, `_render_waveform` (`media/frames.py`, `media/thumbnails.py`) | Either real ffmpeg (an opt-in integration test) or mocking `subprocess.run` (a change-detector against our own argv). The valuable byte-parsing it delegates to — `_jpeg_dimensions` — and the pure sampling math — `sample_times` — are tested separately. |
| **torch / transformers / ctranslate2 GPU** | `run_transcribe` (`asr/transcribe.py`), the GPU-coupled body of `detect_and_sort` (`asr/detect_language.py`) | Pure model boundaries needing the models + a device. The one piece of real logic — the sample-window planner `_plan_sample_starts` — is extracted and unit-tested (`test_detect_language.py`). |
| **Declarative constants** | `CHUNK_SCHEMA` / `DOC_SCHEMA` / `CHUNK_FRAMES_SCHEMA` / `*_STORAGE_VERSION` (`model/schema.py`), `ISO_639_3_TO_1` | Asserting field names/types is a change-detector with zero bug-catching power. The ingest round-trip validates schema *conformance* for free. |
| **Thin arg-forwarding wrappers** | `ingest_document` → `ingest_many`, `reindex_fts`, the Typer CLI commands (`cli/`) | Plumbing, covered transitively by the round-trip and feature tests. Standalone tests would re-create the same fixtures with more mocking and less meaning. |

---

## 3. Current coverage — the test files

`tests/fakes.py` is the shared backbone: a deterministic `FakeEmbedClient`
(maps each exact string to its own unit vector, so an exact-text query is its
own nearest neighbour), `FakeReranker`, `FakeCaption`/`FakeSummarize`, and
builders that write **real** on-disk Lance tables through the production helpers
(`ingest_many`, `raudio.media.frames.write_chunk_frames`). The no-GPU tests
dogfood the real writers instead of hand-rolling Arrow tables.

| File | Scope | Highlights |
|---|---|---|
| `test_units.py` | Pure read-path helpers, no I/O | `timecode` (HH:MM:SS vs MM:SS + millis rounding), `parse_range` (closed/open/suffix/clamp/malformed), `_build_where_clause` (single-quote SQL-injection guard), `_rrf_fuse` (dedup on `(doc_id, chunk_id)` + summed-score order), `extract_query_terms`, `valid_doc_id`, `parse_alignments_json`, `sample_times`. |
| `test_search_spec.py` | `SearchSpec` / `SearchMode` validation, no dataset | Clamps: `n`→[1, 200], `rerank_n`→[1, 200] (default 20), `fuzziness`→[0, 2], `weight`→[0, 1] (None passes through). `mode` is a `StrEnum` (unknown value rejected; FastAPI maps to 422 at the route). `extra="ignore"` drops unknown fields. |
| `test_audio.py` | `compose_media_uri` + `guess_mime`, pure | base-URI trailing-slash normalization (exactly one `/`), base-over-source precedence, `file://` fallback, basename-only key; MIME never None. |
| `test_ingest.py` | Write path, real tmp-dir Lance | `_pick_alignments` closed-interval containment + float-cast words; `flatten_chunks` language fallback, `speech_id` None→0 / str→int, per-speech `chunk_id` enumerate, `text` None→`''`, metadata join by `bildid` stem; `load_metadata_csv` / `_metadata_for`; `_doc_id` stable 16-hex; and the round-trip: `ingest_many` then `nearest_chunks("brown")` finds the chunk. |
| `test_embed.py` | `add_columns` embedding pipeline, real Lance | `embed_text_column` attaches `text_embedding` and round-trips; `embed_frame_column` keys vectors by `_rowid` (each row gets *its own* frame's embedding regardless of scan order); brute-force vector search finds the planted nearest; re-ingest of the same doc is idempotent. |
| `test_features.py` | Type-agnostic feature engine | The engine attaches non-vector columns (int, string) the same way it attaches embeddings; `summary` / `caption` round-trip; `upsert_scan_column` only-null vs overwrite modes; `FEATURES` registry well-formed. |
| `test_detect_language.py` | `_plan_sample_starts`, pure math | Windows spread across the whole file, never run past EOF, short-file collapse to a single start, non-positive duration safe. |
| `test_backend_clients.py` | Lazy vLLM accessors + DI seam | `ensure_embedder` / `ensure_reranker` cache-then-construct; construction failure maps to **503**; `get_embedder` / `get_reranker` / `get_state` bind to `app.state.resources`. Uses the documented monkeypatch seam (the deferred `raudio.vllm.*` imports inside the accessors). |
| `test_backend_search.py` | Search modes, no GPU | Injects `FakeEmbedClient` so `semantic` / `visual` / `hybrid` / `all` actually run the real `LanceTable.search()` chain (the sync `.search()` vs async-only `.query()` path). Semantic ranks the planted nearest; GET and POST both covered; `all` + rerank runs. |
| `test_backend_service.py` | `run_search` error & degradation branches | Against a real chunks table with **no** embedding columns: `semantic`/`hybrid`-without-embeddings → 400, `hybrid`-without-text → 400, `visual`-without-frames → empty, `all`-without-embeddings falls back to FTS, `_vector_search`/`_frame_search` missing-column → empty, `_postprocess_hits` parses + pops `alignments_json`. The reranker getter raises if touched — none of these paths should construct it. A `TestSceneSearch` class additionally builds a captioned tmp dataset (`caption` + `caption_embedding` via offline fakes) to cover the scene legs: `scene` ranks frames over `caption_embedding` and joins back to chunks, `scene_fts` runs BM25 over the `caption` text, captions ride along on every mode's hits, and both scene modes degrade to `[]` (not a 500) when frames/captions are absent. |
| `test_backend_media.py` | Media endpoints, real Blob V2 | Builds a tmp documents table with an External `media_blob` URI + Inline `thumbnail`, mirroring ingest. Covers thumbnail inline JPEG, full + ranged (206) + suffix + unsatisfiable (416) media GETs, invalid-doc-id 400, and `chunk-frame` (404 until frames exist, `frame_idx` selection). |
| `test_backend_smoke.py` (dataset-gated) | Real corpus end-to-end | Skips cleanly without `transcripts.lance/chunks.lance`. Asserts real contracts: health DB facts, FTS hits, empty-query→empty, unknown mode→**400/422**, semantic-without-vLLM→**503 not 500**, documents pagination, thumbnail + media Range round-trip, invalid doc-id→400. |

---

## 4. Key seams that make this testable

The design choices that let the suite stay offline and fast:

- **Framework-free search core.** `backend.search.service.run_search` takes the
  two vLLM client getters as plain `Callable`s and a `SearchSpec`; it never
  imports the FastAPI app or app state. `test_backend_service.py` /
  `test_backend_search.py` call it (or the app over `TestClient`) with a
  `FakeEmbedClient`, so the genuine Lance query builder runs without a GPU.
- **`SearchSpec` is pure validation.** No Lance/HTTP/embedding deps, so the
  clamp/enum logic is unit-tested in isolation (`test_search_spec.py`).
- **Lazy, monkeypatchable vLLM clients.** The accessors in `backend/clients.py`
  import `raudio.vllm.*` *inside* the function and cache the instance on
  `AppState`, giving a clean swap point and a 503-on-failure contract
  (`test_backend_clients.py`).
- **Pure helpers extracted from I/O boundaries.** `vllm/image.py`
  (`l2_normalize`, `image_to_data_url`), `media/frames.py` (`_jpeg_dimensions`,
  `sample_times`), `asr/detect_language.py` (`_plan_sample_starts`),
  `ingest/audio.py` (`compose_media_uri`, `guess_mime`) — each testable without
  the surrounding ffmpeg/torch/httpx machinery.
- **Real writers in fakes.** `tests/fakes.py` builds Lance tables via
  `ingest_many` and `write_chunk_frames`, so integration tests exercise the
  production schema-promotion path (including the `alignments_json` →
  `pa.json_()` extension-type promotion and idempotent re-ingest).

---

## 5. How to run + where tests live

Tests live in `tests/` (`pyproject.toml [tool.pytest.ini_options]`:
`testpaths = ["tests"]`, `addopts = "-ra -q"`). There is no `conftest.py`; shared
doubles are in `tests/fakes.py`, importable because pytest puts `tests/` on
`sys.path`.

```bash
# Everything except the dataset-gated smoke (no dataset, no GPU, no network):
uv run --with pytest --with httpx pytest --ignore=tests/test_backend_smoke.py

# A single file:
uv run pytest tests/test_ingest.py -v

# The full suite (smoke auto-skips when the corpus is absent):
uv run --with pytest --with httpx pytest
```

`httpx` is needed by anything that constructs the app via `fastapi.testclient.
TestClient` (the backend tests).

**Dataset-gated skip pattern** (from `test_backend_smoke.py`) — reuse this for
any test that needs the local corpus so CI without the dataset stays green:

```python
from pathlib import Path

import pytest

DB_PATH = Path(__file__).resolve().parent.parent / "transcripts.lance"

pytestmark = pytest.mark.skipif(
    not (DB_PATH / "chunks.lance").exists(),
    reason="local transcripts.lance dataset not present",
)
```

**Where new tests go:**
- Pure helpers → `test_units.py` (or a focused `test_<area>.py`). No skip guard.
- Real-Lance integration → a tmp-dir test using `tmp_path` + `tests/fakes.py`.
  No dataset gate and no GPU — it builds its own DB in ~tens of milliseconds.
- Anything needing the real corpus → gate it with the skip pattern above.
