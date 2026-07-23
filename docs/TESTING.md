# Testing strategy — what we test, what we don't, and why

Audience: a developer who opened `tests/` and wants to know what these tests are
protecting, where the seams are, and whether we're testing the right things.
This document answers that directly. It judges **value**, not coverage %.

All references use the current package layout — `src/ratch/{cli,clients,core,
features,ingest,modalities,model,retrieval}/…`, `runners/{asr,diarize,voiceprint,
topics,kg}/`, and the split services `services/{common,viewer,search,annotator}/`. Every
behaviour called "verified" below runs offline (no GPU, no network, no ffmpeg)
via deterministic fakes and tmp-dir Lance datasets.

---

## 1. Verdict — are we testing the right stuff?

**Yes.** The suite sits at the right altitude: branchy pure logic that regresses
silently, plus integration tests that build a *real* Lance dataset in a tmp dir
and exercise the genuine `LanceTable.search()` / `add_columns` paths with an
offline embedder — never a mocked Lance. Nothing in the suite depends on the
local corpus. (The script-level full-pipeline smoke `scripts/e2e_smoke.py`
drives the real CLI against a live embed server into a throwaway DB — see
section 5 — and the browser E2E suites live in `frontend/apps/media/e2e/`.)

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

1. `ratch`, `runners`, and the service packages (`common`/`viewer`/`search`/
   `annotator`) are installed (editable), and `tests/` has no `__init__.py`, so
   pytest's default import mode puts `tests/` on `sys.path` — that's how
   `from fakes import …` resolves the shared doubles in `tests/fakes.py`.
   There is no `conftest.py`.
2. The whole suite runs with no dataset, no GPU, and no network (Ray-cluster
   tests spin a local `ray.init` and are the slowest files).

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
- **Live-stack proof outside pytest**: the browser E2E suites (51 checks) and
  `scripts/e2e_smoke.py` assert product contracts end-to-end (FTS hits,
  503-when-vLLM-down, 4xx-not-500, Range streaming).

### We deliberately DON'T test (and why it's the right call)

| Boundary | Examples | Why we skip |
|---|---|---|
| **vLLM HTTP transport** | `VLLMEmbeddingClient.embed_text`/`embed_image` (`clients/embedding.py`), `VLLMReranker.rerank` (`clients/reranker.py`), `caption`/`summarize` clients | Mocking httpx just restates our own request shape. The *pure* helpers are extracted to `clients/image.py` (`l2_normalize`, `image_to_data_url`, `_square_crop`) and tested there; the lazy-construction + 503 seam is tested via fakes in the search-service tests. |
| **ffmpeg subprocess** | `extract_chunk_frame`, `extract_chunk_frames_parallel`, `generate_thumbnails`, `_render_waveform` (`modalities/av/frames.py`, `modalities/av/thumbnails.py`) | Either real ffmpeg (an opt-in integration test) or mocking `subprocess.run` (a change-detector against our own argv). The valuable byte-parsing it delegates to — `_jpeg_dimensions` — and the pure sampling math — `sample_times` — are tested separately. |
| **torch / transformers / ctranslate2 GPU** | `run_transcribe` (`runners/asr/transcribe.py`), the GPU-coupled body of `detect_and_sort` (`runners/asr/detect_language.py`) | Pure model boundaries needing the models + a device. The one piece of real logic — the sample-window planner `_plan_sample_starts` — is extracted and unit-tested (`test_detect_language.py`). |
| **Declarative constants** | `CHUNK_SCHEMA` / `DOC_SCHEMA` / `CHUNK_FRAMES_SCHEMA` / `*_STORAGE_VERSION` (`model/schema.py`), `ISO_639_3_TO_1` | Asserting field names/types is a change-detector with zero bug-catching power. The ingest round-trip validates schema *conformance* for free. |
| **Thin arg-forwarding wrappers** | `ingest_document` → `ingest_many`, `reindex_fts`, the Typer CLI commands (`cli/`) | Plumbing, covered transitively by the round-trip and feature tests. Standalone tests would re-create the same fixtures with more mocking and less meaning. |

---

## 3. Current coverage — the test files

The suite is grouped by seam rather than enumerated per file (41 files — the
per-file detail is the files' own docstrings; this list won't rot):

- **Core contracts** — `test_core_contract.py` (`ratch.core` imports no
  modality/client code; `import ratch` loads zero model deps, not even ray),
  `test_registry.py` (Stage validation), `test_core_dataset.py` (create/append/
  overwrite invariants), `test_driver_wiring.py` + `test_runner_convention.py`
  (real local-Ray scan/append drivers; fake runner proves "new model = 1 runner
  dir + 1 Stage entry"), `test_core_jobs.py` (the Ray Jobs seam vs an id-honest
  fake client), `test_features_topics_seam.py` (topics feature → jobs argv/token
  contract), `test_lineage.py` (OpenLineage facets).
- **Pipeline compute** — `test_ingest.py` / `test_ingest_documents.py` /
  `test_audio.py` (write path, real tmp-dir Lance incl. Blob V2),
  `test_embed.py` / `test_features.py` / `test_features_columns.py` /
  `test_indexing.py` (the type-agnostic column engine + indexes),
  `test_asr_detect.py` / `test_asr_duration.py` / `test_detect_language.py`
  (runners/asr logic, GPU body stubbed), `test_voiceprint.py`,
  `test_kg_adapter.py`, `test_retrieval_search.py`, `test_materialize.py`,
  `test_vllm_schemas.py` (client wire contract), `test_units.py` (pure helpers).
- **Services (split backend)** — `test_media_api_{core,atlas,diarization,graph,
  topics,voice}.py` (viewer read plane over synthetic Lance),
  `test_search_api_{service,spec}.py` (modes, degradation branches, clamps),
  `test_annotate.py` + `test_assist.py` + `test_jobs.py` +
  `test_cli_maintain.py` (annotator write plane: merge_insert + 409, assist
  mock, LabelOp jobs, version GC), `test_lancekit_{descriptor,predicate,reader,
  writer}.py` (the shared kernel + catalog-transport parity).

Shared doubles: `tests/fakes.py` (deterministic `FakeEmbedClient` — each exact
string maps to its own unit vector — `FakeReranker`, caption/summarize fakes,
and builders that write **real** on-disk Lance tables through the production
helpers).

---

## 4. Key seams that make this testable

The design choices that let the suite stay offline and fast:

- **Framework-free search core.** `backend.search.service.run_search` takes the
  two vLLM client getters as plain `Callable`s and a `SearchSpec`; it never
  imports the FastAPI app or app state. `test_search_api_service.py`
  calls it (or the app over `TestClient`) with a
  `FakeEmbedClient`, so the genuine Lance query builder runs without a GPU.
- **`SearchSpec` is pure validation.** No Lance/HTTP/embedding deps, so the
  clamp/enum logic is unit-tested in isolation (`test_search_spec.py`).
- **Lazy, monkeypatchable vLLM clients.** The accessors in `backend/clients.py`
  import `ratch.vllm.*` *inside* the function and cache the instance on
  `AppState`, giving a clean swap point and a 503-on-failure contract
  (`test_search_api_service.py`).
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
# The full suite (no dataset, no GPU, no network):
uv run pytest tests/ -m "not slow"

# A single file:
uv run pytest tests/test_ingest.py -v
```

There is **no `make test` target** — the `uv run … pytest` invocations above are
the way to run the suite. `pytest`, `httpx`, and `pytest-cov` live in the `dev`
dependency group (`pyproject.toml [dependency-groups]`), so
`uv sync --group dev && uv run pytest …` is the equivalent of the
`--with pytest --with httpx` form shown above. `httpx` is needed by anything that
constructs the app via `fastapi.testclient.TestClient` (the backend tests).

### Frontend E2E (real browser, WebGPU) — three suites

`frontend/apps/media/e2e/` drives the REAL app in a headless Chromium **with
WebGPU/Vulkan live** (not jsdom, not a smoke). `bun run test:e2e` runs all three;
shared plumbing (chromium resolution across both playwright layouts, launch args,
preconditions, PASS/FAIL collector) lives in `e2e/lib.mjs`:

- **`annotator.e2e.mjs`** — the image tool palette: every drawing tool commits
  (rect/point/line/polygon/pencil/brush + magnetic corner-snap incl. a live SNAP
  assertion; lasso multi-selects), AI-assist Detect (GroundingDINO) + SAM Segment,
  draw → save → persist-across-reload. This suite caught the orphaned CV tools and
  the swallowed Enter-commit.
- **`temporal.e2e.mjs`** — the audio + video viewers against fixture media in
  `static/e2e/` (`tone.wav`, `clip.mp4` — the demo doc has no media blob): audio
  waveform mounts → drag-creates a segment → region-resize round-trips
  `t_start/t_end` → persists, times shown in the review list; video scrubs → frame
  snapshot under the overlay → a rect drawn on the paused frame is pinned at the
  playhead (`t_start≈currentTime`) → persists. Deep-link overrides used:
  `/annotate?keys=…&kind=audio|video&media=/e2e/…` (same-origin only).
- **`read-plane.e2e.mjs`** — WebGPU adapter, SavedViews save/apply/delete, and the
  compare-versions History panel (`/versions` fetch + version rows).

```bash
# Preconditions: services up (make services-up → viewer:8101 search:8102 annotator:8103;
#   MEDIA_ASSIST_URL unset → deterministic assist mocks), frontend dev proxy on :5175
# + `bun run dev --port 5175` (apps/media) running.
cd frontend/apps/media && bun run test:e2e
# Chromium: auto-resolved from ~/.cache/ms-playwright (full build, NOT headless-shell —
# that lacks WebGPU); override with E2E_CHROME=/path/to/chrome. E2E_KEY overrides the
# demo unit.
```

Each suite re-seeds the demo annotations (`make seed-annotations`) before + after, so
runs are deterministic and leave the demo clean. NOT covered here: annotations over the
S3 backend (`MEDIA_S3_*` — read paths live-verified separately; the annotation WRITE
plane over S3 was live-verified 2026-07-21 against MinIO: `materialize-blobs` → managed
media_blob streams over S3, then wire GET + save-insert commit + stale-base-version 409 +
`?version` time-travel + tag batch, all against an S3-backed backend — the browser suite
itself still runs against the local dataset), and CI wiring (merge-side).

### Full-pipeline e2e smoke

`scripts/e2e_smoke.py` drives the actual CLI end-to-end into a throwaway temp DB
(`tempfile.mkdtemp`, never touches your real DB) and **needs the vLLM embed
server on `:8001`**:

```bash
# Full-pipeline e2e (real CLI end-to-end into a throwaway temp DB; needs :8001):
make e2e-smoke
# = uv run --extra multimodal python scripts/e2e_smoke.py --docs 2 --frame-limit 24
```

It runs `thumbnail` → `ingest` → embed-chunks
(`feature text_embedding --no-create-index`) → `extract-chunk-frames` →
embed-chunk-frames (`feature frame_embedding --no-create-index`), then boots the
backend over `TestClient` and asserts `/api/health`, `/api/documents`, the
`fts`/`semantic`/`hybrid`/`all` + `visual` search modes, `/api/thumbnail`,
`/api/media` Range→206, and `/api/chunk-frame`. It reads sample data from
`output/sv-test/alignments` + `input/sv-test`, requires the `multimodal` extra,
and prints the temp path to delete afterward.

**Where new tests go:**
- Pure helpers → `test_units.py` (or a focused `test_<area>.py`). No skip guard.
- Real-Lance integration → a tmp-dir test using `tmp_path` + `tests/fakes.py`.
  No dataset gate and no GPU — it builds its own DB in ~tens of milliseconds.
- Anything needing the local corpus → guard with `pytest.mark.skipif` on the
  dataset path so CI without the dataset stays green.
