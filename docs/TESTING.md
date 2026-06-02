# Testing strategy — what we test, what we don't, and why

Audience: a developer who looked at the two test files and thought "I don't get
what these are protecting, and I'm not sure we're testing the right things."
This document answers that directly. It judges **value**, not coverage %.

All file/line references use the post-refactor package layout
(`src/raudio/{model,ingest,media,asr,vllm,features,retrieval}/…`, `backend/app.py`). Every
behavior called "verified" below was run offline (no GPU, no network, no
ffmpeg) before this doc was written.

---

## 1. Verdict — are we testing the right stuff?

**The tests you HAVE are the right stuff. The problem is the half you're NOT
testing.**

`test_units.py` and `test_backend_smoke.py` are not coverage-theater. They sit
at exactly the right altitude: branchy pure logic that regresses silently, plus
a dataset-gated integration smoke that asserts real product contracts (not
mocks). Keep both as-is.

But they only cover the **read / query** half of the system — search, Range
streaming, RRF fusion. The entire **write / ingest** half has **zero** tests,
and that is where a silent corruption is both most likely and most damaging: a
bug in `flatten_chunks` or `_pick_alignments` writes a wrong-but-parseable value
into *every* Lance row and **nothing raises**. The smoke test cannot see it,
because it reads back whatever ingest wrote.

This is not hypothetical. I ran an `ingest_many` round-trip into a tmp dir
(0.034 s, fully offline). The first ingest is correct. **The second ingest into
the same DB currently RAISES:**

```
RuntimeError: lance error: Append with different schema:
`alignments_json` should have type json but type was large_binary
```

`_build_chunks_table` (`ingest/ingest.py:296-302`) rebuilds the column as
`large_binary`, but the on-disk table promoted it to the `json` extension type,
so `table.add` (`ingest/ingest.py:361`) rejects the append. **This regression is
live in the current refactored state and is invisible without a write-path
test.** It is the single strongest argument that we're testing the wrong *scope*.

Two structural weaknesses (not flaws in *what* the existing tests assert):

1. `test_units.py` is the only thing that runs in CI without the dataset, yet it
   leaves the new Pydantic validation layer (`SearchSpec` clamps) and the whole
   write path uncovered.
2. The smoke fixture is `scope="module"` and gated on `transcripts.lance`
   existing, so on a CI box without the dataset it hides *all* backend behavior —
   including the dataset-free `SearchSpec` clamps, which deserve their own
   no-dataset home in `test_units.py`.

---

## 2. Philosophy — what we test vs what we deliberately DON'T

### We test (high value)
Branchy, deterministic, pure logic where a regression is **silent and likely**:
parsing, interval containment, SQL quote-escaping, MRL math, byte-level JPEG
marker parsing, URI normalization, Pydantic clamps. Plus **one** no-GPU
integration test (a real Lance ingest round-trip into a tmp dir) — that counts
as high value because it exercises the schema + writer + reader + append in one
deterministic test and, today, lands red on the append bug.

### We deliberately DON'T test (and why it's the right call)

| Boundary | Examples | Why we skip |
|---|---|---|
| **vLLM HTTP** | `VLLMEmbeddingClient` (`embed_text`/`embed_image`, `vllm/embedding.py`), `VLLMReranker.rerank` + `QwenVLReranker` (`vllm/reranker.py`) | Mocking the HTTP transport just restates our own request shape. The smoke test already covers the 503-when-down contract at integration altitude. |
| **ffmpeg subprocess** | `extract_chunk_frame`, `extract_chunk_frames_parallel`, `generate_thumbnails`, `_extract_video_frame`, `_render_waveform` (`media/frames.py`, `media/thumbnails.py`) | Either real ffmpeg (an opt-in integration test, fine to skip for unit value) or mocking `subprocess.run` (a change-detector against the argv we wrote). The valuable byte-parsing it *delegates* to — `_jpeg_dimensions` — is tested separately. |
| **torch / transformers / ctranslate2 GPU** | `run_transcribe` (`asr/transcribe.py`), `_mms_probe`/`_whisper_probe`, the GPU-coupled body of `detect_and_sort` (`asr/detect_language.py`) | Pure model boundaries. Testing them needs the models + a device. The one piece of real logic (offset-voting) is inlined inside a 115-line fs/GPU/ffmpeg function — untestable without mocking the whole world. |
| **Thin arg-forwarding wrappers** | `ingest_document` → `ingest_many`, `reindex_fts`, `_build_chunks_table`/`_write_documents_table` | Plumbing. Covered transitively by the round-trip. Standalone tests would re-create the same Lance fixtures with more mocking and less meaning. |
| **Declarative constants** | `CHUNK_SCHEMA`/`DOC_SCHEMA`/`*_STORAGE_VERSION` (`model/schema.py`), `ISO_639_3_TO_1` | Asserting field names/types is a change-detector that breaks on every intentional edit with zero bug-catching power. The round-trip validates schema *conformance* for free. |
| **Mode-routing orchestration** | `_run_search`, `_vector_search` (`app.py:574+`) | Every non-FTS path needs a live Lance table *and* the vLLM client. The isolatable pure pieces (`SearchSpec`, `_build_where_clause`, `_rrf_fuse`) are already extracted — good design. The smoke test covers FTS + 503/400 end-to-end. |

The throughline: **we never mock the world.** If a function's only logic is
building an argv, a request body, or a column list that's immediately handed to
ffmpeg / httpx / lancedb, the mocked test would just restate the implementation.

---

## 3. Current coverage — existing tests, honest value

| Test (file) | Target | Value | Honest assessment |
|---|---|---|---|
| `TestTimecode` (`test_units.py:17`) | `retrieval/search.py:125` `timecode` | **High** | HH:MM:SS vs MM:SS branch + millis rounding (`3661.4567`→`.457`). Exactly the silent-regression formatter to lock. Keep. |
| `TestParseRange` (`test_units.py:33`) | `app.py:106` `_parse_range` | **High** | Closed / open-ended / suffix / clamp / malformed / start≥total. The HTTP Range edge matrix. Strong. Optional add: `bytes=-0`→None. |
| `TestBuildWhereClause` (`test_units.py:54`) | `app.py:158` `_build_where_clause` | **High** | Covers None / single / AND-join and the **single-quote-doubling SQL-injection guard** (`O'Brien`). Gap: `extraid` uses `=` and `referenskod` uses `LIKE` — those two branches are untested. |
| `TestRrfFuse` (`test_units.py:74`) | `app.py:178` `_rrf_fuse` | **High** | Dedup on `(doc_id, chunk_id)` + summed-score ordering + first-seen canonical row. Solid. |
| `TestExtractQueryTerms` (`test_units.py:89`) | `retrieval/search.py:38` `extract_query_terms` | **High** | Operator-strip + lowercase. Gap worth one line: assert stopword match is exact-token not substring (`android` must survive). |
| `test_backend_smoke.py` (7 cases) | `backend.create_app` API surface | **High (gated)** | Dataset-gated, skips cleanly without `transcripts.lance/chunks.lance`. Standout assertions are real contracts: `alignments_json`→list (`:52-53`), unknown mode→**400** (`:62`), semantic-without-vLLM→**503 not 500** (`:67`). The 503/400 distinction *is* the product contract. Keep. |

**What this leaves uncovered:** the entire `model` + `ingest` write path, the new
`SearchSpec` Pydantic clamps (reachable today *only* through the gated smoke
test), `l2_normalize`, `_jpeg_dimensions`, and the FTS word-highlight path
(`iter_matching_words` / `parse_alignments_json`).

---

## 4. Ranked gap plan — highest value first

All cases below were execution-verified offline. Kind legend: **integration**
(real tmp-dir Lance, no GPU/net), **unit** (pure, no I/O), **dataset-gated**
(needs the local corpus).

| # | Target (`file::function`) | Value | Kind | Concrete cases |
|---|---|---|---|---|
| 1 | `ingest/ingest.py::ingest_many` (tmp-dir round-trip) | **Highest** | integration | (a) `ingest_many([doc], tmpdir)` → chunk row count == total chunks (verified 2); (b) FTS query for a known token returns the chunk (`search("klimat", query_type="fts")` → 1 hit, verified); (c) `alignments_json` round-trips → `list[dict]` with words (verified); (d) **second `ingest_many` into the same DB — pins the LIVE `json`-vs-`large_binary` append RuntimeError** (verified); (e) empty input → `ValueError("No chunks produced…")` (`:354`); (f) with `audio_root` at a tmp media file → `documents.lance` written + `media_blob` external URI; without any of `audio_root`/`media_base_uri`/`thumbnail_dir` → no documents table (`:405`). |
| 2 | `backend/app.py::SearchSpec` (clamp + `build`→400) | **High** | unit (no dataset) | `n`: 0→1, 50→50, 99999→**100** (verified); `fuzziness`: -2→0, 9→**2** (verified); `weight`: None→None, -1.0→0.0, 5.0→**1.0** (verified); `mode="bogus"`→`HTTPException(400)` (verified); `extra` kwarg dropped, no raise (verified); empty `build()`→`n=20, mode="fts"` (verified). **Belongs in `test_units.py`** — this is the headline of the Pydantic refactor and has no dataset-free home today. |
| 3 | `ingest/ingest.py::flatten_chunks` | **High** | unit | per-chunk `language` overrides else falls back to `doc_language` (verified `['sv','en']`); `speech_id` None→0 and `"3"`→int 3 (verified); `chunk_id` is per-speech `enumerate` (`[0,1]`); `text=None`→`''` (so the non-nullable `text` column never gets null); `speeches=None`/`chunks=None`→no rows, no raise; metadata match populates `referenskod/namn/bildid/extraid`. **Pin:** non-numeric `speech_id` (e.g. `"abc"`) raises `ValueError` via the `int()` cast (`:147`) — pin current behavior or flag as a latent gap. |
| 4 | `ingest/ingest.py::_pick_alignments` | **High** | unit | Closed-interval containment `a.start>=start and a.end<=end`: exact-bounds alignment KEPT, one-epsilon-outside DROPPED, alignment for a later chunk lands on that chunk; words copied with float-cast start/end. **Correction to audit input:** drop the proposed "start/end == None is skipped" case — `AlignmentSegment.start/end` are *required* floats (`model/datamodel.py:43-44`), so Pydantic rejects None *before* `_pick_alignments` runs; the None-guard at `ingest.py:98` is effectively dead for parsed input. The boundary cases remain the valuable part. |
| 5 | `media/frames.py::_jpeg_dimensions` | **High** | unit | baseline SOF0 with preceding APP0 → correct `(w,h)` (verified `(16,32)` — proves seg-len skip math + the deliberate `(h,w)→return(w,h)` swap); progressive SOF2 (`0xC2`) recognized; garbage / non-`0xFF` lead → `(0,0)` (verified); truncated / empty → `(0,0)` (verified); restart markers `0xD0-0xD7` before SOF skipped without consuming a length. The whole fn is wrapped in `except→(0,0)`, so *any* regression (offset, endianness, marker skip, w/h swap) fails **silently** into `(0,0)` metadata. |
| 6 | `vllm/image.py::l2_normalize` | **High** | unit (numpy) | 1D 2048-vec → shape `(1,2048)`, L2 norm ≈ 1.0; zero vector → all-zeros, `isfinite` all True (zero-norm guard, no NaN); wrong-width input (≠`EMBED_DIM`) raises `ValueError`; output dtype `float32`; 2D `(N,2048)` each row unit-norm. Applied to **every** stored and query embedding — a silent regression here corrupts cosine search globally. |
| 7 | `retrieval/search.py::iter_matching_words` + `parse_alignments_json` | **High** | unit | `parse_alignments_json`: None/`''`/0→`[]`; valid JSON-string→decoded list; already-a-list→identity passthrough (Lance JSONB may pre-decode); malformed string→`[]` no raise (verified). `iter_matching_words`: `terms=[]`→`[]`; leading/trailing `\W` stripped (`'Klimat,'` matches `'klimat'`); case-fold (`'MILJÖ.'` matches `'miljö'`, verified); null column→`[]`; word with no `"text"` key tolerated. Drives the transcript word-highlight feature on both the FTS path and `_postprocess_hits`. |
| 8 | `ingest/audio.py::compose_media_uri` | **High** | unit | base_uri without trailing slash → exactly one `/` before basename (verified `hf://buckets/v` and `hf://buckets/v/` both → `…/T1.mp4`); base_uri precedence over source; no base + source → absolute `file://`; no base + no source → None; only the `audio_path` basename used as the key. URI normalization is the exact silent-regression class — a double/dropped slash breaks media fetch with no exception. |
| 9 | `model/datamodel.py::AudioMetadata.model_validate_json` | **Medium** | unit (pydantic) | unmodeled extra key dropped (`extra="ignore"`, verified `UNKNOWN_EXTRA`); omitting `alignments`/`chunks` → `[]` via `default_factory` not None (flatten relies on this); missing required `audio_path`/`sample_rate`/`duration` → `ValidationError`; `speech_id` accepts str and int. Downgraded from "high": the round-trip (#1) already drives this on realistic input; the focused unit mainly guards `extra="ignore"` + `default_factory` against a careless refactor. |
| 10 | `ingest/ingest.py::load_metadata_csv` + `_metadata_for` | **Medium** | unit (tmp CSV) | semicolon CSV → `{bildid: {…4 cols…}}`, whitespace stripped, blank-bildid row skipped; `_metadata_for` keys on `Path(audio_path).stem` not basename-with-ext (verified `'T1'` miss); miss → all-None over exactly `METADATA_COLUMNS`; empty CSV value → None (`'or None'`). |
| 11 | `ingest/audio.py::resolve_source` + `guess_mime` | **Medium** | unit (tmp fs) | `resolve_source`: relative joined under `audio_root`; absolute verbatim; **nonexistent → None + warning, no raise** (the branch that silently yields a NULL `media_blob`). `guess_mime`: `.mp4`→`video/mp4`, unknown ext→`application/octet-stream` never None (verified). |
| 12 | `ingest/ingest.py::_doc_id` | **Medium** | unit | same path → same 16-char lowercase-hex id (golden pin); different paths → different ids; `_document_row` and `flatten_chunks` produce the **same** doc_id for the same path (join-key consistency). One line, but it's the stable join key between chunks and documents — a hashing/truncation regression silently orphans every document row. |
| 13 | `vllm/image.py::_square_crop` + `image_to_data_url` | **Medium** | unit (Pillow CPU) | `_square_crop`: non-square 640×480 → output `(_IMAGE_SIDE, _IMAGE_SIDE)`; **assert against the `_IMAGE_SIDE` constant** (`image.py:31`) so the documented 448→392 vision-token fix flips one expectation. `image_to_data_url`: PIL and raw bytes both → `data:image/jpeg;base64,` decodable JPEG; RGBA → `convert('RGB')` no raise; int/str → `TypeError`. |
| 14 | `media/download.py::read_manifest` | **Medium** | unit (tmp CSV) | semicolon CSV → expected row dicts; blank/whitespace `bildid` rows dropped (verified 3→1); every field `.strip()`'d; a comma-delimited file does NOT accidentally produce valid keyed rows (guards a silent delimiter regression). Overlaps `load_metadata_csv` — one focused test, not a suite. |

**Top priority is unambiguous:** #1 (round-trip) and #2 (`SearchSpec`). #1
lands red on a live bug; #2 is the new validation layer with no dataset-free
home today. Together they fill the largest holes (write path + Pydantic) for
the least code.

---

## 5. Explicitly NOT worth testing (the rejects)

| Target | One-line reason |
|---|---|
| `model/schema.py::CHUNK_SCHEMA`/`DOC_SCHEMA`/`*_STORAGE_VERSION` | Declarative PyArrow fields + `Final` constants — change-detector; the round-trip validates conformance for free. |
| `ingest/ingest.py::ingest_document` / `reindex_fts` | Pure arg-forwarding over lancedb; covered transitively. |
| `ingest/ingest.py::_build_chunks_table` / `_write_documents_table` | Column-marshalling internals; standalone tests need the same Lance fixtures with more mocking. The round-trip already proves the `json_()`/`blob_array` promotion (and surfaced the append bug). |
| `asr/detect_language.py::detect_and_sort` / `_mms_probe` / `_whisper_probe` / `ISO_639_3_TO_1` | Offset-voting is inlined inside a 115-line fs+GPU+ffmpeg function; probes are GPU boundaries; the dict is a constant. *Recommendation:* extract a pure `_tally_votes(list[tuple[str,float]])` helper and it becomes a clean high unit. |
| `media/frames.py::extract_chunk_frame` / `_extract_one` / `extract_chunk_frames_parallel` | ffmpeg argv + `subprocess.run` + `ThreadPoolExecutor`; an opt-in ffmpeg integration test on one committed clip beats any mock. The byte-parsing it delegates to (`_jpeg_dimensions`) is tested separately. |
| `media/thumbnails.py::generate_thumbnails` / `_extract_video_frame` / `_render_waveform` | ffmpeg argv builders + rglob; the ext-routing is mildly branchy but inseparable from real ffmpeg. |
| `media/download.py::_download_one` / `_run` / `download_manifest` | httpx-streaming + asyncio orchestration + CLI plumbing; needs a mock transport or live server. Only `read_manifest` is kept. |
| `asr/transcribe.py::run_transcribe` | Documented thin wrapper around `easytranscriber.pipelines.pipeline`; two `dict.get` lookups + ~15 kwarg forwards over a torch/pyannote/Whisper boundary. |
| `VLLMEmbeddingClient` / `VLLMReranker` / `QwenVLReranker` (`vllm/embedding.py`, `vllm/reranker.py`) | HTTP/threadpool plumbing against vLLM; the testable pure helpers (`l2_normalize`, `image_to_data_url`) are extracted to `vllm/image.py`. |
| `backend/app.py::_run_search` / `_vector_search` | Mode routing needs a live Lance table + the vLLM client — the whole world. Smoke covers FTS + 503/400. *Only worthwhile add:* one more smoke assertion — `mode="hybrid"` without text → 400. |
| `_pick_alignments` None-guard case | Unreachable: `AlignmentSegment.start/end` are required floats, Pydantic rejects None first. Drop that one case; keep the boundary cases. |

---

## 6. How to run + where tests live

Tests live in `tests/` (`pyproject.toml [tool.pytest.ini_options]`:
`testpaths = ["tests"]`, `addopts = "-ra -q"`).

```bash
# Pure unit tests — run everywhere, no dataset, no GPU, no network:
uv run --with pytest pytest tests/test_units.py -v

# Dataset-gated backend smoke (needs httpx; skips cleanly without the corpus):
uv run --with pytest --with httpx pytest tests/test_backend_smoke.py -v

# Everything:
uv run --with pytest --with httpx pytest
```

**Dataset-gated skip pattern** (from `test_backend_smoke.py:20`) — reuse this for
any test that needs the local corpus so CI without the dataset stays green:

```python
DB_PATH = Path(__file__).resolve().parent.parent / "transcripts.lance"

pytestmark = pytest.mark.skipif(
    not (DB_PATH / "chunks.lance").exists(),
    reason="local transcripts.lance dataset not present",
)
```

**Where the new tests go:**
- Gaps #2–#14 (pure units) → `test_units.py` (or split a new `test_ingest_units.py`
  if it grows). These run in CI without the dataset — they need **no** skip guard.
- Gap #1 (round-trip) → a new `test_ingest_roundtrip.py` using `tmp_path`. It
  needs **no** dataset gate and **no** GPU; it builds its own Lance DB in a tmp
  dir in ~0.03 s. There is no `conftest.py` and no committed sample transcriber
  JSON today — a small inline JSON dict (as used in the verification above) is
  the obvious enabler.
