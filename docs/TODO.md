# TODO — what's left

Single, forward-looking backlog for `raudio`. **Done work is not listed here** —
it lives in the shipped code + git history and in [REPRODUCE.md](REPRODUCE.md) /
[PIPELINE.md](PIPELINE.md) / [GUIDE.md](GUIDE.md) / [STORAGE.md](STORAGE.md).
This file replaces the old root `TODO.md` (a closed-item changelog) and `todo2.md`
(the curation roadmap) — both consolidated here; their full detail is in git history.

> Status: ✅ done · ⏳ in progress · 📋 planned · 🟡 parked/optional.
> The product is shipped — there are **no active blockers**. Everything below is
> deliberate backlog, ordered roughly by value-per-effort.

---

## In flight

### ⏳ Diarization — full-corpus coverage
- The `make speaker-turns` batch is backfilling `speaker_turns.lance` (resumable,
  ~2–4 min/video on a shared GPU → ~1–2 days for all 1,154). Shipped + live for
  done videos (Speakers tab).
- 📋 **3-GPU sharding** to finish in ~8–16 h: split videos across GPUs 0/1/2,
  each writing `speaker_turns_gpu{n}.lance`, merged at the end (no concurrent-write race).
- 📋 **On-demand diarization** — diarize a video the first time it's opened + cache,
  instead of (or alongside) the full batch. `POST /api/diarization/{doc}` running pyannote.
- 🟡 **"Diarized only" filter/badge** in the hit list (`GET /api/diarization` list route + a toggle).
- 🟡 Declare `pyannote` as a first-class dep (currently transitive via easytranscriber).

---

## Curation / exploration roadmap (FiftyOne-inspired)

The high-value/low-effort pair is **#1 + #2** — they fix the "one press conference
floods the page" redundancy (adjacent 30 s clips are near-identical).

1. 📋 **Group-by-video** (S, high) — collapse the result list by `doc_id` (frontend
   reshape; optional backend `per_doc_cap`). Hits already carry `doc_id`+`namn`.
2. 📋 **Uniqueness / near-dup collapse** (M, high) — `raudio feature uniqueness` over an
   embedding column; retrieval-level dedup of adjacent chunks.
3. 📋 **More-like-this** (M, high) — similarity sort from a hit (reuses `_vector_search`/
   stored embeddings; zero new data/GPU).
4. 📋 **Stats / histograms** (M, med-high) — faceted, count-annotated filter panel
   (`chunks_ds` scan + `/api/columns` + LayerChart).
5. 📋 **Tags + saved views** (M, med) — first curation loop; introduces mutable state
   (new SQLite store).

---

## Bigger bets

- 🟡 **Voice / speaker SEARCH (ECAPA `speaker_embedding`)** — *distinct from diarization,
  which shipped.* De-risk verdict: cross-video AUC **~0.74 (AMBER)**, channel-inflated +
  label-noisy. To revisit: diarization-clean labels + **AS-norm/PLDA**, and try the **1.7B**
  encoder (2048-d). Then a `voice` search mode + atlas `--space voice`.
- 📋 **Video-level text + summary** — `documents.full_text` (concat chunk text per `doc_id`)
  + `documents.doc_summary` (map-reduce LLM). Enables full-video FTS + summaries.
- 📋 **Studio desktop merge** — fold ranymizer + raudio + multimodal-webgpu-demo into a
  Tauri "Studio" shell (full plan: [STUDIO_MERGE.md](STUDIO_MERGE.md)).

---

## Parked — YAGNI at this scale (145k rows, single local node)

Revisit only if a profiler or real concurrency makes them bite.

- **Search perf:** query-vector LRU cache; run `hybrid`/`all` legs concurrently
  (`asyncio.gather`); try `IVF_HNSW_SQ` for `frame_embedding`.
- **vLLM perf:** async per-query embed client; confirm `--enable-prefix-caching`;
  `/metrics` bottleneck check; FP8 / `--async-scheduling` (stretch).
- **Housekeeping:** prune old dataset versions (disk); `make compact` after multi-stage writes.

## Code-quality (deferred)

- `DomainError` hierarchy + exception handlers (vs inline `HTTPException`).
- CORS `allow_origins` → settings-driven (currently `*`, fine behind the local proxy).
- `_Ctx` global state → Typer `ctx.obj`.
- `print()` → `logging` in library modules (`media/thumbnails.py`, `media/download.py`, `asr/detect_language.py`).
- Minor typing/dedup (untyped `frames._extract_one` args; reranker prefix/suffix ↔ jinja cross-ref).
