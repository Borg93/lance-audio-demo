# TODO — what's left

Single, forward-looking backlog for `ratch`. **Done work is not listed here** —
it lives in the shipped code + git history and in [REPRODUCE.md](REPRODUCE.md) /
[PIPELINE.md](PIPELINE.md) / [GUIDE.md](GUIDE.md) / [STORAGE.md](STORAGE.md).
This file replaces the old root `TODO.md` (a closed-item changelog) and `todo2.md`
(the curation roadmap) — both consolidated here; their full detail is in git history.

> Status: ✅ done · ⏳ in progress · 📋 planned · 🟡 parked/optional.
> The product is shipped — there are **no active blockers**. Everything below is
> deliberate backlog, ordered roughly by value-per-effort.

---

## In flight

### ⏳ ratch model-free — finish the runners/ extraction (merge-time)
*Full plan + rationale: [RATCH_MODEL_FREE.md](RATCH_MODEL_FREE.md). Pick up here.*

**Done (this session, on main):** ratch's CORE is model-free — `easytranscriber`/
`torch`/`torchaudio` are a `[models]` optional extra, the one top-level model import
(`detect_language`) is lazy; `import ratch` + the Ray Data path load no model. Model
services live in top-level **`runners/<name>/`** (own env + `deployment.py` Ray Serve
+ README), called via **`ratch/endpoints/<name>.py`** (Protocol + Local[sealed] +
Remote[Serve] + factory). **`runners/topics`** and **`runners/kg`** are extracted as
the template. `runners/` sits beside `services/` (NOT under it) so it can't clash with
lance-ns `services/{catalog,lineage,medallion,compaction}` at merge.

**📋 Left (merge-time — needs the Ray Serve runtime to build + verify):**
- Extract **asr / diarize / voiceprint** → `runners/{asr,diarize,voiceprint}/` as Ray
  Serve deployments, replicating `runners/topics` (pyproject env + `deployment.py` +
  `ratch/endpoints/<name>.py` client). They run **per-batch inside Ray Data actors**,
  so their model-free form is the Serve-handle call — pre-merge they run in-process via
  `--extra models` (that's why they're deferred; extracting blind would put subprocess-
  per-batch in a hot actor loop, and there's no GPU/Serve here to verify).
- Then remove `[models]` from ratch entirely + `grep` no torch/easytranscriber/pyannote/
  wespeaker under `src/ratch/`.
- The vLLM model set (embed/rerank/caption/summarize) also becomes `runners/*` at merge
  (endpoint clients already env-URL-agnostic in `ratch/clients/`).

### ⏳ Diarization — full-corpus coverage
- The `make speaker-turns` batch is backfilling `speaker_turns.lance` (resumable,
  ~2–4 min/video on a shared GPU → ~1–2 days for all ~1,576). Shipped + live for
  done videos (Speakers tab).
- ✅ **Sharded diarization is shipped** — `ratch extract-speaker-turns --num-shards N
  --shard-index i` writes each disjoint slice to `speaker_turns_shard{i}.lance`, folded
  back with `ratch merge-speaker-turns` (no concurrent-write race). NOTE: the
  `make speaker-turns` target itself does not pass shard flags; sharding is a manual
  N-process launch (one per GPU).
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
2. 📋 **Uniqueness / near-dup collapse** (M, high) — `ratch feature uniqueness` over an
   embedding column; retrieval-level dedup of adjacent chunks.
3. 📋 **More-like-this** (M, high) — similarity sort from a hit (reuses `_vector_search`/
   stored embeddings; zero new data/GPU).
4. 📋 **Stats / histograms** (M, med-high) — faceted, count-annotated filter panel
   (`chunks_ds` scan + `/api/columns` + LayerChart).
5. 📋 **Tags + saved views** (M, med) — first curation loop; introduces mutable state
   (new SQLite store).

---

## Bigger bets

- 📋 **Voice / speaker search — remaining surface.** The search itself **shipped**
  (2026-06-10, see [VOICE.md](VOICE.md)): pyannote WeSpeaker 256-d turn voiceprints
  (590k turns / 9,941 speakers), `/api/voice/{similar,status,identity}`, hit-card /
  timeline / upload UX, and seeded-EVoC identity clusters. The old plan on this line
  (ECAPA / AMBER verdict / AS-norm / 2048-d encoder) is obsolete — diarization-clean
  turn labels + the WeSpeaker encoder resolved it (AUC 1.000 on the human labels).
  What genuinely remains:
  - 📋 **Speaker naming** — `speakers.speaker_name` is still all-NULL; a write route +
    UI to name an identity cluster, then show names on hits/timeline.
  - 📋 **Speakers browse page** — list/browse the identity clusters
    (`GET /api/voice/identity` exists; the frontend doesn't consume
    `speaker_cluster` beyond the hit field yet).
  - 🟡 **Atlas `--space voice`** — EVōC projection over `speakers.embedding`.
  - (The frame/caption embedding-space redo is a *separate* track — unrelated to voice.)
- 📋 **Video-level text + summary** — `documents.full_text` (concat chunk text per `doc_id`)
  + `documents.doc_summary` (map-reduce LLM). Enables full-video FTS + summaries.
- 📋 **Studio desktop merge** — fold ranymizer + ratch + multimodal-webgpu-demo into a
  Tauri "Studio" shell (full plan: [STUDIO_MERGE.md](STUDIO_MERGE.md)).

---

## Parked — YAGNI at this scale (145k rows, single local node)

Revisit only if a profiler or real concurrency makes them bite.

- **Search perf:** query-vector LRU cache; run `hybrid`/`all` legs concurrently
  (`asyncio.gather`); try `IVF_HNSW_SQ` for `frame_embedding`.
- **vLLM perf:** async per-query embed client; confirm `--enable-prefix-caching`;
  `/metrics` bottleneck check; FP8 / `--async-scheduling` (stretch).
- **Housekeeping:** prune old dataset versions (disk).

## Code-quality (deferred)

- `DomainError` hierarchy + exception handlers (vs inline `HTTPException`).
- CORS `allow_origins` → settings-driven (currently `*`, fine behind the local proxy).
- `_Ctx` global state → Typer `ctx.obj`.
- `print()` → `logging` in library modules (`media/thumbnails.py`, `media/download.py`, `asr/detect_language.py`).
- Minor typing/dedup (untyped `frames._extract_one` args; reranker prefix/suffix ↔ jinja cross-ref).
