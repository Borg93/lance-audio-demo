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

### ⏳ runners/ = every model's home; ratch drives them (USER-DECIDED architecture)
*Full plan: [RATCH_MODEL_FREE.md](RATCH_MODEL_FREE.md). Pick up here.*

**The decided model (2026-07-21):** `runners/<name>/` owns EVERY conflicting-env /
heavy model — **offline AND online**. A runner is the model's home: its env
(`pyproject.toml`) + its **actor code**. How it's driven is orthogonal:

- **offline batch** — ratch's Ray Data stages IMPORT the runner's actor class and
  run it via `map_batches(RunnerActor, runtime_env=<that runner's env>)`. Ray's
  per-stage `runtime_env` resolves the conflicting deps ON THE WORKERS; ratch's
  driver env never carries a model. (This replaces the pre-merge sealed-subprocess
  path in `ratch/endpoints/` — subprocess-per-batch must never sit in an actor loop.)
- **online** — the same runner exposes `deployment.py` (Ray Serve) for query-time /
  interactive use (embed, rerank, assist).

**The clarity rule:** ratch's `Stage` declares its runner EXPLICITLY (a `runner=`
binding generalizing today's `client=`), so reading `features/stages.py` tells you
exactly which stages are runner-backed and which are pure compute. ratch = pure
orchestration; runners = the models.

**Done (EXECUTED 2026-07-21):** ratch core model-free (`import ratch` loads no
model; model stack = `[models]` extra for local single-node runs). ALL five models
live in `runners/`: topics, kg, **asr** (transcribe + detect_language), **diarize**,
**voiceprint** — each with its own `pyproject.toml` env; diarize + voiceprint have
`actor.py` (the Ray Data actor factories ratch imports). `Stage.runner=` declared
on the runner-backed stages (`features/stages.py` is the legible map);
`ratch/core/runners.py` builds the per-stage `runtime_env` from the runner's
pyproject and the driver attaches it in `map_batches` when
`RATCH_RUNNER_ISOLATION=1` (cluster mode; local single-node shares the driver env
— no per-run pip of torch). `ray_av.py` is pure composition (only the model-free
ffmpeg frames factory remains). `src/ratch/modalities/av/` holds ONLY pure compute
(frames/thumbnails/download/cluster). ty fully clean.

**📋 Left (merge-time — needs the live Ray cluster to verify):**
- **Per-runner container images, not pip runtime_envs, in production** (Ray docs:
  pip runtime_env = dev/experimentation; bake deps into images for prod; torch
  stacks are specifically painful in pip runtime_envs — cu128 extra index, build-
  order — and conflicting runtime_envs between communicating actors can hit
  unpickling errors). Each `runners/<name>/pyproject.toml` → a
  `.docker/<name>.dockerfile` image on KubeRay worker groups (or the runner as a
  Serve deployment with its own image). `RATCH_RUNNER_ISOLATION=1` (pip
  runtime_env) stays as the DEV bridge only.
- `runners/{embed,rerank,caption,summarize}/` — the vLLM set joins the same shape
  (offline actor + online Serve deployment.py; one model, two drivers).
- Retire the `[models]` extra + the `endpoints/` sealed-subprocess stand-in once
  runners run isolated on the cluster; `topics`/`kg` gain `actor.py` when they
  become in-pipeline stages (today they're one-shot workers).
- The viewer's voice-upload encoder (`services/viewer/services/wespeaker.py`) is
  the LAST in-process model (online, lazy imports, `--extra models` pre-merge) —
  becomes a runners/ Serve deployment the annotator/viewer call at merge.

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
