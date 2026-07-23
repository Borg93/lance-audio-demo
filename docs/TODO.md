# TODO — what's left

Single, forward-looking backlog for `ratch`. **Done work is not listed here** —
it lives in the shipped code + git history and in [REPRODUCE.md](REPRODUCE.md) /
[PIPELINE.md](PIPELINE.md) / [GUIDE.md](GUIDE.md) / [STORAGE.md](STORAGE.md).
This file replaces the old root `TODO.md` (a closed-item changelog) and `todo2.md`
(the curation roadmap) — both consolidated here; their full detail is in git history.

> Status: ⏳ in progress · 📋 planned · 🟡 parked/optional.
> The product is shipped and **merge-ready** — there are no active blockers and
> nothing in flight. Everything below is deliberate backlog, ordered roughly by
> value-per-effort.

---

## Merge-time (deferred to the lance-ns merge by design — needs the live cluster)

The runners/ architecture is complete here ([RATCH_MODEL_FREE.md](RATCH_MODEL_FREE.md));
these are its cluster-side halves. The merge itself is driven by
[LANCE_NS_HANDOFF.md](LANCE_NS_HANDOFF.md) (the 8 questions + first integration
milestone) with [RASK_COMPARE.md](RASK_COMPARE.md) covering the rask side.

- 📋 **Per-runner container images, not pip runtime_envs, in production** (Ray docs:
  pip runtime_env = dev/experimentation; torch cu128 stacks are specifically painful).
  Each `runners/<name>/pyproject.toml` → a `.docker/<name>.dockerfile` image on KubeRay
  worker groups. `RATCH_RUNNER_ISOLATION=1` (pip runtime_env, cu128 index read from the
  runner's `[[tool.uv.index]]`) stays as the DEV bridge only.
- 📋 `runners/{embed,rerank,caption,summarize}/` — the vLLM set joins the runner shape
  (offline actor + online Serve `deployment.py`), replacing the `client=` HTTP stages.
- 📋 Retire the `[models]` extra once runners run isolated on the cluster; `kg` gains a
  `worker.py` argv entrypoint when its scripts fold into one job (topics/kg stay
  job-only permanently — corpus-global fits can't be `map_batches` stages).
- 📋 The viewer's voice-upload encoder (`services/viewer/services/wespeaker.py`) is the
  LAST in-process model — becomes a runners/ Serve deployment the viewer/annotator call.
- 📋 asr-as-deriver writes the transcript column directly (folds away the
  easytranscriber-JSON ingest hop).

---

## Diarization — remaining surface

- 📋 **On-demand diarization** — diarize a video the first time it's opened + cache,
  instead of (or alongside) the batch. `POST /api/diarization/{doc}` running pyannote.
- 🟡 **"Diarized only" filter/badge** in the hit list (`GET /api/diarization` list route
  + a toggle).

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

## Bigger bets

- 📋 **Voice / speaker search — remaining surface** (the search itself shipped, see
  [VOICE.md](VOICE.md)):
  - 📋 **Speaker naming** — `speakers.speaker_name` is still all-NULL; a write route +
    UI to name an identity cluster, then show names on hits/timeline.
  - 📋 **Speakers browse page** — list/browse the identity clusters
    (`GET /api/voice/identity` exists; the frontend doesn't consume
    `speaker_cluster` beyond the hit field yet).
  - 🟡 **Atlas `--space voice`** — EVōC projection over `speakers.embedding`.
- 📋 **Video-level text + summary** — `documents.full_text` (concat chunk text per `doc_id`)
  + `documents.doc_summary` (map-reduce LLM). Enables full-video FTS + summaries.
- 🟡 **Studio desktop shell** — parked: a Tauri desktop merge was once planned; see git
  history if it revives.

---

## Parked — YAGNI at this scale (145k rows, single local node)

Revisit only if a profiler or real concurrency makes them bite.

- **Search perf:** query-vector LRU cache; run `hybrid`/`all` legs concurrently
  (`asyncio.gather`); try `IVF_HNSW_SQ` for `frame_embedding`.
- **vLLM perf:** async per-query embed client; confirm `--enable-prefix-caching`;
  `/metrics` bottleneck check; FP8 / `--async-scheduling` (stretch).
- **Housekeeping:** prune old dataset versions (disk).
