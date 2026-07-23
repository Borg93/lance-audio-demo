# Making ratch model-free — the plan

**Goal:** `ratch` = pure Ray Data orchestration over Lance, **zero model deps**.
Every model (inference) lives in `services/models/<name>/` with its own env + Ray
Serve deployment; ratch calls it through a thin `ratch/endpoints/<name>.py` client.

**Why:** once the input is Lance (medallion bronze blob), every model step —
embed, caption, topics, **and asr/diarize/voiceprint** — is a *deriver stage*
(`read_lance → map_batches(stage-that-calls-a-Serve-handle) → write column`), not
a "preprocess the files first" step. ratch's process should never load a model;
it holds Serve handles and does read→map→write. That IS the lance-ns medallion
deriver shape.

## The rule (why some things are services and some aren't)

A model becomes a `services/models/*` service when **it runs inference** — full
stop, in the model-free target. (Pre-merge, the endpoint client keeps a local
sealed impl so it still runs without a Serve cluster.) Pure compute — ffmpeg
frames, thumbnails, download, clustering, the stage registry, ingest IO — stays
in ratch.

## Established pattern (done: `topics`, `kg`)

```
services/models/<name>/
  pyproject.toml     the service's OWN env (conflicting/heavy deps live here)
  worker.py          the compute (run() + CLI main())
  deployment.py      the Ray Serve @serve.deployment (merge-time online form)
  README.md
ratch/endpoints/<name>.py   Protocol + Local<Name>Client (sealed env) +
                            Remote<Name>Client (Ray Serve) + get_<name>_client()
```

## Status (2026-07-21)

**DONE — ratch's CORE is model-free (the headline):** `easytranscriber`/`torch`/
`torchaudio` moved from `[project.dependencies]` → the `[models]` optional extra;
the one top-level model import (detect_language) made lazy. Verified: `import ratch`
loads no torch/easytranscriber, `uv sync` (core) installs no model stack, 635 tests
pass with only the non-model `multimodal`/`atlas` extras. `topics` + `kg` extracted
to `services/models/*` (Ray Serve template). Model-running Make targets take
`--extra models`.

**The correct pre-merge state for the actor models:** `asr`/`diarize`/`voiceprint`
run inside Ray Data actors (per-batch). Their model-free form is the *merge-time*
Serve-handle call — pre-merge they run **in-process** in the actor via the `[models]`
extra. Physically relocating them to `services/models/*` only pays off once they are
Ray Serve deployments (needs the merge runtime to build + verify); doing it blind
here would put subprocess-per-batch in a hot actor loop. So they stay in
`ratch/modalities/` **behind `[models]`** until merge, when they become Serve
deployments (template: `topics`/`kg`) and the actors call handles.

## TODO (remaining = merge-time)

### Phase 1 — asr  (easytranscriber / transformers 5 / torch)
- [ ] `services/models/asr/` — pyproject (easytranscriber, torch, torchaudio),
      worker.py (from `modalities/av/asr/{transcribe,detect_language}.py`),
      deployment.py, README
- [ ] `ratch/endpoints/asr.py` — `AsrClient` (transcribe + detect-language) with
      Local (sealed) + Remote (Serve) impls + factory (`MEDIA_ASR_URL`)
- [ ] rewire `cli/transcribe.py` → the endpoint client; drop the model import
- [ ] `test_asr_detect.py` → repoint (the classifier logic can stay pure/importable)
- [ ] verify: `import ratch` needs no easytranscriber; tests green

### Phase 2 — diarize  (pyannote)
- [ ] `services/models/diarize/` (pyannote env) + worker + deployment + README
- [ ] `ratch/endpoints/diarize.py`
- [ ] rewire `features/ray_av.py` + `cli/speaker.py`
- [ ] verify

### Phase 3 — voiceprint  (wespeaker)
- [ ] `services/models/voiceprint/` (wespeaker env) + worker + deployment + README
- [ ] `ratch/endpoints/voiceprint.py`
- [ ] rewire `cli/speaker.py` + `features/ray_av.py`; decouple `model/schema.py`
      (it only needs the embedding DIM constant, not the model)
- [ ] verify

### Phase 4 — drop the model deps from ratch
- [ ] remove `easytranscriber`, `torch`, `torchaudio` from `[project] dependencies`
- [ ] ratch core deps = `ray[data]`, `lance-ray`, `pylance`, `pydantic`, `typer`,
      `fastapi`, `uvicorn`, `lance-graph`, `pydantic-settings`, `fastmcp`
- [ ] `uv sync` re-resolves; `import ratch` model-free; grep: no torch/transformers/
      easytranscriber/pyannote/wespeaker import anywhere under `src/ratch/`
- [ ] full test suite green

### Phase 5 — docs
- [ ] `services/models/README.md` — the pattern + the rule
- [ ] update the architecture doc: ratch = Ray Data over Lance; models = services;
      the deriver-stage-calls-Serve-handle mechanism
- [ ] mark this TODO done

## Caveats (honest)

- The model **envs can't be installed/run here** (GPU + large downloads). We verify
  **structure + env-lightening + the endpoint clients + `import ratch` model-free +
  the test suite** — not live GPU inference.
- Each phase keeps the tree **green** (`import ratch` + tests) before commit; if
  Phase 4's dep-drop can't re-resolve cleanly, we stop there and report rather than
  leave a broken env.
- `ingest/` stays: it parses easytranscriber's **JSON shape** (a data schema, not a
  model import) → bronze Lance. In the medallion target, asr-as-deriver writes the
  transcript column directly and this intermediate JSON step folds away — a merge
  follow-up, noted not done here.
