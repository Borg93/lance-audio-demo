# rask ↔ ratch/runners — what survives the merge, what gets altered

*2026-07-23 · rask (`/home/blackwell/Desktop/rask`) merges into lance-ns as the
compute side; lance-media merges as the catalog read/write layer. This maps
rask's Ray usage onto the runners/ architecture so the merge keeps rask's good
bones and drops its structural mistakes. Grounding: rask source read 2026-07-22;
Ray-idiom claims verified against the Ray docs (batch inference, dependency
handling) — see docs/TODO.md for the images-over-pip-runtime_env note.*

## What rask KEEPS (already idiomatic — adopt as-is)

| rask piece | Why it stays |
|---|---|
| `packages/htr/src/htr/actors/*` — callable-class actors driven by `map_batches(ActorCls, concurrency=…, num_gpus=…)` | The canonical Ray Data batch-inference shape: model loads once per actor (warm), batches stream through. This is exactly what `runners/{diarize,voiceprint}/actor.py::compute_factory` does — rask's actors become `runners/<model>/actor.py` bodies nearly verbatim. |
| `packages/ray-kit` — `JobSubmissionClient` wrapper + dashboard address plumbing | The submit pattern `ratch/core/jobs.py` adopts: `submit_job(entrypoint=…, submission_id=<deterministic>, runtime_env={working_dir, env_vars filtered}, metadata={"kind": …})` (rask `components/scripts/submit_index.py` has it right). At merge, lance-ns's `ray_submit.py` and ratch's `jobs.py` are the same seam — rask jobs plug into it unchanged. |
| Per-model GPU sizing on the stage call (`num_gpus`, actor pool bounds) | Same knobs as ratch's `ActorConfig` — carries over 1:1. |

## What rask ALTERS at merge (the three structural fixes)

| rask mistake | The runners/ fix |
|---|---|
| **Model deps live in the pipeline env** — `torch`/`transformers` imported at module top of the pipeline package, so the driver env carries every model and two models' pins collide (the exact conflict that forced topics/kg isolation here). | Each model moves to `runners/<name>/` with its OWN `pyproject.toml`; the pipeline stays model-free (`import ratch` loads no model — pinned by `tests/test_core_contract.py`). Stages bind by name (`Stage.runner`), actors resolve by convention. |
| **Hand-managed side venv** — `deploy_qwen_llm.py` bootstraps `~/qwen-serve/.venv-ray` by hand (mkdir + pip inside a script). | The runner's env IS the declaration: dev = pip `runtime_env` built from `runners/<name>/pyproject.toml` (`RATCH_RUNNER_ISOLATION=1`), prod = a container image built from the same file. No filesystem-side venvs. |
| **IO via `read_binary_files` over raw files** — rask streams file bytes through Ray object store and re-parses per run. | Input is Lance (`lance_ray.read_lance` with column projection + filters; blobs stay out of the object store via `take_blobs` actor-side). rask's derivers become Lance-in → Lance-out stages like every ratch stage. |

## Job-vs-stage sorting for rask's workloads

Same rule as here: **per-item models = actors** (HTR line recognition, layout —
`map_batches`), **corpus-global or one-shot builds = jobs** (index builds like
`submit_index.py` — the `ratch/core/jobs.py` path, deterministic id, worker
entrypoint). rask's `submit_index.py` was already a job; it keeps that shape and
gains only the runner-env `runtime_env` + the settings flag.
