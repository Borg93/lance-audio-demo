# ratch is model-free — the shipped architecture

**State (2026-07-23, DONE):** `ratch` = pure Ray Data orchestration over Lance,
**zero model deps** — `import ratch` loads no torch/easytranscriber/pyannote/
toponymy/lightrag and not even `ray` (the driver imports it lazily at run time);
pinned by `tests/test_core_contract.py`. Every model lives in `runners/<name>/`
with its own env. There are NO endpoint clients and NO subprocess calls in the
compute paths — the old sealed-subprocess `ratch/endpoints/` stand-in is deleted.

## The two ways ratch drives a runner

**1. Per-item models → Ray Data actors (stages).** A `Stage` declares
`runner="<name>"`; the composition root resolves
`runners.<name>.actor.compute_factory` + `OUTPUT_SCHEMA` by convention
(`ratch/core/runners.py::resolve_runner_actor`) and hands the factory to the
driver's `map_batches` — one warm model per actor. **A new model is one runner
dir + one Stage entry; the driver and `ray_av` never change** (proof:
`tests/test_runner_convention.py` drives the real append path with a fake
runner). The stage-side facts travel as a `RunnerContext` (paths only — the
runner owns its model config). Runner-backed today: `diarize`, `voiceprint`
(and `asr` holds the transcribe/detect-language compute the CLI imports).

**2. Corpus-global / one-shot models → Ray Jobs.** `ratch/core/jobs.py`
(`run_runner`) mirrors lance-ns `medallion/services/ray_submit.submit_stage_job`:
deterministic uuid5 submission id per (runner, token) — a resubmit RE-ATTACHES
to a running job instead of racing it; a terminal prior job is deleted and
resubmitted (deviation from lance-ns, documented in the module); the runner's
env + forwarded `MEDIA_*`/`AWS_*` vars ride in the job's `runtime_env`;
`metadata={"kind": <runner>}`. `RATCH_RAY_ENABLED=1` submits to the cluster
(`ray.job_submission.JobSubmissionClient`); off (default) runs the worker
in-process (`runners/<name>/worker.py::main`, same argv contract). Job-driven
today: `topics` (`ratch feature topics`); `kg` (its scripts; gains a worker.py
when it becomes job-submittable). Local sealed-env convenience is Make targets
(`make topics` = `uv run --project runners/topics python -m runners.topics.worker`),
never Python `subprocess`.

**The sorting rule is honest, not aesthetic:** `map_batches` streams disjoint
batches through parallel actors, so a whole-corpus fit (Toponymy clusters the
entire atlas map; LightRAG builds one graph) structurally cannot be a stage.
`runners/topics/actor.py` raises its own explanation at resolution time; kg
ships no actor module and `resolve_runner_actor` points at the jobs seam.

## Layout

```
runners/<name>/
  pyproject.toml     the runner's OWN env (conflicting/heavy deps live here)
  actor.py           per-item runners: compute_factory(ctx) + OUTPUT_SCHEMA
  worker.py          job runners: run() + main() (argv contract, Ray Job entrypoint)
  deployment.py      online form (Ray Serve), where applicable
src/ratch/core/runners.py   RunnerContext, resolve_runner_actor, runner_env
src/ratch/core/jobs.py      RunnerJob, JobsSettings, run_runner (the jobs seam)
```

Pure compute stays in ratch: ffmpeg frames/thumbnails/WAV transcode
(`modalities/av/`, incl. `wav.py::extract_wav_16k_mono` shared by the speech
runners), ingest IO, the stage registry, retrieval.

## Envs: dev bridge vs production

- **Local single-node (dev):** actors share the driver env; the `[models]` extra
  supplies asr/diarize/voiceprint deps. No per-run pip of torch.
- **Cluster (dev bridge):** `RATCH_RUNNER_ISOLATION=1` attaches each runner's
  pip `runtime_env` (built from its pyproject) per stage; jobs always carry it.
- **Production (merge-time):** per-runner container images on KubeRay worker
  groups — pip runtime_env is dev-only per Ray docs (torch + cu128 index is
  specifically painful). Tracked in TODO.

## Merge-time follow-ups (needs the live cluster)

- Per-runner images replace pip runtime_envs; retire the `[models]` extra.
- `runners/{embed,rerank,caption,summarize}/` — the vLLM set joins the shape.
- The viewer's voice-upload encoder (`services/viewer/services/wespeaker.py`) is
  the LAST in-process model — becomes a runners/ Serve deployment.
- asr-as-deriver writes the transcript column directly (folds away the
  easytranscriber-JSON ingest hop).
