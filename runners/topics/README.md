# `topics` model service — the reference template for `services/models/*`

An **online model service**: ratch's offline compute calls it; it is not part of
ratch. It exists because Toponymy transitively needs **transformers < 5** while
ratch resolves **transformers 5.x** — the conflict is resolved by giving this
service its **own env**, never by leaking deps into ratch.

## What lives here (the pattern every `services/models/<name>/` follows)

| file | role |
|---|---|
| `pyproject.toml` | **this service's env** — the conflicting deps live here and nowhere else |
| `worker.py` | the compute (`run()` + a CLI `main()`) — the model work |
| `deployment.py` | the **Ray Serve** `@serve.deployment` (merge-time online form) |
| `README.md` | this |

ratch reaches it through **`ratch/endpoints/topics.py`** — a `TopicsClient`
Protocol with two impls behind one factory:

- `LocalTopicsClient` (pre-merge default): `uv run --project services/models/topics
  topics-worker --db <db>` — runs `worker.run` in *this* sealed env.
- `RemoteTopicsClient` (merge): POST to the Ray Serve deployment at
  `MEDIA_TOPICS_URL`.

`get_topics_client()` picks remote when `MEDIA_TOPICS_URL` is set, else local —
so the same `ratch feature topics` call works before and after the merge with no
stage change.

## Run it

```bash
# pre-merge, sealed env (what ratch does under the hood):
uv run --project services/models/topics topics-worker --db transcripts_v2.lance

# at merge, as a Ray Serve deployment (needs the `serve` extra):
#   serve run services/models/topics/deployment.py:app
# then point ratch at it:  export MEDIA_TOPICS_URL=http://topics:8000
```

## Why this shape

- **ratch stays model-free** — pure Ray Data compute; it holds only the thin
  endpoint client, imports no model library.
- **The dep conflict dissolves structurally** — one env per service.
- **It's the merge target, pre-built** — this dir is one Ray Serve deployment;
  at merge it drops into lance-ns's Ray cluster, and `.docker/topics.dockerfile`
  builds its image (RA convention).
