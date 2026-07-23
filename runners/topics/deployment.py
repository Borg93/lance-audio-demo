"""Ray Serve deployment for the topics model service (the merge-time online form).

This is the ONLINE service ratch calls. Pre-merge, ratch runs the same
:func:`worker.main` compute in this dir's sealed env via
``ratch.endpoints.topics.LocalTopicsClient`` (no server needed). At merge, this
``@serve.deployment`` is what runs — ratch's ``RemoteTopicsClient`` POSTs to it,
zero change to the calling stage.

Runs ONLY in this service's env (``ray[serve]`` + toponymy, the ``serve`` extra),
never in ratch's — so it is excluded from ratch's type-check/lint.
"""

from __future__ import annotations

from typing import Any

import worker
from ray import serve  # type: ignore[import-not-found]


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0})
class TopicsDeployment:
    """Build Swedish topic layers on a Lance DB's chunks (Toponymy).

    The request carries the DB path (S3/local URI) + the optional LLM endpoint;
    the deployment runs the same compute as the sealed CLI worker and returns the
    row count. Heavy imports (toponymy) stay lazy inside :func:`worker.main` so
    replica start-up is cheap.
    """

    async def __call__(self, request: Any) -> dict[str, int]:
        body = await request.json()
        rows = worker.run(db_path=body["db"], llm_url=body.get("llm_url"))
        return {"rows": rows}


app = TopicsDeployment.bind()
