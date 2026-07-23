"""Runner bindings — how ratch's driver reaches a runner's env, knowing no models.

A ``Stage.runner=`` names a ``runners/<name>/`` directory. On a Ray cluster the
driver attaches that runner's env to the stage's actors via a per-stage
``runtime_env`` (deps resolve on the WORKERS — the driver env stays model-free).
On a local single-node run the actors share the driver env, so the ``[models]``
extra supplies the deps and no runtime_env is attached (isolation is opt-in via
``RATCH_RUNNER_ISOLATION=1`` — pip-installing torch per local run would be waste).
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

#: Opt-in flag: attach each runner's env as a per-stage runtime_env (cluster mode).
_ISOLATION_ENV = "RATCH_RUNNER_ISOLATION"


def runners_root() -> Path:
    """The repo's ``runners/`` directory (three parents up from ``src/ratch/core``)."""
    return Path(__file__).resolve().parents[3] / "runners"


@lru_cache
def runner_env(name: str) -> dict[str, Any]:
    """A Ray ``runtime_env`` built from ``runners/<name>/pyproject.toml`` deps."""
    pyproject = runners_root() / name / "pyproject.toml"
    with pyproject.open("rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    return {"pip": list(deps)}


def runner_ray_remote_args(runner: str | None) -> dict[str, Any]:
    """Extra ``ray_remote_args`` for a stage: the runner's runtime_env when
    isolation is enabled, else nothing (local single-node shares the driver env)."""
    if runner is None or os.getenv(_ISOLATION_ENV) != "1":
        return {}
    return {"runtime_env": runner_env(runner)}
