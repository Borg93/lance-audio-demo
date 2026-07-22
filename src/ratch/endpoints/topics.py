"""Client for the `topics` model service (Toponymy topic modelling).

The service lives in ``services/models/topics/`` with its own env (toponymy
pins transformers<5, isolated from ratch). ratch calls it through this client
and never imports toponymy. Pre-merge: run it in its sealed env
(``LocalTopicsClient``). At merge: POST to the Ray Serve deployment
(``RemoteTopicsClient``), same interface, zero stage change.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Protocol

#: URL of the topics Ray Serve deployment; unset → run locally in the sealed env.
_TOPICS_URL_ENV = "MEDIA_TOPICS_URL"
#: The service dir whose pyproject defines the sealed env for the local path.
_SERVICE_DIR = "services/models/topics"


class TopicsClient(Protocol):
    """Build topic layers on a Lance DB's chunks; returns the row count."""

    def build(self, db_path: Path, *, llm_url: str | None = None) -> int: ...


class LocalTopicsClient:
    """Pre-merge impl: run the service's worker in ITS sealed env.

    ``uv run --project services/models/topics`` resolves that dir's pyproject
    (transformers<5 etc.) — the conflicting deps never touch ratch's env. Same
    isolation the vLLM servers get, just process-local instead of over HTTP.
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        # repo root = three parents up from this file (src/ratch/endpoints/…).
        self._root = repo_root or Path(__file__).resolve().parents[3]

    def command(self, db_path: Path, *, llm_url: str | None = None) -> list[str]:
        """The sealed-env command (pure — unit-testable without running it)."""
        cmd = [
            "uv", "run", "--project", str(self._root / _SERVICE_DIR),
            "topics-worker", "--db", str(db_path),
        ]
        if llm_url:
            cmd += ["--llm-url", llm_url]
        return cmd

    def build(self, db_path: Path, *, llm_url: str | None = None) -> int:
        proc = subprocess.run(
            self.command(db_path, llm_url=llm_url),
            stdout=subprocess.PIPE, text=True, check=False, cwd=self._root,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"topics service failed (exit {proc.returncode})")
        lines = (proc.stdout or "").strip().splitlines()
        try:
            return int(lines[-1]) if lines else 0
        except ValueError:
            return 0


class RemoteTopicsClient:
    """Merge-time impl: POST to the topics Ray Serve deployment."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def build(self, db_path: Path, *, llm_url: str | None = None) -> int:
        import httpx

        body = {"db": str(db_path), "llm_url": llm_url}
        resp = httpx.post(f"{self._base_url}/topics", json=body, timeout=None)
        resp.raise_for_status()
        return int(resp.json()["rows"])


def get_topics_client() -> TopicsClient:
    """Remote when ``MEDIA_TOPICS_URL`` is set (Ray Serve), else the sealed-env
    local run — the endpoint-agnostic selection, so both drop in at merge."""
    url = os.getenv(_TOPICS_URL_ENV)
    return RemoteTopicsClient(url) if url else LocalTopicsClient()
