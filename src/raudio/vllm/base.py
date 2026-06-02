"""Shared HTTP transport for the vLLM OpenAI-style model servers.

Every model client (embedding / reranker / caption / summarize) POSTs JSON to a
long-running vLLM server and fans concurrent calls out over a thread pool —
vLLM's continuous batching fuses them into one GPU pass, so each client maps its
items across a pool sized to saturate the GPU. This transport owns the httpx
connection pool and the fan-out so the clients only shape requests + parse replies.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

DEFAULT_TIMEOUT_S = 120.0


class VLLMTransport:
    """POST JSON to a vLLM server, with a pooled client and concurrent fan-out."""

    def __init__(self, base_url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S, pool_size: int = 32) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            timeout=timeout_s,
            limits=httpx.Limits(max_connections=pool_size * 2, max_keepalive_connections=pool_size),
        )

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST ``body`` to ``{base_url}{path}`` and return the decoded JSON."""
        r = self._http.post(f"{self.base_url}{path}", json=body)
        r.raise_for_status()
        return r.json()

    def map(self, fn: Callable[[Any], Any], items: Iterable[Any], *, concurrency: int) -> list[Any]:
        """Run ``fn`` over ``items`` across a thread pool, preserving input order."""
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(fn, items))
