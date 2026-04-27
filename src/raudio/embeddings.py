"""Multimodal embedding + reranking client for raudio.

Single integration point for Qwen3-VL-Embedding-8B (text + image) and
Qwen3-VL-Reranker-8B, served as long-running vLLM HTTP servers (online
inference). Both the CLI commands (`embed-chunks`, `embed-chunk-frames`)
and the FastAPI backend (`/api/search?mode=semantic`) are HTTP clients
of those servers — the model loads once and stays warm across all uses.

Why this architecture (vs in-process):
    Loading Qwen3-VL-Embedding-8B with transformers takes 30–60 s and
    pins ~25 GB of GPU memory. Doing that inside every CLI invocation
    means every `raudio embed-chunks` resume re-pays the cost; the
    FastAPI worker would also re-load on each restart. Online vLLM
    amortizes the load and gets continuous-batching throughput for free.

The vLLM servers are launched via `make embed-server` / `make rerank-server`.

Optional dependencies live behind the ``[multimodal]`` extra
(``uv sync --extra multimodal``). We lazy-import inside ``__init__`` so
that `raudio --help` / `raudio ingest …` etc. stay snappy and don't
break on installs that skipped the extra.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Protocol


logger = logging.getLogger(__name__)


# Default vLLM server endpoints (override via `make_client(...)` or env).
DEFAULT_EMBED_URL: str = "http://127.0.0.1:8001"
DEFAULT_RERANK_URL: str = "http://127.0.0.1:8002"

# Matryoshka dim — Qwen3-VL emits 4096-d, we slice to this for storage.
DEFAULT_MRL_DIM: int = 1024

# Qwen system instruction wrapping every embed call. Per the model card,
# instructions in English yield best results even when content is Swedish.
DEFAULT_EMBED_INSTRUCTION: str = "Represent the user's input."

# Reranker prompt scaffolding — verbatim from the Qwen3-VL-Reranker model
# card. The vLLM server reads `query` + `documents` and runs each (q, d)
# pair through these templates internally; we send the cleaned strings.
RERANKER_INSTRUCTION: str = (
    "Given a search query, retrieve relevant candidates that answer the query."
)
_RERANKER_PREFIX: str = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
_RERANKER_SUFFIX: str = "<|im_end|>\n<|im_start|>assistant\n"


_EXTRA_HINT = (
    "The multimodal embedding client requires the [multimodal] extra.\n"
    "Install with:  uv sync --extra multimodal\n"
    "           or: pip install 'raudio[multimodal]'"
)


# ─────────────────────────────────────────────────────────────────────
# Public protocol
# ─────────────────────────────────────────────────────────────────────


class EmbeddingClient(Protocol):
    """Backend-agnostic embedding + rerank surface.

    All methods return numpy arrays / lists of floats. Embeddings are
    L2-normalized and MRL-truncated to ``DEFAULT_MRL_DIM`` (1024) so they
    can be written directly to the Lance ``list<float32>(1024)`` columns
    and compared via cosine distance.
    """

    def embed_text(self, texts: list[str]) -> "Any":
        """Return ``(N, 1024) float32`` array, L2-normalized."""

    def embed_image(self, images: "list[Any]") -> "Any":
        """Return ``(N, 1024) float32`` array, L2-normalized.

        Accepts a list of PIL.Image.Image OR raw JPEG bytes — the client
        normalizes either to a base64 data URL before sending.
        """

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Return one relevance score per candidate (higher = more relevant)."""


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def truncate_to_mrl(vectors: "Any", dim: int = DEFAULT_MRL_DIM) -> "Any":
    """Slice MRL embeddings to ``dim`` and re-L2-normalize.

    Qwen3-VL embeddings are Matryoshka-trained — taking the first ``dim``
    components and re-normalizing is the official truncation path. Don't
    PCA, don't pool — just slice + normalize.
    """
    import numpy as np

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < dim:
        raise ValueError(
            f"Cannot MRL-truncate to {dim}: input has only {arr.shape[1]} dims"
        )
    sliced = arr[:, :dim]
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (sliced / norms).astype(np.float32)


# vLLM warmup vs runtime token-count jitter (even with min/max pixel pins)
# kills the engine with `num_tokens=N > buffer=N-k`. Workaround: pre-size
# every image to the *exact* resolution the server's Qwen processor is
# pinned to (see `make embed-server-docker` → min_pixels == max_pixels).
# 448×448 = 200,704 px → 16×16 raw patches → 8×8 = 64 vision tokens after
# 2× spatial-merge. Identical for every image, so the warmup buffer always
# matches. Aspect ratio is sacrificed (center-crop) — fine for whole-image
# similarity embedding.
_IMAGE_SIDE: int = 448


def _square_crop_resize(image: "Any") -> "Any":
    from PIL import Image

    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if image.size != (_IMAGE_SIDE, _IMAGE_SIDE):
        image = image.resize((_IMAGE_SIDE, _IMAGE_SIDE), Image.LANCZOS)
    return image


def _image_to_data_url(image: "Any") -> str:
    """Convert PIL.Image or raw JPEG bytes into ``data:image/jpeg;base64,…``.

    Resizes to a fixed 56-multiple grid so vision-token count matches the
    vLLM warmup profile.
    """
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise SystemExit(_EXTRA_HINT + f"\n(underlying error: {e})") from e

    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(bytes(image)))
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"embed_image expects PIL.Image.Image or bytes, got {type(image).__name__}"
        )

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    image = _square_crop_resize(image)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ─────────────────────────────────────────────────────────────────────
# vLLM HTTP client
# ─────────────────────────────────────────────────────────────────────


class VLLMClient:
    """HTTP client for a long-running vLLM Qwen3-VL embedding/reranker pair.

    Embedding:
        POST {embed_url}/v1/embeddings — OpenAI-compatible. We send the
        Qwen-VL chat-shaped prompt (system="Represent the user's input.",
        user=text or image+text) using the ``messages`` extension with
        ``continue_final_message=True``.

    Reranker:
        POST {rerank_url}/v1/rerank — Qwen3-VL-Reranker exposes a `score`
        endpoint when launched with the right ``hf_overrides`` and chat
        template (see Makefile target). We pass query + document strings
        wrapped in the model card's prefix/suffix scaffolding and read
        the relevance_score per result.

    Errors surface as :class:`httpx.HTTPError` subclasses; callers are
    expected to wrap them into structured 503s for the API.
    """

    def __init__(
        self,
        embed_url: str = DEFAULT_EMBED_URL,
        rerank_url: str = DEFAULT_RERANK_URL,
        *,
        embed_model: str = "Qwen/Qwen3-VL-Embedding-8B",
        rerank_model: str = "Qwen/Qwen3-VL-Reranker-8B",
        embed_instruction: str = DEFAULT_EMBED_INSTRUCTION,
        rerank_instruction: str = RERANKER_INSTRUCTION,
        mrl_dim: int = DEFAULT_MRL_DIM,
        timeout: float = 120.0,
        # In-flight HTTP requests per call. vLLM's continuous batching
        # fuses concurrent requests into one GPU pass, so the right number
        # is "as high as the server can saturate the GPU" — 16 is a safe
        # default that gives ~10-15x speedup over serial RTT.
        concurrency_text: int = 16,
        concurrency_image: int = 4,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise SystemExit(_EXTRA_HINT + f"\n(underlying error: {e})") from e
        import httpx

        self.embed_url = embed_url.rstrip("/")
        self.rerank_url = rerank_url.rstrip("/")
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.embed_instruction = embed_instruction
        self.rerank_instruction = rerank_instruction
        self.mrl_dim = mrl_dim
        self.concurrency_text = concurrency_text
        self.concurrency_image = concurrency_image
        # httpx default pool is 10; bump for our concurrency targets.
        self._client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max(concurrency_text, concurrency_image) * 2,
                max_keepalive_connections=max(concurrency_text, concurrency_image),
            ),
        )

    # ── text embedding ───────────────────────────────────────────────

    def _text_messages(self, text: str) -> list[dict[str, Any]]:
        """Build the system+user+assistant chat shape Qwen3-VL expects."""
        return [
            {"role": "system", "content": [{"type": "text", "text": self.embed_instruction}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]

    def _post_embeddings(self, messages: list[dict[str, Any]]) -> list[float]:
        """Single call to /v1/embeddings using the chat-extension shape."""
        body = {
            "model": self.embed_model,
            "messages": messages,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        }
        r = self._client.post(f"{self.embed_url}/v1/embeddings", json=body)
        r.raise_for_status()
        data = r.json()
        return data["data"][0]["embedding"]

    def embed_text(self, texts: list[str]) -> "Any":
        if not texts:
            import numpy as np
            return np.zeros((0, self.mrl_dim), dtype="float32")
        # vLLM's chat-embeddings endpoint takes one chat at a time, but
        # it has internal continuous batching — sending many requests
        # concurrently lets the engine fuse them into one GPU pass.
        # Throughput improves ~10-15x vs serial RTT.
        from concurrent.futures import ThreadPoolExecutor

        msgs = [self._text_messages(t) for t in texts]
        with ThreadPoolExecutor(max_workers=self.concurrency_text) as pool:
            vectors = list(pool.map(self._post_embeddings, msgs))
        return truncate_to_mrl(vectors, self.mrl_dim)

    # ── image embedding ──────────────────────────────────────────────

    def _image_messages(self, image: "Any", text: str = "") -> list[dict[str, Any]]:
        url = _image_to_data_url(image)
        user_content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url}},
        ]
        if text:
            user_content.append({"type": "text", "text": text})
        else:
            # Qwen3-VL embedding examples include an empty trailing text
            # block in pure-image mode.
            user_content.append({"type": "text", "text": ""})
        return [
            {"role": "system", "content": [{"type": "text", "text": self.embed_instruction}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]

    def embed_image(self, images: "list[Any]") -> "Any":
        if not images:
            import numpy as np
            return np.zeros((0, self.mrl_dim), dtype="float32")
        from concurrent.futures import ThreadPoolExecutor

        msgs = [self._image_messages(img) for img in images]
        with ThreadPoolExecutor(max_workers=self.concurrency_image) as pool:
            vectors = list(pool.map(self._post_embeddings, msgs))
        return truncate_to_mrl(vectors, self.mrl_dim)

    # ── rerank ───────────────────────────────────────────────────────

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        # Per the Qwen3-VL-Reranker model card: wrap each side in the
        # reranker's prefix/suffix template before posting. The vLLM
        # server, configured with classifier_from_token=["no", "yes"],
        # returns relevance scores ∈ [0, 1] (softmax over those two
        # tokens, projected to "yes" probability).
        q_template = (
            f"{_RERANKER_PREFIX}<Instruct>: {self.rerank_instruction}\n"
            f"<Query>: {query}\n"
        )
        docs_template = [f"<Document>: {c}{_RERANKER_SUFFIX}" for c in candidates]
        body = {
            "model": self.rerank_model,
            "query": q_template,
            "documents": docs_template,
        }
        r = self._client.post(f"{self.rerank_url}/v1/rerank", json=body)
        r.raise_for_status()
        data = r.json()
        # Results may come back unordered → preserve our index ordering.
        scored = [(item["index"], item["relevance_score"]) for item in data["results"]]
        scored.sort(key=lambda x: x[0])
        return [s for _, s in scored]


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────


def make_client(
    backend: str = "vllm",
    *,
    embed_url: str = DEFAULT_EMBED_URL,
    rerank_url: str = DEFAULT_RERANK_URL,
    mrl_dim: int = DEFAULT_MRL_DIM,
    timeout: float = 120.0,
) -> EmbeddingClient:
    """Pick an embedding backend.

    ``vllm`` (default): HTTP client of the long-running vLLM servers.

    Other backends (e.g. inline transformers) can be slotted in here
    later behind the same protocol.
    """
    if backend == "vllm":
        return VLLMClient(
            embed_url=embed_url,
            rerank_url=rerank_url,
            mrl_dim=mrl_dim,
            timeout=timeout,
        )
    raise ValueError(f"Unknown embedding backend: {backend!r} (expected: 'vllm')")


# ─────────────────────────────────────────────────────────────────────
# Custom Lance reranker
# ─────────────────────────────────────────────────────────────────────


def get_qwen_vl_reranker(client: EmbeddingClient, top_k_to_rerank: int = 100):
    """Build a :class:`lancedb.rerankers.Reranker` backed by Qwen3-VL-Reranker.

    Returned object plugs straight into Lance's hybrid query API:

        chunks.query() \
            .full_text_search(q) \
            .nearest_to(vec) \
            .rerank(get_qwen_vl_reranker(client)) \
            .limit(20)

    Implemented as a factory so the module imports cleanly without
    lancedb installed for the (rare) case where someone imports
    `raudio.embeddings` outside the API server.
    """
    from lancedb.rerankers import Reranker
    import pyarrow as pa

    class QwenVLReranker(Reranker):
        """Cross-encoder rerank using Qwen3-VL-Reranker-8B via vLLM."""

        def __init__(self, client: EmbeddingClient, top_k: int = 100) -> None:
            super().__init__()
            self.client = client
            self.top_k = top_k

        def _score(self, query: str, table: "pa.Table") -> "pa.Table":
            if table.num_rows == 0:
                return table.append_column(
                    "_relevance_score",
                    pa.array([], type=pa.float32()),
                )
            keep = min(self.top_k, table.num_rows)
            head = table.slice(0, keep)
            # `text` column is always present in our chunks table; the
            # FTS path keeps it, vector path also selects it.
            docs = head.column("text").to_pylist()
            scores = self.client.rerank(query, docs)
            scored = head.append_column(
                "_relevance_score",
                pa.array(scores, type=pa.float32()),
            )
            return scored.sort_by([("_relevance_score", "descending")])

        def rerank_hybrid(
            self, query: str, vector_results: "pa.Table", fts_results: "pa.Table"
        ) -> "pa.Table":
            merged = self.merge_results(vector_results, fts_results)
            return self._score(query, merged)

        def rerank_vector(self, query: str, vector_results: "pa.Table") -> "pa.Table":
            return self._score(query, vector_results)

        def rerank_fts(self, query: str, fts_results: "pa.Table") -> "pa.Table":
            return self._score(query, fts_results)

    return QwenVLReranker(client, top_k=top_k_to_rerank)
