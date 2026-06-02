"""Frame captioning client: image → short text caption via a vLLM VLM server.

A generative counterpart to the embedding/reranker clients — it asks a
vision-language model (chat-completions) to describe each frame in one line. The
caption is stored as a plain ``string`` feature column on ``chunk_frames`` and
indexed for FTS, so visual content becomes keyword-searchable.
"""

from __future__ import annotations

from typing import Protocol

from .base import DEFAULT_TIMEOUT_S, VLLMTransport
from .image import image_to_data_url

CAPTION_MODEL = "Qwen/Qwen3-VL-Instruct-2B"
DEFAULT_CAPTION_URL = "http://127.0.0.1:8003"
CAPTION_INSTRUCTION = "Describe this video frame in one concise sentence."
CAPTION_CONCURRENCY = 8
CAPTION_MAX_TOKENS = 64

_ChatMessage = dict[str, object]


class CaptionClient(Protocol):
    """Images → one caption string each. The contract the caption feature needs."""

    def caption(self, images: list[bytes]) -> list[str]: ...


class VLLMCaptionClient:
    """HTTP client for a vLLM chat-completions VLM server.

    ``POST {caption_url}/v1/chat/completions`` with one image per request; returns
    the assistant's text. Errors surface as :class:`httpx.HTTPError`.
    """

    def __init__(
        self,
        caption_url: str = DEFAULT_CAPTION_URL,
        *,
        model: str = CAPTION_MODEL,
        instruction: str = CAPTION_INSTRUCTION,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        concurrency: int = CAPTION_CONCURRENCY,
        max_tokens: int = CAPTION_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self._t = VLLMTransport(caption_url, timeout_s=timeout_s, pool_size=concurrency)

    def caption(self, images: list[bytes]) -> list[str]:
        """Return one caption per image, in input order."""
        if not images:
            return []
        return self._t.map(self._caption_one, images, concurrency=self.concurrency)

    def _caption_one(self, image: bytes) -> str:
        messages: list[_ChatMessage] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    {"type": "text", "text": self.instruction},
                ],
            }
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        return self._t.post("/v1/chat/completions", body)["choices"][0]["message"]["content"].strip()
