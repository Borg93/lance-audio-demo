"""vLLM request/response value objects — the wire contract, pure (no network).

Two things break silently if these models drift:
  * ``ChatMessage.model_dump()`` must reproduce the EXACT OpenAI message shape the
    servers expect (a typo'd key or a missing ``type`` is a 400 from the server,
    not a Python error) — so the dump is asserted byte-for-byte.
  * the response models must read the one field each client needs AND ignore the
    rest of the reply (``id`` / ``usage`` / ``object`` …), or a future server
    version adding fields would start raising.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rmedia.clients.schemas import (
    ChatCompletionResponse,
    ChatMessage,
    EmbeddingResponse,
    ImagePart,
    ImageUrl,
    RerankResponse,
    TextPart,
)


class TestChatMessageWireShape:
    def test_text_content_dumps_openai_shape(self) -> None:
        msg = ChatMessage(role="user", content=[TextPart(text="hej")])
        assert msg.model_dump() == {
            "role": "user",
            "content": [{"type": "text", "text": "hej"}],
        }

    def test_image_content_dumps_openai_shape(self) -> None:
        msg = ChatMessage(
            role="user",
            content=[ImagePart(image_url=ImageUrl(url="data:image/jpeg;base64,AAAA"))],
        )
        assert msg.model_dump() == {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
            ],
        }

    def test_string_content_dumps_plain(self) -> None:
        # text-only LLMs (summarize) send a bare string, not a parts list.
        msg = ChatMessage(role="system", content="Summarize this.")
        assert msg.model_dump() == {"role": "system", "content": "Summarize this."}


class TestResponseParsing:
    def test_embedding_response_reads_data_and_ignores_extras(self) -> None:
        resp = EmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": "Qwen/Qwen3-VL-Embedding-2B",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 7},
            }
        )
        assert resp.data[0].embedding == [0.1, 0.2, 0.3]

    def test_chat_completion_reads_content_and_ignores_extras(self) -> None:
        resp = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-abc",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "En lila yta."}}
                ],
                "usage": {"completion_tokens": 4},
            }
        )
        assert resp.choices[0].message.content == "En lila yta."

    def test_rerank_response_reads_results(self) -> None:
        resp = RerankResponse.model_validate(
            {"results": [{"index": 2, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.4}]}
        )
        assert [(r.index, r.relevance_score) for r in resp.results] == [(2, 0.91), (0, 0.4)]

    def test_empty_choices_fails_at_the_parse_boundary(self) -> None:
        # A malformed reply with no choices must raise a ValidationError from the
        # transport parse (min_length=1), not an IndexError at choices[0] later.
        with pytest.raises(ValidationError):
            ChatCompletionResponse.model_validate({"choices": []})
