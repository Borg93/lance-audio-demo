"""Step 2/3 of the knowledge-graph build: LightRAG extraction.

Runs in an ISOLATED, ephemeral env (NOT the project venv) so LightRAG's
``pandas<2.4`` / ``pipmaster`` deps never collide with the project's pandas:

    uv run --no-project \
        --with lightrag-hku --with openai --with tiktoken \
        --with nano-vectordb --with networkx --with numpy \
        python scripts/kg/build_kg.py --chunks kg_work/chunks.jsonl --work kg_work/rag

Inserts each chunk as its own LightRAG document (``file_path`` = chunk key) so
entity/edge ``source_id`` resolves back to a clip. Drives the remote gemma-4-31B
(extraction) + the local Qwen embedding model, in Swedish. The ``--work`` dir
holds LightRAG's graphml + kv stores AND its doc-status — so the run is
**resumable**: re-run after any interruption and processed chunks are skipped.
Keep ``--work`` on persistent disk (not /tmp) for a multi-hour corpus run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from functools import partial
from pathlib import Path

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightRAG KG build (isolated venv).")
    parser.add_argument("--chunks", default="kg_work/chunks.jsonl")
    parser.add_argument("--work", default="kg_work/rag")
    parser.add_argument("--gemma-url", default="https://dev-kuberay.ra.se/gemma-31b/v1")
    parser.add_argument("--gemma-model", default="google/gemma-4-31B-it")
    parser.add_argument("--embed-url", default="http://localhost:8001/v1")
    parser.add_argument("--embed-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embed-dim", type=int, default=2048)
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--llm-workers", type=int, default=8)
    parser.add_argument("--max-parallel-insert", type=int, default=8)
    parser.add_argument("--language", default="Swedish")
    parser.add_argument("--batch", type=int, default=1000, help="chunks per ainsert call")
    return parser.parse_args()


ARGS = parse_args()


async def llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    keyword_extraction: bool = False,  # LightRAG-internal flag — absorb, don't forward
    **kwargs: object,
) -> str:
    return await openai_complete_if_cache(
        ARGS.gemma_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=ARGS.gemma_url,
        api_key=ARGS.api_key,
        **kwargs,
    )


embedding_func = EmbeddingFunc(
    embedding_dim=ARGS.embed_dim,
    max_token_size=8192,
    func=partial(
        openai_embed.func, model=ARGS.embed_model, base_url=ARGS.embed_url, api_key=ARGS.api_key
    ),
)


def already_done(work: Path) -> set[str]:
    """Chunk keys LightRAG has finished (or deduped) — skipped on resume.

    ``processed`` = extracted; ``failed`` = LightRAG's label for an exact-duplicate
    content hash (benign — the identical text is captured by its first occurrence).
    Anything else (pending / processing / parsing / analyzing) is re-inserted so an
    interrupted chunk is retried.
    """
    status_path = work / "kv_store_doc_status.json"
    if not status_path.exists():
        return set()
    status = json.loads(status_path.read_text())
    return {k for k, v in status.items() if v.get("status") in ("processed", "failed")}


async def main() -> None:
    work = Path(ARGS.work)
    work.mkdir(parents=True, exist_ok=True)
    chunks = [
        json.loads(line) for line in Path(ARGS.chunks).read_text().splitlines() if line.strip()
    ]

    rag = LightRAG(
        working_dir=str(work),
        llm_model_func=llm_func,
        llm_model_name=ARGS.gemma_model,
        embedding_func=embedding_func,
        llm_model_max_async=ARGS.llm_workers,
        embedding_func_max_async=4,
        embedding_batch_num=16,
        max_parallel_insert=ARGS.max_parallel_insert,
        addon_params={"language": ARGS.language},
    )
    await rag.initialize_storages()

    # Resume cheaply: only insert chunks not already finished. Inserting in
    # batches (rather than one 145k-doc call) keeps the enqueue/dedup pass small,
    # flushes progress regularly, and bounds the re-work after any interruption.
    done = already_done(work)
    pending = [(f"{c['doc_id']}:{c['speech_id']}:{c['chunk_id']}", c["text"]) for c in chunks]
    pending = [(k, t) for k, t in pending if k not in done]
    print(f"resume: {len(done)} done/dup, {len(pending)} to process (batch={ARGS.batch})")

    for start in range(0, len(pending), ARGS.batch):
        batch = pending[start : start + ARGS.batch]
        await rag.ainsert(
            [t for _, t in batch], ids=[k for k, _ in batch], file_paths=[k for k, _ in batch]
        )
        print(
            f"batch done: {start + len(batch)}/{len(pending)} pending chunks inserted", flush=True
        )

    await rag.finalize_storages()
    print(f"KG build complete -> {work}  ({len(chunks)} chunks, {len(pending)} newly processed)")


if __name__ == "__main__":
    asyncio.run(main())
