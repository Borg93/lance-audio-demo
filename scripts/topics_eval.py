# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "toponymy",
#   "openai>=1.40",
#   "pylance>=4.0",
#   "numpy<3",
# ]
# ///
"""Quick, NON-DESTRUCTIVE topic-modelling sample for evaluation.

Fits Toponymy on a spread sample of chunks (reusing text + text_embedding + the
EVōC atlas map) and PRINTS the Swedish topic names per layer with cluster sizes.
Writes nothing to the DB — just shows what `ratch feature topics` would produce,
so you can judge quality before the full run. Uses the async Gemma namer for
fast concurrent naming. Runs in its own isolated uv env (transformers<5 sealed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_topics import SWEDISH_STOPWORDS  # DRY: reuse the one stop-word list


def main() -> int:
    ap = argparse.ArgumentParser(description="Print sample Swedish topics (no DB writes).")
    ap.add_argument("--db", default="transcripts_v2.lance")
    ap.add_argument("--n", type=int, default=4000, help="sample size (spread across the corpus)")
    ap.add_argument("--base-min-cluster-size", type=int, default=50)
    ap.add_argument("--max-layers", type=int, default=2)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8003/v1")
    ap.add_argument("--llm-model", default="google/gemma-4-31B-it")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--embed-model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    import os

    # The async namer doesn't reliably forward base_url to the OpenAI client, so
    # it would hit api.openai.com (401). Point the client at the local Gemma via
    # env; the embedder's explicit base_url (:8001) still overrides this for itself.
    os.environ["OPENAI_API_KEY"] = "local"
    os.environ["OPENAI_BASE_URL"] = args.llm_url

    import lance
    from toponymy import KeyphraseBuilder, Toponymy, ToponymyClusterer
    from toponymy.embedding_wrappers import OpenAIEmbedder

    # Sync namer: AsyncOpenAINamer binds its semaphore to the first event loop and
    # Toponymy runs one asyncio.run() per layer → "bound to a different event loop".
    from toponymy.llm_wrappers import OpenAINamer

    ds = lance.dataset(str(Path(args.db) / "chunks.lance"))
    total = ds.count_rows()
    step = max(1, total // args.n)
    idx = list(range(0, total, step))[: args.n]
    tbl = ds.take(idx, columns=["text", "text_embedding", "atlas_x", "atlas_y"])

    documents = [t or "" for t in tbl.column("text").to_pylist()]
    embedding_vectors = np.asarray(tbl.column("text_embedding").to_pylist(), dtype=np.float32)
    clusterable_vectors = np.column_stack(
        [
            np.asarray(tbl.column("atlas_x").to_pylist(), dtype=np.float32),
            np.asarray(tbl.column("atlas_y").to_pylist(), dtype=np.float32),
        ]
    )
    print(f"SAMPLE: {len(documents)} of {total} chunks", file=sys.stderr)

    embedder = OpenAIEmbedder(api_key="local", base_url=args.embed_url, model=args.embed_model)
    namer = OpenAINamer(
        api_key="local",
        base_url=args.llm_url,
        model=args.llm_model,
        llm_specific_instructions="Svara alltid på svenska. Ge korta, beskrivande och distinkta ämnesnamn.",
    )
    model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=ToponymyClusterer(
            min_clusters=4, base_min_cluster_size=args.base_min_cluster_size,
            max_layers=args.max_layers, verbose=True,
        ),
        keyphrase_builder=KeyphraseBuilder(stop_words=SWEDISH_STOPWORDS, embedder=embedder, verbose=True),
        object_description="en kort transkriberad mening ur en svensk arkivvideo (tal, anförande eller intervju)",
        corpus_description="transkriberade svenska arkivvideor från Riksarkivet",
        verbose=True,
    )
    model.fit(documents, embedding_vectors, clusterable_vectors)

    for layer_id, layer in enumerate(model.cluster_layers_):
        names = model.topic_names_[layer_id]
        labels = layer.cluster_labels
        sizes = np.bincount(labels[labels >= 0], minlength=len(names))
        order = np.argsort(sizes)[::-1]
        print(f"\n=== LAYER {layer_id}: {len(names)} topics · {(labels >= 0).sum()} clustered ===")
        for ci in order[:15]:
            if sizes[ci] > 0:
                print(f"  {int(sizes[ci]):>5}  {names[ci]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
