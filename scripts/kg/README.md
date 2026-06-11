# Knowledge-graph build (LightRAG → lance-graph)

Builds the `kg_entities` / `kg_chunks` / `kg_mentions` / `kg_relationships` Lance
tables that the backend's `/api/graph` router queries (the `/graph` explorer
page). Three steps because LightRAG's deps (`pandas<2.4`, `pipmaster`) must stay
**isolated** from the project venv — only steps 1 and 3 touch project code.

```bash
# 1. export chunks → JSONL (project venv)
uv run python scripts/kg/export_chunks.py --db transcripts_v2.lance --out kg_work/chunks.jsonl
#    scope a smaller run:   --max-per-doc 50      (cap chunks/video)
#                           --limit 5000          (cap total chunks)

# 2. LightRAG extraction (ISOLATED ephemeral env — never the project venv)
uv run --no-project \
    --with lightrag-hku --with openai --with tiktoken \
    --with nano-vectordb --with networkx --with numpy \
    python scripts/kg/build_kg.py --chunks kg_work/chunks.jsonl --work kg_work/rag \
    --gemma-url http://localhost:8003/v1 --gleaning 0 --persist-interval 300 --dummy-embeddings
#    resumable: re-run after any interruption; processed chunks are skipped.
#    PREFER your LOCAL gemma at :8003 (single-tenant, no network) over the remote
#    default https://dev-kuberay.ra.se/gemma-31b/v1 — same model, identical output.
#    --gleaning 0 / --persist-interval / --dummy-embeddings are the speed flags
#    (8 → ~190 docs/min). Corpus scale: shard with --num-shards N --shard-index i
#    into per-shard --work dirs, then fold them all in step 3.
#    embeddings default to the local Qwen at http://localhost:8001/v1.

# 3. fold LightRAG output → kg_* Lance tables (project venv). DETERMINISTIC, no LLM.
uv run --with networkx python scripts/kg/adapter.py --work kg_work/rag --db transcripts_v2.lance
#    pass several --work dirs to fold sharded runs in one pass.
#    adapter.py + generic_sv.py clean junk and demote generic group/role/category
#    nouns to OTHER (Politiker/Konsumenter as PERSON, Arbetsplats/Skola as GEO,
#    Rapporten as WORK, Möte/Debatt as EVENT, Företag as ORG) by pure morphology +
#    blocklist — byte-stable across runs. refine_person_types.py is an OPTIONAL
#    manual LLM tie-breaker, no longer part of the build.
#    overwrites kg_* in the DB + reclaims old versions; the /api/graph router's
#    version-keyed cache picks up the new graph with no server restart.
```

`kg_work/` is build scratch (chunks JSONL + LightRAG graphml/kv/doc-status) — keep
it on persistent disk for a multi-hour corpus run, and gitignore it. Only the
four `kg_*.lance` tables under the DB are the durable output.
