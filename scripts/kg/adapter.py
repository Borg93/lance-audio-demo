"""Step 3/3 of the knowledge-graph build: fold LightRAG output → kg_* Lance tables.

Runs in the PROJECT venv (uses lance + lance_graph + networkx). Reads the
LightRAG working dir's ``graph_chunk_entity_relation.graphml`` +
``kv_store_text_chunks.json`` and the chunks JSONL, then writes the four tables
the backend's ``/api/graph`` router queries — ``kg_entities`` / ``kg_chunks`` /
``kg_mentions`` / ``kg_relationships`` — into the Lance DB (mode=overwrite, so a
rebuild replaces the previous graph). Old table versions are reclaimed at the
end. Finally it runs a lance-graph Cypher sanity query so the fold is verified.

    uv run --with networkx python scripts/kg/adapter.py \
        --work kg_work/rag --db transcripts_v2.lance
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import lance
import lance_graph as lg
import networkx as nx
import pyarrow as pa

SEP = "<SEP>"  # GRAPH_FIELD_SEP in lightrag 1.5.x


def slug(name: str) -> str:
    return hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()[:16]


def norm_type(t: str | None) -> str:
    s = (t or "").lower()
    if "person" in s:
        return "PERSON"
    if "org" in s:
        return "ORG"
    if any(k in s for k in ("geo", "location", "plats", "place", "land", "stad")):
        return "GEO"
    if any(k in s for k in ("event", "händels", "seminar")):
        return "EVENT"
    return "OTHER"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold LightRAG output → kg_* Lance tables.")
    parser.add_argument("--work", default="kg_work/rag")
    parser.add_argument("--chunks", default="kg_work/chunks.jsonl")
    parser.add_argument("--db", default="transcripts_v2.lance")
    args = parser.parse_args()

    work = Path(args.work)
    db = Path(args.db)

    kv = json.loads((work / "kv_store_text_chunks.json").read_text())
    md5_to_key: dict[str, str] = {}
    for cid, rec in kv.items():
        key = rec.get("file_path") or rec.get("full_doc_id") or ""
        if key and key != "unknown_source":
            md5_to_key[cid] = key.split(SEP)[0]

    chunk_meta = {
        f"{c['doc_id']}:{c['speech_id']}:{c['chunk_id']}": c
        for c in (
            json.loads(line) for line in Path(args.chunks).read_text().splitlines() if line.strip()
        )
    }

    def keys_of(source_id: str | None) -> set[str]:
        out: set[str] = set()
        for tok in (source_id or "").split(SEP):
            tok = tok.strip()
            if not tok:
                continue
            out.add(md5_to_key.get(tok, tok if tok in chunk_meta else ""))
        return {k for k in out if k in chunk_meta}

    g = nx.read_graphml(work / "graph_chunk_entity_relation.graphml")

    ent_name: dict[str, str] = {}
    ent_type: dict[str, str] = {}
    ent_chunks: dict[str, set[str]] = defaultdict(set)
    for node, data in g.nodes(data=True):
        name = (data.get("entity_id") or node or "").strip()
        if not name:
            continue
        eid = slug(name)
        ent_name[eid] = name
        ent_type[eid] = norm_type(data.get("entity_type"))
        ent_chunks[eid] |= keys_of(data.get("source_id"))

    rels: list[dict] = []
    for src, tgt, data in g.edges(data=True):
        s, t = slug(str(src).strip()), slug(str(tgt).strip())
        cks = keys_of(data.get("source_id")) or {next(iter(ent_chunks[s]), "")}
        for ck in cks:
            if not ck:
                continue
            ent_chunks[s].add(ck)
            ent_chunks[t].add(ck)
            rels.append(
                {
                    "source_entity_id": s,
                    "target_entity_id": t,
                    "relationship_type": "RELATIONSHIP",
                    "description": (data.get("description") or data.get("keywords") or "")[:120],
                    "chunk_id": ck,
                    "doc_id": ck.split(":")[0],
                }
            )
        ent_name.setdefault(s, str(src).strip())
        ent_name.setdefault(t, str(tgt).strip())
        ent_type.setdefault(s, "OTHER")
        ent_type.setdefault(t, "OTHER")

    eids = sorted(ent_name)
    print(f"graph: {len(eids)} entities, {len(rels)} relation-rows")

    entity_tbl = pa.table(
        {
            "entity_id": eids,
            "name": [ent_name[e] for e in eids],
            "entity_type": [ent_type[e] for e in eids],
            "name_lower": [ent_name[e].lower() for e in eids],
            "mention_count": [len(ent_chunks[e]) for e in eids],
        }
    )
    cks = sorted({c for e in eids for c in ent_chunks[e]})
    chunk_tbl = pa.table(
        {
            "chunk_id": cks,
            "doc_id": [chunk_meta[c]["doc_id"] for c in cks],
            "namn": [chunk_meta[c]["namn"] for c in cks],
            "start_s": [chunk_meta[c]["start"] for c in cks],
            "end_s": [chunk_meta[c]["end"] for c in cks],
            "text": [(chunk_meta[c]["text"] or "")[:280] for c in cks],
        }
    )
    m_src, m_dst = [], []
    for eid, cset in ent_chunks.items():
        for ck in cset:
            m_src.append(eid)
            m_dst.append(ck)
    mentions_tbl = pa.table({"source_entity_id": m_src, "target_chunk_id": m_dst})
    cols = (
        "source_entity_id",
        "target_entity_id",
        "relationship_type",
        "description",
        "chunk_id",
        "doc_id",
    )
    rel_tbl = (
        pa.table({k: [r[k] for r in rels] for k in cols})
        if rels
        else pa.table({k: [] for k in cols}, schema=pa.schema([(k, pa.string()) for k in cols]))
    )

    tables = {
        "kg_entities": entity_tbl,
        "kg_chunks": chunk_tbl,
        "kg_mentions": mentions_tbl,
        "kg_relationships": rel_tbl,
    }
    for name, tbl in tables.items():
        path = str(db / f"{name}.lance")
        lance.write_dataset(tbl, path, mode="overwrite")
        lance.dataset(path).cleanup_old_versions(older_than=timedelta(0))

    cfg = (
        lg.GraphConfigBuilder()
        .with_node_label("Entity", "entity_id")
        .with_node_label("Chunk", "chunk_id")
        .with_relationship("MENTIONS", "source_entity_id", "target_chunk_id")
        .with_relationship("RELATIONSHIP", "source_entity_id", "target_entity_id")
        .build()
    )
    name_map = {
        "Entity": "kg_entities",
        "Chunk": "kg_chunks",
        "MENTIONS": "kg_mentions",
        "RELATIONSHIP": "kg_relationships",
    }
    ds = {
        label: lance.dataset(str(db / f"{fname}.lance")).to_table()
        for label, fname in name_map.items()
    }
    engine = lg.CypherEngine(cfg, ds)
    res = engine.execute("MATCH (a:Entity)-[:MENTIONS]->(c:Chunk) RETURN a.name, c.doc_id LIMIT 3")
    sample = res.to_pylist() if hasattr(res, "to_pylist") else list(res)
    print(f"wrote kg_* into {db} | Cypher sanity: {len(sample)} rows -> {sample[:2]}")


if __name__ == "__main__":
    main()
