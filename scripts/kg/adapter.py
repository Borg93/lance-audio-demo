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
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import lance
import lance_graph as lg
import networkx as nx
import pyarrow as pa

SEP = "<SEP>"  # GRAPH_FIELD_SEP in lightrag 1.5.x

# Pronouns / meta-speech "entities" the LLM extracts from first-person Swedish
# transcripts — zero informational value and they poison co-occurrence queries.
STOPWORDS = {
    "jag", "vi", "man", "du", "ni", "han", "hon", "de", "dem", "det", "den",
    "alla", "andra", "många", "ingen", "någon", "folk", "talaren", "talare",
    "moderator", "moderatorn", "publiken", "deltagarna", "deltagare",
    "åhörarna", "frågeställaren", "frågor", "frågan", "kronor", "procent",
}

_NUMERIC = re.compile(r"[\d\s.,:%–—-]+")
# bare amounts ("55 Miljoner", "4 Procent", "106 År") — but NOT decades
# ("1980-talet"), which are real discourse concepts
_AMOUNT = re.compile(r"\d[\d\s.,]*\s*(procent|kronor|miljoner|miljarder|år|%)")
_EDGE_PUNCT = " \t\n'\"«»“”*–—-,;:."


def clean_name(raw: str) -> str:
    """Strip extraction artifacts — angle brackets, backslashes, and stray
    edge punctuation — so '<Chirac>', ',Kommuner' and '::Staten' fold into
    the same entities as their clean forms."""
    return raw.strip().strip("<>").replace("\\", "").strip(_EDGE_PUNCT)


def is_junk(name: str) -> bool:
    low = name.lower()
    return (
        len(name) <= 1
        or low in STOPWORDS
        or bool(_NUMERIC.fullmatch(name))
        or bool(_AMOUNT.fullmatch(low))
    )


_DESC_MAX = 160


def truncate_desc(text: str, limit: int = _DESC_MAX) -> str:
    """Cap a relationship description at a word boundary with an ellipsis.

    A hard ``text[:120]`` slice cut mid-word (``...har en etablerad ans``) on
    169 edges. Trim back to the last space within budget and append ``…`` so
    the UI shows a complete-looking phrase; fall back to a hard slice only when
    a single word already exceeds the budget.
    """
    text = " ".join(text.split())  # collapse the <SEP>-join whitespace too
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    spaced = cut.rsplit(" ", 1)[0]
    return (spaced if len(spaced) >= limit // 2 else cut).rstrip() + "…"


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
    parser.add_argument(
        "--work",
        nargs="+",
        default=["kg_work/rag"],
        help="one or more LightRAG work dirs — sharded builds fold into one graph "
        "(entity identity is name-based, so the union equals a single-run graph)",
    )
    parser.add_argument("--chunks", default="kg_work/chunks.jsonl")
    parser.add_argument("--db", default="transcripts_v2.lance")
    parser.add_argument(
        "--type-overrides",
        default="",
        help="JSON {entity_id: TYPE} corrections applied after norm_type — "
        "produced by refine_person_types.py to demote generic 'persons' "
        "(Barn, Forskare, Jag...) to OTHER",
    )
    args = parser.parse_args()

    db = Path(args.db)

    chunk_meta = {
        f"{c['doc_id']}:{c['speech_id']}:{c['chunk_id']}": c
        for c in (
            json.loads(line) for line in Path(args.chunks).read_text().splitlines() if line.strip()
        )
    }

    ent_name: dict[str, str] = {}
    ent_type: dict[str, str] = {}
    ent_chunks: dict[str, set[str]] = defaultdict(set)
    rels: list[dict] = []
    seen_rels: set[tuple[str, str, str]] = set()

    for workdir in args.work:
        work = Path(workdir)
        if not (work / "graph_chunk_entity_relation.graphml").exists():
            print(f"fold {work}: SKIPPED (no graphml)")
            continue

        kv = json.loads((work / "kv_store_text_chunks.json").read_text())
        md5_to_key: dict[str, str] = {}
        for cid, rec in kv.items():
            key = rec.get("file_path") or rec.get("full_doc_id") or ""
            if key and key != "unknown_source":
                md5_to_key[cid] = key.split(SEP)[0]

        def keys_of(source_id: str | None, _md5: dict[str, str] = md5_to_key) -> set[str]:
            out: set[str] = set()
            for tok in (source_id or "").split(SEP):
                tok = tok.strip()
                if not tok:
                    continue
                out.add(_md5.get(tok, tok if tok in chunk_meta else ""))
            return {k for k in out if k in chunk_meta}

        g = nx.read_graphml(work / "graph_chunk_entity_relation.graphml")
        print(f"fold {work}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

        for node, data in g.nodes(data=True):
            name = clean_name(data.get("entity_id") or node or "")
            if not name or is_junk(name):
                continue
            eid = slug(name)
            ent_name[eid] = name
            new_type = norm_type(data.get("entity_type"))
            if ent_type.get(eid) in (None, "OTHER"):  # prefer a concrete type across shards
                ent_type[eid] = new_type
            ent_chunks[eid] |= keys_of(data.get("source_id"))

        for src, tgt, data in g.edges(data=True):
            src_name, tgt_name = clean_name(str(src)), clean_name(str(tgt))
            if not src_name or not tgt_name or is_junk(src_name) or is_junk(tgt_name):
                continue
            # LightRAG relations are undirected — canonicalize the pair order
            # so the same relation never lands as BOTH a->b and b->a (which
            # happens across shards and renders as double edges)
            s, t = sorted((slug(src_name), slug(tgt_name)))
            # merged descriptions are <SEP>-joined; keep the first fragment so
            # the raw delimiter never leaks into the UI, then word-boundary cap
            desc = truncate_desc((data.get("description") or data.get("keywords") or "").split(SEP)[0])
            cks = keys_of(data.get("source_id")) or {next(iter(ent_chunks[s]), "")}
            for ck in cks:
                if not ck or (s, t, ck) in seen_rels:
                    continue
                seen_rels.add((s, t, ck))
                ent_chunks[s].add(ck)
                ent_chunks[t].add(ck)
                rels.append(
                    {
                        "source_entity_id": s,
                        "target_entity_id": t,
                        "relationship_type": "RELATIONSHIP",
                        "description": desc,
                        "chunk_id": ck,
                        "doc_id": ck.split(":")[0],
                    }
                )
            ent_name.setdefault(s, src_name)
            ent_name.setdefault(t, tgt_name)
            ent_type.setdefault(s, "OTHER")
            ent_type.setdefault(t, "OTHER")

    # ── alias merging ────────────────────────────────────────────────────────
    # 1) Swedish definite/plural suffix duplicates (Kommun/Kommunen/Kommunerna)
    #    merge non-PERSON variants into the most-mentioned form.
    # 2) Single-token person names merge into a multi-token superset ONLY when
    #    exactly one candidate exists ('Bosse'→'Bosse Ringholm'; 'Pettersson'
    #    with two Petterssons stays split — ambiguity blocks the merge).
    alias: dict[str, str] = {}
    by_lower = {ent_name[e].lower(): e for e in ent_name}
    suffix_groups: dict[str, set[str]] = defaultdict(set)  # base form -> all variants
    for eid in list(ent_name):
        if ent_type.get(eid) == "PERSON":
            continue
        low = ent_name[eid].lower()
        for suf in ("arna", "erna", "orna", "en", "et", "na"):
            base = low[: -len(suf)] if low.endswith(suf) else ""
            if len(base) >= 4 and base in by_lower:
                other = by_lower[base]
                if other != eid and ent_type.get(other) != "PERSON":
                    # group by base so Kommun/Kommunen/Kommunerna ALL collapse
                    # into one entity, not just pairwise with the base form
                    suffix_groups[base] |= {other, eid}
                break
    for cluster in suffix_groups.values():
        win = max(cluster, key=lambda e: len(ent_chunks[e]))
        for lose in cluster:
            if lose != win:
                alias[lose] = win
    multi = [
        (eid, set(ent_name[eid].lower().split()))
        for eid in ent_name
        if ent_type.get(eid) == "PERSON" and len(ent_name[eid].split()) > 1
    ]
    for eid in list(ent_name):
        if ent_type.get(eid) != "PERSON" or eid in alias:
            continue
        toks = ent_name[eid].lower().split()
        if len(toks) != 1:
            continue
        cands = [m for m, mtoks in multi if toks[0] in mtoks and m != eid]
        if len(cands) == 1:
            alias[eid] = cands[0]

    def resolve(eid: str) -> str:
        seen: set[str] = set()
        while eid in alias and eid not in seen:
            seen.add(eid)
            eid = alias[eid]
        return eid

    merged = 0
    for lose in list(alias):
        win = resolve(lose)
        if win == lose or lose not in ent_name:
            continue
        ent_chunks[win] |= ent_chunks.pop(lose, set())
        ent_name.pop(lose, None)
        ent_type.pop(lose, None)
        merged += 1
    if merged:
        remapped: list[dict] = []
        seen_rels.clear()
        for r in rels:
            # re-sort after alias resolution to keep the undirected canonical order
            s, t = sorted((resolve(r["source_entity_id"]), resolve(r["target_entity_id"])))
            key = (s, t, r["chunk_id"])
            if s == t or key in seen_rels:
                continue
            seen_rels.add(key)
            remapped.append({**r, "source_entity_id": s, "target_entity_id": t})
        rels = remapped
    print(f"alias merge: {merged} entities folded into their canonical form")

    if args.type_overrides:
        corrections = json.loads(Path(args.type_overrides).read_text())
        applied = 0
        for eid, etype in corrections.items():
            if eid in ent_name:
                ent_type[eid] = etype
                applied += 1
        print(f"type overrides: {applied} applied from {args.type_overrides}")

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
