"""Knowledge-graph endpoints — lance-graph Cypher over four Lance tables.

The graph lives in ``kg_entities`` / ``kg_chunks`` / ``kg_mentions`` /
``kg_relationships`` (built offline by the LightRAG pipeline + adapter). A
single :class:`lance_graph.CypherEngine` is built over pyarrow snapshots of the
four tables and cached, keyed by the entities table's Lance ``version`` — so a
KG rebuild is picked up without a server restart (the cache key changes), while
repeat requests reuse the engine. Every endpoint returns ``built: false`` when
``kg_entities.lance`` is absent (mirroring :mod:`backend.diarization.router`).

``/cypher`` runs arbitrary read Cypher (the explorer's REPL); invalid queries
come back as a 200 with ``error`` set, not an HTTP failure. The ``q`` /
``entity_id`` inputs to the other routes are whitelisted before they are ever
inlined into a Cypher literal — entity ids are 16-char hex slugs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import lance
import lance_graph as lg
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from backend.deps import StateDep
from backend.schemas.graph import (
    CypherResponse,
    EntityClip,
    EntityCooccur,
    EntityDetail,
    EntityMatch,
    EntityNeighbor,
    GraphEdge,
    GraphEntityResponse,
    GraphNode,
    GraphSearchResponse,
    GraphStatusResponse,
    SubgraphResponse,
)

router = APIRouter(prefix="/api/graph", tags=["graph"])

#: Node/relationship label -> on-disk Lance table file under the DB dir.
_TABLE_FILES: dict[str, str] = {
    "Entity": "kg_entities.lance",
    "Chunk": "kg_chunks.lance",
    "MENTIONS": "kg_mentions.lance",
    "RELATIONSHIP": "kg_relationships.lance",
}

#: entity_id is sha1(name)[:16] — 16 lowercase hex chars. Validated before any
#: Cypher interpolation so a crafted id can't break out of the single-quoted
#: literal (the same threat ``valid_doc_id`` guards in the diarization route).
_ENTITY_ID = re.compile(r"^[0-9a-f]{16}$")

_DEFAULT_LIMIT = 200
_MAX_LIMIT = 1000

#: Trailing ``LIMIT <n>`` (the only place a LIMIT can legally sit in a single
#: read statement). Matched case-insensitively at end-of-query.
_TRAILING_LIMIT = re.compile(r"(?is)\blimit\s+(\d+)\s*$")


def _enforce_limit(query: str, cap: int) -> str:
    """Guarantee the query returns at most ``cap`` rows.

    Clamps a user-supplied trailing ``LIMIT n`` down to ``cap`` (the old code
    only *appended* a cap when the word ``limit`` was textually absent, so
    ``LIMIT 999999`` — or any query merely containing the substring "limit",
    e.g. in a string literal — escaped the cap entirely and could pull the
    whole table). Appends ``LIMIT cap`` when no trailing LIMIT is present.
    """
    trailing = _TRAILING_LIMIT.search(query)
    if trailing is None:
        return f"{query} LIMIT {cap}"
    n = min(int(trailing.group(1)), cap)
    return _TRAILING_LIMIT.sub(f"LIMIT {n}", query)


class _GraphResources(BaseModel):
    """Cached, request-shared graph handles + precomputed lookups.

    Holds the live :class:`CypherEngine` (which keeps its four pyarrow tables
    alive) plus small dicts/lists derived once so the non-Cypher routes
    (status, search, subgraph) are pure in-memory ops.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: Any  # lance_graph.CypherEngine
    ent_by_id: dict[str, dict[str, Any]]
    ent_videos: dict[str, int]
    rels: list[tuple[str, str, str]]
    n_mentions: int
    n_videos: int


# version-keyed cache: (db_path, kg_entities version) -> resources. The version
# bump on a rebuild changes the key, so a stale engine is never served.
_CACHE: dict[tuple[str, int], _GraphResources] = {}


def _build_resources(db_path: Path) -> _GraphResources:
    tables = {
        label: lance.dataset(str(db_path / fname)).to_table()
        for label, fname in _TABLE_FILES.items()
    }
    cfg = (
        lg.GraphConfigBuilder()
        .with_node_label("Entity", "entity_id")
        .with_node_label("Chunk", "chunk_id")
        .with_relationship("MENTIONS", "source_entity_id", "target_chunk_id")
        .with_relationship("RELATIONSHIP", "source_entity_id", "target_entity_id")
        .build()
    )
    engine = lg.CypherEngine(cfg, tables)

    ent_by_id = {
        row["entity_id"]: {
            "name": row["name"],
            "entity_type": row["entity_type"],
            "mention_count": int(row["mention_count"]),
        }
        for row in tables["Entity"].to_pylist()
    }
    chunk_doc = {r["chunk_id"]: r["doc_id"] for r in tables["Chunk"].to_pylist()}
    ent_videos: dict[str, set[str]] = {}
    for m in tables["MENTIONS"].to_pylist():
        doc = chunk_doc.get(m["target_chunk_id"])
        if doc is not None:
            ent_videos.setdefault(m["source_entity_id"], set()).add(doc)
    rels = [
        (r["source_entity_id"], r["target_entity_id"], r.get("description") or "")
        for r in tables["RELATIONSHIP"].to_pylist()
    ]
    return _GraphResources(
        engine=engine,
        ent_by_id=ent_by_id,
        ent_videos={k: len(v) for k, v in ent_videos.items()},
        rels=rels,
        n_mentions=tables["MENTIONS"].num_rows,
        n_videos=len(set(chunk_doc.values())),
    )


def _resources(db_path: Path) -> _GraphResources | None:
    """The cached graph engine + lookups, or ``None`` if the graph isn't built."""
    entities_path = db_path / _TABLE_FILES["Entity"]
    if not entities_path.exists() or not all((db_path / f).exists() for f in _TABLE_FILES.values()):
        return None
    version = lance.dataset(str(entities_path)).version
    key = (str(db_path), version)
    cached = _CACHE.get(key)
    if cached is None:
        cached = _build_resources(db_path)
        _CACHE[key] = cached
    return cached


def _node(res: _GraphResources, entity_id: str) -> GraphNode | None:
    props = res.ent_by_id.get(entity_id)
    if props is None:
        return None
    return GraphNode(
        id=entity_id,
        name=props["name"],
        type=props["entity_type"],
        mentions=props["mention_count"],
        videos=res.ent_videos.get(entity_id, 0),
    )


def _cell(value: Any) -> str | float | int | None:
    return value if value is None or isinstance(value, (str, int, float)) else str(value)


def _run_rows(res: _GraphResources, cypher: str) -> list[dict[str, Any]]:
    """Execute read Cypher and return rows as dicts (raises on a bad query)."""
    result = res.engine.execute(cypher)
    return result.to_pylist() if hasattr(result, "to_pylist") else list(result)


@router.get("/status")
def get_status(state: StateDep) -> GraphStatusResponse:
    """Row counts for the explorer header, or ``built: false`` if the KG is absent."""
    res = _resources(state.db_path)
    if res is None:
        return GraphStatusResponse(built=False)
    return GraphStatusResponse(
        built=True,
        entities=len(res.ent_by_id),
        relations=len(res.rels),
        mentions=res.n_mentions,
        videos=res.n_videos,
    )


class CypherRequest(BaseModel):
    """Body for the Cypher REPL endpoint."""

    query: str
    limit: int = _DEFAULT_LIMIT


@router.post("/cypher")
def run_cypher(body: CypherRequest, state: StateDep) -> CypherResponse:
    """Run arbitrary read Cypher; invalid queries return ``error`` (HTTP 200)."""
    res = _resources(state.db_path)
    if res is None:
        return CypherResponse(built=False)
    raw = body.query.strip()
    if not raw:
        return CypherResponse(built=True, error="Empty query.")
    statements = [s for s in raw.split(";") if s.strip()]
    if len(statements) > 1:
        return CypherResponse(built=True, error="Only a single statement is allowed.")
    query = statements[0].strip()
    query = _enforce_limit(query, max(1, min(body.limit, _MAX_LIMIT)))
    try:
        rows = _run_rows(res, query)
    except Exception as exc:  # noqa: BLE001 — surface the engine message in-band, REPL-style
        return CypherResponse(built=True, error=f"{type(exc).__name__}: {exc}")
    columns = list(rows[0].keys()) if rows else []
    return CypherResponse(
        built=True,
        columns=columns,
        rows=[[_cell(row.get(c)) for c in columns] for row in rows],
    )


@router.get("/search")
def search_entities(q: str, state: StateDep) -> GraphSearchResponse:
    """Entity-name substring matches (top 10 by mention count). ``q`` never hits Cypher."""
    res = _resources(state.db_path)
    if res is None:
        return GraphSearchResponse(built=False)
    needle = q.strip().lower()
    if len(needle) < 2:
        return GraphSearchResponse(built=True)
    hits = [
        EntityMatch(
            entity_id=eid,
            name=props["name"],
            entity_type=props["entity_type"],
            mention_count=props["mention_count"],
            videos=res.ent_videos.get(eid, 0),
        )
        for eid, props in res.ent_by_id.items()
        if needle in props["name"].lower()
    ]
    hits.sort(key=lambda m: m.mention_count, reverse=True)
    return GraphSearchResponse(built=True, matches=hits[:10])


@router.get("/entity/{entity_id}")
def get_entity(entity_id: str, state: StateDep) -> GraphEntityResponse:
    """An entity's properties + clips + relationship neighbours + co-occurrences.

    Clips, neighbours and co-occurrences are answered through the live
    :class:`CypherEngine` (the prod Cypher path); ``entity: None`` for an unknown
    or malformed id.
    """
    res = _resources(state.db_path)
    if res is None:
        return GraphEntityResponse(built=False)
    if not _ENTITY_ID.match(entity_id):
        return GraphEntityResponse(built=True)
    props = res.ent_by_id.get(entity_id)
    if props is None:
        return GraphEntityResponse(built=True)

    clip_rows = _run_rows(
        res,
        f"MATCH (e:Entity {{entity_id:'{entity_id}'}})-[:MENTIONS]->(c:Chunk) "
        "RETURN c.chunk_id, c.doc_id, c.namn, c.start_s, c.end_s, c.text LIMIT 40",
    )
    clips = sorted(
        (
            EntityClip(
                chunk_id=str(r["c.chunk_id"]),
                doc_id=str(r["c.doc_id"]),
                namn=str(r.get("c.namn") or r["c.doc_id"]),
                start=float(r["c.start_s"]),
                end=float(r["c.end_s"]),
                text=str(r.get("c.text") or ""),
            )
            for r in clip_rows
        ),
        key=lambda c: (c.doc_id, c.start),
    )

    neighbors: list[EntityNeighbor] = []
    for direction, pattern in (
        ("out", f"(a:Entity {{entity_id:'{entity_id}'}})-[r:RELATIONSHIP]->(b:Entity)"),
        ("in", f"(b:Entity)-[r:RELATIONSHIP]->(a:Entity {{entity_id:'{entity_id}'}})"),
    ):
        for r in _run_rows(
            res,
            f"MATCH {pattern} RETURN b.entity_id, b.name, b.entity_type, r.description LIMIT 30",
        ):
            neighbors.append(
                EntityNeighbor(
                    entity_id=str(r["b.entity_id"]),
                    name=str(r["b.name"]),
                    entity_type=str(r.get("b.entity_type") or "OTHER"),
                    direction=direction,
                    description=str(r.get("r.description") or ""),
                )
            )

    cooccur_rows = _run_rows(
        res,
        f"MATCH (a:Entity {{entity_id:'{entity_id}'}})-[:MENTIONS]->(c:Chunk)"
        "<-[:MENTIONS]-(b:Entity) "
        f"WHERE b.entity_id <> '{entity_id}' "
        "RETURN b.entity_id, b.name, count(c.chunk_id) AS shared "
        "ORDER BY shared DESC LIMIT 12",
    )
    cooccur = [
        EntityCooccur(
            entity_id=str(r["b.entity_id"]), name=str(r["b.name"]), shared=int(r["shared"])
        )
        for r in cooccur_rows
    ]

    return GraphEntityResponse(
        built=True,
        entity=EntityDetail(entity_id=entity_id, **props),
        clips=clips,
        neighbors=neighbors,
        cooccur=cooccur,
    )


@router.get("/subgraph")
def get_subgraph(
    state: StateDep, entity_id: str | None = None, limit: int = 150
) -> SubgraphResponse:
    """Nodes + edges for the force-layout view.

    No ``entity_id`` → overview (top ``limit`` entities by mention count + every
    relationship among them). With one → that entity plus its 1-hop relationship
    neighbourhood.
    """
    res = _resources(state.db_path)
    if res is None:
        return SubgraphResponse(built=False)
    limit = max(1, min(limit, _MAX_LIMIT))

    if entity_id is not None:
        if not _ENTITY_ID.match(entity_id) or entity_id not in res.ent_by_id:
            return SubgraphResponse(built=True)
        node_ids = {entity_id}
        for src, tgt, _desc in res.rels:
            if src == entity_id:
                node_ids.add(tgt)
            elif tgt == entity_id:
                node_ids.add(src)
        if len(node_ids) > limit:
            ranked = sorted(
                node_ids,
                key=lambda e: (e != entity_id, -res.ent_by_id.get(e, {}).get("mention_count", 0)),
            )
            node_ids = set(ranked[:limit])
    else:
        ranked = sorted(
            res.ent_by_id,
            key=lambda e: res.ent_by_id[e]["mention_count"],
            reverse=True,
        )
        node_ids = set(ranked[:limit])

    nodes = [n for eid in node_ids if (n := _node(res, eid)) is not None]
    seen: set[tuple[str, str]] = set()
    edges: list[GraphEdge] = []
    for src, tgt, desc in res.rels:
        if src in node_ids and tgt in node_ids and (src, tgt) not in seen:
            seen.add((src, tgt))
            edges.append(GraphEdge(source=src, target=tgt, description=desc))
    return SubgraphResponse(built=True, nodes=nodes, edges=edges)
