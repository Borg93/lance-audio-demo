# Knowledge graph — using `/graph` and writing Cypher

The graph is extracted from the press-conference transcripts by LightRAG
(gemma-4-31B entity/relation extraction, Swedish), folded into four Lance
tables (`kg_entities` / `kg_chunks` / `kg_mentions` / `kg_relationships`), and
queried **live** by [lance-graph]'s Cypher engine via `GET/POST /api/graph/*`.
How to (re)build the tables: [`src/ratch/kg/README.md`](../src/ratch/kg/README.md).

## The data model

```
(:Entity)-[:MENTIONS]->(:Chunk)          entity is spoken about in a ~30s clip
(:Entity)-[:RELATIONSHIP]->(:Entity)     LLM-extracted relation between entities
```

| `:Entity` property | meaning |
| --- | --- |
| `entity_id` | stable 16-hex id (`sha1(lowercased name)`) |
| `name` / `name_lower` | display name / lowercase for filtering |
| `entity_type` | `PERSON` · `ORG` · `GEO` · `EVENT` · `CONCEPT` · `WORK` · `OTHER` (the viewer's node colours; CONCEPT = ideas/policy/methods, WORK = reports/artifacts/datasets) |
| `mention_count` | number of distinct clips mentioning it |

| `:Chunk` property | meaning |
| --- | --- |
| `chunk_id` | `doc_id:speech_id:chunk_id` key |
| `doc_id` / `namn` | video id / video title |
| `start_s` / `end_s` | clip window in seconds |
| `text` | transcript excerpt (first 280 chars) |

`RELATIONSHIP` edges carry a `description` (LLM's phrasing, truncated 160 chars
at a word boundary) and a `weight` = how many distinct clips support the relation
(repeated source→target pairs are collapsed into one weighted edge).

## Using the `/graph` page

- **Search** an entity top-left, or **click any node** → side panel with its
  clips (click one to jump into the player at that timestamp), related
  entities, and co-occurring entities.
- **Cypher presets** dropdown = ready-made queries; the **REPL** below it runs
  anything you type (Ctrl-Enter). Results render in **Graph / Table / JSON**
  tabs.
- The `?` button explains how to read the view; node colour = entity type,
  node size = mentions.

## Cypher cookbook (all verified against the live engine)

Every clip where an entity is mentioned:

```cypher
MATCH (a:Entity)-[:MENTIONS]->(c:Chunk)
WHERE a.name_lower = 'stockholm'
RETURN c.namn, c.start_s, c.text LIMIT 25
```

Fuzzy entity lookup (substring):

```cypher
MATCH (a:Entity) WHERE a.name_lower CONTAINS 'miljö'
RETURN a.name, a.entity_type, a.mention_count LIMIT 20
```

Most-mentioned people:

```cypher
MATCH (a:Entity) WHERE a.entity_type = 'PERSON'
RETURN a.name, a.mention_count ORDER BY a.mention_count DESC LIMIT 20
```

Who is related to an entity, and how (edge property access):

```cypher
MATCH (a:Entity)-[r:RELATIONSHIP]->(b:Entity)
WHERE a.name_lower = 'stockholm'
RETURN b.name, r.description LIMIT 25
```

Entities that share clips (co-occurrence):

```cypher
MATCH (a:Entity)-[:MENTIONS]->(c:Chunk)<-[:MENTIONS]-(b:Entity)
WHERE a.entity_id < b.entity_id
RETURN a.name, b.name, count(c.chunk_id) AS shared ORDER BY shared DESC LIMIT 15
```

Timeline of one video — which entities come up, in order:

```cypher
MATCH (a:Entity)-[:MENTIONS]->(c:Chunk)
WHERE c.doc_id = '<doc_id>'
RETURN a.name, c.start_s ORDER BY c.start_s LIMIT 50
```

Entity-type breakdown of the whole graph:

```cypher
MATCH (a:Entity)
RETURN a.entity_type, count(a.entity_id) AS n ORDER BY n DESC LIMIT 10
```

## Engine rules & limits

- **Read-only** — `CREATE` / `SET` / `DELETE` / `MERGE` are rejected by the
  engine; one statement per request.
- Supported and verified: `MATCH` patterns (incl. multi-hop and reversed
  arrows), `WHERE` with `=`, `<`, `CONTAINS`, edge variables + properties,
  `RETURN` with aliases, `count()`, `ORDER BY`, `LIMIT`.
- A `LIMIT` is appended automatically when missing (cap 1000). Invalid Cypher
  returns HTTP 200 with an `error` field — the REPL shows it inline.

## API (for scripts)

| endpoint | purpose |
| --- | --- |
| `GET /api/graph/status` | `{built, entities, relations, mentions, videos}` |
| `POST /api/graph/cypher` `{query, limit}` | run Cypher → `{columns, rows}` |
| `GET /api/graph/search?q=` | substring entity search (not Cypher) |
| `GET /api/graph/entity/{id}` | clips + neighbours + co-occurrence for one entity |
| `GET /api/graph/subgraph?entity_id=&limit=` | nodes/edges for the canvas view |

[lance-graph]: https://github.com/lancedb/lance-graph
