"""Record search goldens from the RUNNING pre-rewrite backend (P2.6 baseline).

Replays a fixed query set against ``GET /api/search`` on the old backend
(serving ``transcripts_v2.lance``) and stores each query's ranked top-10
``doc_id|speech_id|chunk_id`` keys. Phase 2's parity check replays the same
set against the descriptor-driven search service on the SAME DB and compares.

Run while the old backend is up:  uv run python scripts/record_search_golden.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
from pydantic import BaseModel

TOP_K = 10

# Mixed fixed set per docs/LANCE_MEDIA_MERGE.md §5.3: >=3 fts, >=3 vector
# (semantic), >=2 hybrid, >=2 filtered. Params mirror the SearchSpec query API.
GOLDEN_QUERIES: list[dict[str, str | int]] = [
    {"q": "statsministern", "mode": "fts"},
    {"q": "klimatförhandlingarna", "mode": "fts"},
    {"q": "polisen", "mode": "fts"},
    {"q": "diskussion om klimatpolitik", "mode": "semantic"},
    {"q": "presskonferens om terrorism", "mode": "semantic"},
    {"q": "ny generaldirektör utses", "mode": "semantic"},
    {"q": "regeringens budget", "mode": "hybrid"},
    {"q": "utrikespolitik", "mode": "hybrid"},
    {"q": "regeringen", "mode": "fts", "language": "sv"},
    {"q": "minister", "mode": "hybrid", "language": "sv"},
]


class SearchGolden(BaseModel):
    db: str
    backend: str
    queries: list[dict[str, object]]


def top_keys(client: httpx.Client, params: dict[str, str | int]) -> list[str]:
    resp = client.get("/api/search", params={**params, "n": TOP_K})
    resp.raise_for_status()
    payload = resp.json()
    hits = payload["hits"] if isinstance(payload, dict) else payload
    return [f"{h['doc_id']}|{h['speech_id']}|{h['chunk_id']}" for h in hits]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--db", default="transcripts_v2.lance")
    parser.add_argument("--out", default="baselines/search_golden.json")
    args = parser.parse_args()

    with httpx.Client(base_url=args.backend, timeout=120) as client:
        records = [
            {"params": params, "top_keys": top_keys(client, params)}
            for params in GOLDEN_QUERIES
        ]

    golden = SearchGolden(db=args.db, backend=args.backend, queries=records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(golden.model_dump_json(indent=2) + "\n")
    for record in records:
        params, keys = record["params"], record["top_keys"]
        print(f"  {params}: {len(keys)} keys")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
