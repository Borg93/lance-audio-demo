"""Record the FTS top-10 key sets for the fixed parity queries (P1.7).

Runs the three fixed Swedish queries against a DB's ``chunks`` FTS index and
writes the ranked ``doc_id|speech_id|chunk_id`` keys per query. The parity
check runs this same recorder against the old-pipeline baseline DB and the
new-pipeline DB and compares the outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lancedb
from pydantic import BaseModel

PARITY_QUERIES = ["regeringen", "myndighet", "Sverige"]
TOP_K = 10


class FtsBaseline(BaseModel):
    db: str
    queries: dict[str, list[str]]


def fts_top_keys(table: lancedb.table.Table, query: str, k: int = TOP_K) -> list[str]:
    rows = (
        table.search(query, query_type="fts")
        .select(["doc_id", "speech_id", "chunk_id"])
        .limit(k)
        .to_list()
    )
    return [f"{r['doc_id']}|{r['speech_id']}|{r['chunk_id']}" for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Lance DB directory")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    table = lancedb.connect(args.db).open_table("chunks")
    baseline = FtsBaseline(
        db=args.db,
        queries={q: fts_top_keys(table, q) for q in PARITY_QUERIES},
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(baseline.model_dump_json(indent=2) + "\n")
    for q, keys in baseline.queries.items():
        print(f"  fts[{q!r}] top-{TOP_K}: {len(keys)} keys")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
