"""P1.7 parity: old-pipeline baseline DB vs new Ray-pipeline DB, computed from BOTH.

Nothing is hardcoded from either side: every number below is read from the two
DBs at run time and compared —

* identical table sets (``*.lance`` dirs, ``__manifest``/staging excluded);
* per-table row-count pairs and identical ``doc_id`` sets;
* key uniqueness on both sides for every table's declared key;
* per-embedding-column min cosine similarity >= 0.999 across rows joined by key
  (absorbs server-side batching jitter; catches any real drift);
* the 3 fixed FTS queries return the same top-10 key SET on both DBs.

Prints the evidence lines, then ``PARITY OK`` only if every check passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lance
import lancedb
import numpy as np

#: table → (key columns, embedding columns)
TABLES: dict[str, tuple[list[str], list[str]]] = {
    "documents": (["doc_id"], []),
    "chunks": (["doc_id", "speech_id", "chunk_id"], ["text_embedding"]),
    "chunk_frames": (
        ["doc_id", "speech_id", "chunk_id", "frame_idx"],
        ["frame_embedding", "caption_embedding"],
    ),
    "speaker_turns": (["doc_id", "turn_id"], []),
    "speaker_embeddings": (["doc_id", "turn_id"], ["embedding"]),
    "speakers": (["doc_id", "speaker_label"], ["embedding"]),
}
FTS_QUERIES = ["regeringen", "myndighet", "Sverige"]
MIN_COSINE = 0.999
TOP_K = 10

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)
        print(f"  ✗ {message}")


def table_names(db: Path) -> set[str]:
    return {p.stem for p in db.glob("*.lance") if p.is_dir() and not p.stem.startswith("__")}


def keyed(table: lance.LanceDataset, keys: list[str], columns: list[str]) -> dict[tuple, list]:
    data = table.to_table(columns=[*keys, *columns])
    key_lists = [data[k].to_pylist() for k in keys]
    value_lists = [data[c].to_pylist() for c in columns]
    return {
        tuple(kl[i] for kl in key_lists): [vl[i] for vl in value_lists]
        for i in range(data.num_rows)
    }


def min_cosine(old: dict[tuple, list], new: dict[tuple, list], idx: int) -> float:
    worst = 1.0
    for key, old_values in old.items():
        a, b = old_values[idx], new[key][idx]
        if a is None or b is None:
            check(a is None and b is None, f"NULL mismatch at {key}")
            continue
        va, vb = np.asarray(a, np.float32), np.asarray(b, np.float32)
        cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
        worst = min(worst, cos)
    return worst


def fts_key_set(db: Path, query: str) -> frozenset[str]:
    tbl = lancedb.connect(str(db)).open_table("chunks")
    rows = (
        tbl.search(query, query_type="fts")
        .select(["doc_id", "speech_id", "chunk_id"])
        .limit(TOP_K)
        .to_list()
    )
    return frozenset(f"{r['doc_id']}|{r['speech_id']}|{r['chunk_id']}" for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", default="baselines/old_pipeline.lance")
    parser.add_argument("--new", default="parity_new.lance")
    args = parser.parse_args()
    old_db, new_db = Path(args.old), Path(args.new)
    print(f"comparing OLD={old_db}  NEW={new_db}")

    old_tables, new_tables = table_names(old_db), table_names(new_db)
    check(old_tables == new_tables, f"table sets differ: old-only={old_tables - new_tables} new-only={new_tables - old_tables}")
    print(f"  table set ({len(old_tables & new_tables)}): {sorted(old_tables & new_tables)}")

    for name, (keys, embeddings) in TABLES.items():
        if name not in old_tables or name not in new_tables:
            continue
        old_ds = lance.dataset(str(old_db / f"{name}.lance"))
        new_ds = lance.dataset(str(new_db / f"{name}.lance"))
        n_old, n_new = old_ds.count_rows(), new_ds.count_rows()
        print(f"  {name}: rows old={n_old} new={n_new}")
        check(n_old == n_new, f"{name}: row counts differ")

        old_map = keyed(old_ds, keys, embeddings)
        new_map = keyed(new_ds, keys, embeddings)
        check(len(old_map) == n_old, f"{name}: duplicate keys in OLD ({n_old - len(old_map)})")
        check(len(new_map) == n_new, f"{name}: duplicate keys in NEW ({n_new - len(new_map)})")
        check(old_map.keys() == new_map.keys(), f"{name}: key sets differ")

        if name == "documents":
            old_docs = {k[0] for k in old_map}
            new_docs = {k[0] for k in new_map}
            check(old_docs == new_docs, "doc_id sets differ")
            print(f"  documents: doc_id sets identical ({sorted(old_docs)})")

        for i, column in enumerate(embeddings):
            if old_map.keys() != new_map.keys():
                continue
            worst = min_cosine(old_map, new_map, i)
            print(f"  {name}.{column}: min cosine = {worst:.6f}")
            check(worst >= MIN_COSINE, f"{name}.{column}: min cosine {worst:.6f} < {MIN_COSINE}")

    for query in FTS_QUERIES:
        old_keys, new_keys = fts_key_set(old_db, query), fts_key_set(new_db, query)
        match = "same" if old_keys == new_keys else "DIFFER"
        print(f"  fts[{query!r}]: old={len(old_keys)} new={len(new_keys)} key sets {match}")
        check(old_keys == new_keys, f"fts[{query!r}]: top-{TOP_K} key sets differ")

    if failures:
        print(f"PARITY FAILED ({len(failures)} issue(s))")
        sys.exit(1)
    print("PARITY OK")


if __name__ == "__main__":
    main()
