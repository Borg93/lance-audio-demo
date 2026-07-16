"""P1.7 parity: old-pipeline baseline DB vs new Ray-pipeline DB, computed from BOTH.

Nothing is hardcoded from either side: every number below is read from the two
DBs at run time and compared —

* identical table sets (``*.lance`` dirs, ``__manifest``/staging excluded);
* per-table row-count pairs and identical ``doc_id`` sets;
* key uniqueness on both sides for every table's declared key;
* blob-derived inputs are BYTE-IDENTICAL: every frame JPEG's sha1 must match
  across DBs (stronger than any embedding comparison);
* per-embedding-column min cosine across rows joined by key: >= 0.999 for
  text/voice; >= 0.995 for the image path (``frame_embedding``) — measured
  vLLM self-jitter on IDENTICAL bytes is ~0.9997 back-to-back but
  cross-session bf16 continuous-batching variance exceeds 1e-3 through the
  vision tower, with input identity proven separately by the byte check;
* GENERATIVE columns (``caption``) are compared as process, not tokens: the
  caption server is not self-deterministic even at temperature 0 (measured:
  same server, same bytes, back-to-back → 4/6 exact match), so the check is
  full population on both sides + the reported exact-match rate; the
  ``caption_embedding`` cosine (>= 0.999) is then evaluated ONLY over rows
  whose caption strings match exactly — isolating the embedding wiring from
  generative token flips;
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
#: The image path tolerates cross-session vLLM bf16 batching variance; input
#: identity is proven separately by the frame-byte sha1 comparison.
MIN_COSINE_BY_COLUMN = {"frame_embedding": 0.995}
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


def compare_blob_hashes(old_db: Path, new_db: Path, table: str, keys: list[str], column: str) -> int:
    """Rows whose blob payload sha1 matches across DBs, joined by key."""
    import hashlib

    def hashes(db: Path) -> dict[tuple, str]:
        ds = lance.dataset(str(db / f"{table}.lance"))
        data = ds.to_table(columns=keys, with_row_id=True)
        key_tuples = list(zip(*(data[k].to_pylist() for k in keys), strict=True))
        out: dict[tuple, str] = {}
        for key, blob in zip(key_tuples, ds.take_blobs(column, ids=data["_rowid"].to_pylist()), strict=True):
            with blob as handle:
                out[key] = hashlib.sha1(handle.read()).hexdigest()
        return out

    old_hashes, new_hashes = hashes(old_db), hashes(new_db)
    return sum(1 for k, v in old_hashes.items() if new_hashes.get(k) == v)


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

        caption_match_keys: set[tuple] | None = None
        if name == "chunk_frames":
            old_caps = keyed(old_ds, keys, ["caption"])
            new_caps = keyed(new_ds, keys, ["caption"])
            check(all(v[0] is not None for v in old_caps.values()), "OLD captions have NULLs")
            check(all(v[0] is not None for v in new_caps.values()), "NEW captions have NULLs")
            caption_match_keys = {k for k, v in old_caps.items() if new_caps.get(k) == v}
            print(
                f"  chunk_frames.caption: fully populated both sides; {len(caption_match_keys)}/{n_old} "
                "exact-match (informational — the server is not self-deterministic at temp 0)"
            )

        for i, column in enumerate(embeddings):
            if old_map.keys() != new_map.keys():
                continue
            old_cmp, new_cmp = old_map, new_map
            scope = ""
            if column == "caption_embedding" and caption_match_keys is not None:
                old_cmp = {k: v for k, v in old_map.items() if k in caption_match_keys}
                new_cmp = {k: v for k, v in new_map.items() if k in caption_match_keys}
                scope = f" over {len(old_cmp)} caption-matching row(s)"
            worst = min_cosine(old_cmp, new_cmp, i)
            floor = MIN_COSINE_BY_COLUMN.get(column, MIN_COSINE)
            print(f"  {name}.{column}: min cosine = {worst:.6f} (floor {floor}){scope}")
            check(worst >= floor, f"{name}.{column}: min cosine {worst:.6f} < {floor}")

        if name == "chunk_frames":
            same = compare_blob_hashes(old_db, new_db, name, keys, "frame_blob")
            print(f"  chunk_frames.frame_blob: {same}/{n_old} JPEGs byte-identical (sha1)")
            check(same == n_old, "chunk_frames.frame_blob: payload bytes differ")

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
