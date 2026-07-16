"""P2.6 search parity: replay the pre-rewrite goldens against the NEW backend.

``baselines/search_golden.json`` was recorded from the pre-split backend on
``transcripts_v2.lance`` (P1.0). This replays every query against a running
backend (the descriptor-driven one) and compares each ranked top-10 key list
as a SET, printing both sides per query, the DB path compared, then
``SEARCH PARITY OK`` only if every query matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def top_keys(client: httpx.Client, params: dict[str, object]) -> list[str]:
    resp = client.get("/api/search", params={**params, "n": 10})
    resp.raise_for_status()
    payload = resp.json()
    hits = payload["hits"] if isinstance(payload, dict) else payload
    return [f"{h['doc_id']}|{h['speech_id']}|{h['chunk_id']}" for h in hits]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--golden", default="baselines/search_golden.json")
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    print(f"replaying {len(golden['queries'])} golden queries")
    print(f"  golden recorded on DB={golden['db']} via {golden['backend']}")
    print(f"  replay target: {args.backend} (must serve the SAME DB)")

    failures = 0
    with httpx.Client(base_url=args.backend, timeout=120) as client:
        health = client.get("/api/health").json()
        print(f"  target db: {health.get('db', {}).get('path', '?')}")
        for record in golden["queries"]:
            params, expected = record["params"], record["top_keys"]
            got = top_keys(client, params)
            match = set(got) == set(expected)
            state = "same" if match else "DIFFER"
            print(f"  {params}: golden={len(expected)} new={len(got)} key sets {state}")
            if not match:
                failures += 1
                print(f"    golden-only: {sorted(set(expected) - set(got))[:3]}")
                print(f"    new-only:    {sorted(set(got) - set(expected))[:3]}")

    if failures:
        print(f"SEARCH PARITY FAILED ({failures} quer{'y' if failures == 1 else 'ies'} differ)")
        sys.exit(1)
    print("SEARCH PARITY OK")


if __name__ == "__main__":
    main()
