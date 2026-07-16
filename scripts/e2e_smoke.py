"""End-to-end smoke test of the whole raudio pipeline on a tiny real sample.

Drives the actual CLI through every stage — ingest → embed-chunks →
extract-chunk-frames → embed-chunk-frames — into a throwaway temp DB (your real
``transcripts.lance`` is never touched), then boots the FastAPI backend and
asserts every endpoint and search mode.

Prereqs: the vLLM embed server on :8001 (``make embed-server-docker``) and a few
files that have BOTH an alignment JSON and a source MP4 (``input/sv-test`` here).

Run:
    uv run --extra multimodal python scripts/e2e_smoke.py
    uv run --extra multimodal python scripts/e2e_smoke.py --docs 3 --frame-limit 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import lance

ALIGN_DIR = Path("output/sv-test/alignments")
AUDIO_ROOT = Path("input/sv-test")


def _run(*args: str) -> None:
    """Invoke the raudio CLI in-process-venv; abort the smoke on non-zero exit."""
    cmd = [sys.executable, "-m", "rmedia.cli", *args]
    print(f"\n$ raudio {' '.join(args)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"stage failed (exit {result.returncode}): raudio {' '.join(args)}")


def _pick_stems(n: int) -> list[str]:
    aligns = {p.stem: p for p in ALIGN_DIR.glob("*.json")}
    mp4s = {p.stem for p in AUDIO_ROOT.glob("*.mp4")}
    both = sorted(set(aligns) & mp4s)
    if len(both) < n:
        raise SystemExit(
            f"need {n} files with alignment+mp4, found {len(both)} in {ALIGN_DIR}/{AUDIO_ROOT}"
        )
    return both[:n]


def _assert(label: str, ok: bool, detail: str = "") -> None:
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}{f'  — {detail}' if detail else ''}")
    if not ok:
        raise SystemExit(f"ASSERTION FAILED: {label} {detail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2, help="number of sample documents")
    ap.add_argument("--frame-limit", type=int, default=24, help="cap frames extracted (speed)")
    args = ap.parse_args()

    stems = _pick_stems(args.docs)
    json_paths = [str(ALIGN_DIR / f"{s}.json") for s in stems]
    print(f"sample ({len(stems)}): {', '.join(stems)}")

    tmp = Path(tempfile.mkdtemp(prefix="raudio_e2e_"))
    db = tmp / "transcripts.lance"
    thumbs = tmp / "thumbnails"

    # ── 1) thumbnails (for the /api/thumbnail endpoint) ──────────────────────
    _run("thumbnail", "--input-dir", str(AUDIO_ROOT), "--output-dir", str(thumbs))

    # ── 2) ingest → chunks + documents (media URIs + thumbnails) ─────────────
    _run(
        "--db",
        str(db),
        "ingest",
        *json_paths,
        "--audio-root",
        str(AUDIO_ROOT),
        "--thumbnail-dir",
        str(thumbs),
        "--doc-language",
        "sv",
        "--fts-language",
        "Swedish",
    )

    # ── 3) embed chunk text (flat — tiny sample, no IVF index) ───────────────
    _run("--db", str(db), "feature", "text_embedding", "--no-create-index")

    # ── 4) extract per-chunk frames → chunk_frames ───────────────────────────
    _run(
        "--db",
        str(db),
        "extract-chunk-frames",
        "--audio-root",
        str(AUDIO_ROOT),
        "--limit",
        str(args.frame_limit),
    )

    # ── 5) embed frames → frame_embedding ────────────────────────────────────
    _run("--db", str(db), "feature", "frame_embedding", "--no-create-index")

    # ── 6) backend: assert every endpoint + search mode ──────────────────────
    print("\n── backend (FastAPI TestClient) ──")
    from backend import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app(db))

    # a query string guaranteed to hit: the first chunk's own text
    chunks = lance.dataset(str(db / "chunks.lance"))
    first_text = chunks.to_table(columns=["text"], limit=1).column("text")[0].as_py()
    query = first_text[:60].strip() or "regeringen"

    health = client.get("/api/health").json()
    _assert("health 200 + chunks>0", health["db"]["chunks"] > 0, f"chunks={health['db']['chunks']}")
    _assert("health embed reachable", health["embed"]["ok"] is True)

    docs = client.get("/api/documents", params={"per_page": 5}).json()
    _assert("documents listed", docs["total"] >= len(stems), f"total={docs['total']}")
    doc_id = docs["docs"][0]["doc_id"]

    for mode in ("fts", "semantic", "hybrid", "all"):
        r = client.get("/api/search", params={"q": query, "mode": mode, "n": 5})
        hits = r.json()
        _assert(f"search mode={mode} 200", r.status_code == 200, f"{r.status_code}")
        _assert(
            f"search mode={mode} returns hits",
            isinstance(hits, list) and len(hits) > 0,
            f"{len(hits)} hits",
        )

    r = client.get("/api/search", params={"q": query, "mode": "visual", "n": 5})
    vhits = r.json()
    _assert(
        "search mode=visual returns hits (frame→chunk join)",
        r.status_code == 200 and len(vhits) > 0,
        f"{r.status_code}, {len(vhits)} hits",
    )

    r = client.get(f"/api/thumbnail/{doc_id}")
    _assert("thumbnail 200/404", r.status_code in (200, 404), f"{r.status_code}")

    r = client.get(f"/api/media/{doc_id}", headers={"Range": "bytes=0-1023"})
    _assert("media Range → 206", r.status_code == 206, f"{r.status_code}")
    _assert(
        "media Content-Range header", r.headers.get("Content-Range", "").startswith("bytes 0-1023/")
    )

    frames = lance.dataset(str(db / "chunk_frames.lance")).to_table(
        columns=["doc_id", "speech_id", "chunk_id"], limit=1
    )
    fd = frames.column("doc_id")[0].as_py()
    fs = frames.column("speech_id")[0].as_py()
    fc = frames.column("chunk_id")[0].as_py()
    r = client.get(f"/api/chunk-frame/{fd}/{fs}/{fc}")
    _assert("chunk-frame 200", r.status_code == 200, f"{r.status_code}")

    print(f"\n✅ end-to-end smoke PASSED — temp DB at {db}")
    print("   (delete it with:  rm -rf", str(tmp), ")")


if __name__ == "__main__":
    main()
