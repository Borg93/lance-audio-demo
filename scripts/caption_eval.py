"""Evaluation harness: caption a diverse sample of REAL frames and emit a gallery.

Reads existing frames from chunk_frames (no extraction, no DB writes), captions
them with the live Gemma client, and writes a self-contained HTML page
(frame + Swedish caption side by side) plus a JSONL of the pairs. Lets you judge
caption quality before committing to the full 145k-frame batch.

    uv run --extra multimodal python scripts/caption_eval.py --n 40
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import lance

from raudio.vllm.caption import CAPTION_INSTRUCTION, CAPTION_MODEL, DEFAULT_CAPTION_URL, VLLMCaptionClient

OUT_DIR = Path("/tmp/raudio_caption_eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="transcripts_v2.lance", help="Lance DB dir")
    ap.add_argument("--n", type=int, default=40, help="sample size (spread across the corpus)")
    args = ap.parse_args()

    ds = lance.dataset(str(Path(args.db) / "chunk_frames.lance"))
    total = ds.count_rows()
    keyed = ds.to_table(columns=["doc_id", "speech_id", "chunk_id"], with_row_id=True)
    row_ids = keyed.column("_rowid").to_pylist()
    docs = keyed.column("doc_id").to_pylist()

    # Evenly-spaced offsets → frames from many different documents.
    step = max(1, total // args.n)
    picks = list(range(0, total, step))[: args.n]

    jpegs: list[bytes] = []
    meta: list[dict] = []
    for i in picks:
        rid = row_ids[i]
        blob = ds.take_blobs("frame_blob", ids=[rid])[0]
        with blob as f:
            jpegs.append(f.read())
        meta.append({"row": i, "doc_id": docs[i]})

    print(f"captioning {len(jpegs)} frames via {CAPTION_MODEL} @ {DEFAULT_CAPTION_URL} …")
    client = VLLMCaptionClient()
    captions = client.caption(jpegs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_html = []
    pairs = []
    for m, jpg, cap in zip(meta, jpegs, captions, strict=True):
        b64 = base64.b64encode(jpg).decode("ascii")
        rows_html.append(
            f'<figure><img src="data:image/jpeg;base64,{b64}"/>'
            f'<figcaption><b>{cap}</b><br><span class="meta">row {m["row"]} · '
            f'doc {m["doc_id"][:10]}…</span></figcaption></figure>'
        )
        pairs.append({**m, "caption": cap})

    html = (
        "<!doctype html><meta charset='utf-8'><title>raudio caption eval</title>"
        "<style>body{background:#111;color:#eee;font:14px system-ui;margin:24px}"
        "h1{font-size:18px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}"
        "figure{margin:0;background:#1c1c1c;border:1px solid #333;border-radius:8px;overflow:hidden}"
        "img{width:100%;display:block;background:#000}figcaption{padding:10px;line-height:1.4}"
        ".meta{color:#888;font-size:11px}</style>"
        f"<h1>raudio — Gemma 4 Swedish captions ({len(pairs)} sampled frames of {total})</h1>"
        f"<p class='meta'>model={CAPTION_MODEL} · prompt={CAPTION_INSTRUCTION}</p>"
        f"<div class='grid'>{''.join(rows_html)}</div>"
    )
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    (OUT_DIR / "captions.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR/'index.html'}  (open in a browser)")
    print(f"wrote {OUT_DIR/'captions.jsonl'}\n")
    for p in pairs[:12]:
        print(f"  [{p['row']:>6}] {p['caption']}")


if __name__ == "__main__":
    main()
