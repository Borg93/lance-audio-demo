#!/usr/bin/env bash
# LANCE_MEDIA_MERGE §5.3 — capture the OLD-pipeline baseline. Run BEFORE the
# raudio→rmedia rename (or from a pinned pre-rename worktree).
#
# Ingests the 5 fixed sample docs (scripts/sample_docs.txt) into a fresh
# baselines/old_pipeline.lance and runs the deterministic feature stages the
# parity check (P1.7) compares: ingest, text/frame embeddings, chunk frames,
# speaker turns + voiceprints + speakers, captions + caption embeddings.
#
# EXCLUDED (documented in docs/LANCE_MEDIA_MERGE.md §5.3): atlas projections
# (the EVoC fit is unseeded → not reproducible run-to-run) and topics (depends
# on the atlas map). Parity for those columns is asserted structurally, not
# value-for-value.
#
# Requires: embed server (EMBED_PORT, default 8011 — :8001 is squatted by the
# lance-ns lineage-api container), caption VLM on :8003 (make caption-server),
# and the 5 sample mp4s restored into input/sv/.
set -euo pipefail
cd "$(dirname "$0")/.."

DB=${BASELINE_DB:-baselines/old_pipeline.lance}
EMBED_PORT=${EMBED_PORT:-8011}
EMBED_URL=${EMBED_URL:-http://127.0.0.1:$EMBED_PORT}
CAPTION_MODEL=${CAPTION_SERVE_MODEL:-Qwen/Qwen3-VL-2B-Instruct}

if [ -e "$DB" ]; then
  echo "refusing to overwrite existing $DB — remove it first for a fresh capture" >&2
  exit 1
fi
mkdir -p baselines

mapfile -t JSONS < <(awk '!/^#/ && NF {sub(/\.mp4$/, "", $2); print "output/sv/alignments/" $2 ".json"}' scripts/sample_docs.txt)
echo "── ingest ${#JSONS[@]} sample docs → $DB"
uv run raudio --db "$DB" ingest "${JSONS[@]}" \
  --audio-root input/sv --metadata-csv video_batcher.csv --thumbnail-dir thumbnails

echo "── text embeddings ($EMBED_URL)"
make embed-chunks DB="$DB" EMBED_URL="$EMBED_URL"

echo "── chunk frames (ffmpeg)"
make extract-chunk-frames DB="$DB" AUDIO_DIR=input/sv

echo "── frame embeddings"
make embed-chunk-frames DB="$DB" EMBED_URL="$EMBED_URL"

echo "── speaker turns → voiceprints → speakers"
make speaker-turns DB="$DB" AUDIO_DIR=input/sv
make embed-speaker-turns DB="$DB" AUDIO_DIR=input/sv
make build-speakers DB="$DB"

echo "── captions + caption embeddings (model $CAPTION_MODEL)"
make captions DB="$DB" EMBED_URL="$EMBED_URL" CAPTION_MODEL="$CAPTION_MODEL"

echo "── record FTS baseline"
uv run python scripts/record_fts_baseline.py --db "$DB" --out baselines/fts_baseline.json

echo "BASELINE OK  ($DB)"
