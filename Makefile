# ─── raudio dev helpers ────────────────────────────────────────────────
# All paths are relative to the project root. Override on the CLI:
#   make search Q='spring of hope'
#   make demo AUDIO_ROOT=./examples

DB       ?= ./transcripts.lance
TABLE    ?= chunks
SAMPLE   ?= examples/taleoftwocities_01_dickens_64kb_trimmed.json

# ─── Corpus defaults (override on CLI: make pipeline LANGUAGE=en) ───────────
AUDIO_DIR       ?= ./input
OUTPUT_ROOT     ?= ./output/$(LANGUAGE)
ALIGNMENTS      ?= $(OUTPUT_ROOT)/alignments
THUMBNAILS_DIR  ?= ./thumbnails
METADATA_CSV    ?= ./video_batcher.csv
HF_BUCKET       ?=                       # e.g. gborg/raudio-demo
GPU             ?= 0

.PHONY: help check-deps bootstrap install lock \
	transcribe detect-language thumbnail \
	ingest ingest-with-media ingest-full reindex-fts \
	pipeline pipeline-sharded \
	backend frontend labeler dev \
	hf-upload-db hf-upload-videos hf-upload-all hf-download-db hf-download-all \
	reingest search query demo shell clean clean-db clean-run reset download

help:                 ## Show this help (default).
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ─── Bootstrap / prereqs ────────────────────────────────────────────────────
check-deps:           ## Verify system prereqs (uv, ffmpeg) and print install hints.
	@echo "── Checking system dependencies ────────────────────────────────"
	@command -v uv >/dev/null 2>&1 && echo "  ✓ uv       ($$(uv --version | head -1))" \
		|| (echo "  ✗ uv        — install:  curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1)
	@command -v ffmpeg >/dev/null 2>&1 && echo "  ✓ ffmpeg   ($$(ffmpeg -version | head -1 | awk '{print $$1, $$2, $$3}'))" \
		|| echo "  ⚠  ffmpeg  — required for `make transcribe` and `make ingest-with-audio`.\n              Install:  sudo apt install ffmpeg   (Linux)\n                        brew install ffmpeg         (macOS / linuxbrew)"
	@command -v hf >/dev/null 2>&1 && echo "  ✓ hf CLI   ($$(hf --version 2>/dev/null | head -1))" \
		|| echo "  ⚠  hf CLI  — only needed for pyannote VAD. `pip install huggingface_hub[cli]`\n              then `hf auth login`, or pass VAD=silero."
	@nvidia-smi >/dev/null 2>&1 && echo "  ✓ NVIDIA GPU ($$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" \
		|| echo "  ⚠  no NVIDIA GPU detected — kb-whisper-large will be very slow on CPU.\n              Use smaller models (kb-whisper-base, whisper-base) or pass DEVICE=cpu."
	@echo "─────────────────────────────────────────────────────────────────"

bootstrap: check-deps install  ## One-shot: verify system + install all Python deps.
	@echo ""
	@echo "✓ Bootstrap complete."
	@echo ""
	@echo "Next steps:"
	@echo "  1. (optional) hf auth login                       # only if using pyannote VAD"
	@echo "  2. make transcribe AUDIO_DIR=/path/to/media        # produces output/alignments/*.json"
	@echo "  3. make reingest SAMPLE=output/alignments/*.json   # builds ./transcripts.lance"
	@echo "  4. make search Q='your query'                      # sanity check"
	@echo ""
	@echo "Or try the bundled sample:"
	@echo "  make demo"

# ─── Environment ────────────────────────────────────────────────────────────
install:              ## Create a .venv and install all runtime deps via uv.
	uv sync

lock:                 ## Re-resolve dependencies and update uv.lock.
	uv lock

# ─── Transcribe (produces the JSON files that `ingest` consumes) ────────────
# Requires ffmpeg + a working HF token (for pyannote VAD).
# GPU strongly recommended for kb-whisper-large.
# Emits one JSON per audio file to $(OUTPUT_ROOT)/alignments/.
#   make transcribe LANGUAGE=en
#   make transcribe AUDIO_DIR=/path/to/sv-audio LANGUAGE=sv
LANGUAGE  ?= sv
MODEL     ?= KBLab/kb-whisper-large
VAD       ?= pyannote
DEVICE    ?= cuda
transcribe:           ## Run easytranscriber → writes output/alignments/*.json.
	uv run raudio transcribe \
		--audio-dir $(AUDIO_DIR) \
		--language  $(LANGUAGE) \
		--model     $(MODEL) \
		--vad       $(VAD) \
		--device    $(DEVICE)

detect-language:      ## Sort $(AUDIO_DIR)/*.{mp4,wav,…} into $(AUDIO_DIR)/<lang>/ subfolders.
	uv run raudio detect-language --audio-dir $(AUDIO_DIR) --device $(DEVICE)

# ─── Download from the Riksarkivet CSV (resumable, concurrent) ──────────────
download:             ## Bulk-download videos listed in $(METADATA_CSV) → $(AUDIO_DIR).
	uv run raudio download --csv $(METADATA_CSV) --output-dir $(AUDIO_DIR) --concurrency 6

# ─── Thumbnails (ffmpeg frame extraction) ───────────────────────────────────
thumbnail:            ## Render one JPEG per video under $(AUDIO_DIR) → $(THUMBNAILS_DIR).
	uv run raudio thumbnail --input-dir $(AUDIO_DIR) --output-dir $(THUMBNAILS_DIR)

# ─── Ingest ─────────────────────────────────────────────────────────────────
ingest:               ## Ingest $(SAMPLE) into $(DB) (append).
	uv run raudio --db $(DB) --table $(TABLE) ingest $(SAMPLE)

reingest: clean-db ingest  ## Nuke the DB first, then ingest — safe to re-run.

# Embed the raw source media (mp3/mp4/wav/…) as a Lance blob-v2 column.
# Adds a second `documents` table alongside `chunks` — one row per source
# file, bytes loaded lazily via `ds.take_blobs("media_blob", ...)`.
#   make ingest-with-media AUDIO_ROOT=./examples
ingest-with-media:    ## Ingest $(SAMPLE) AND embed source media in the `documents` table.
	@test -n "$(AUDIO_ROOT)" || (echo "Set AUDIO_ROOT=/path/to/media-dir"; exit 2)
	uv run raudio --db $(DB) --table $(TABLE) ingest --audio-root $(AUDIO_ROOT) $(SAMPLE)

# ─── Full-corpus ingest (metadata CSV + thumbnails + Swedish FTS) ───────────
# Every row's media_uri is set from --audio-root (→ file:///abs/path/...mp4)
# by default. Override with MEDIA_BASE_URI for remote storage:
#   make ingest-full MEDIA_BASE_URI=hf://buckets/you/raudio-videos/
#   make ingest-full MEDIA_BASE_URI=s3://bucket/videos/
MEDIA_BASE_URI ?=
ingest-full:          ## Ingest all $(ALIGNMENTS)/*.json with metadata + thumbnails + Swedish FTS.
	@test -d "$(ALIGNMENTS)" || (echo "$(ALIGNMENTS) does not exist — run `make transcribe` first."; exit 2)
	@test -f "$(METADATA_CSV)" || (echo "$(METADATA_CSV) not found"; exit 2)
	uv run raudio --db $(DB) --table $(TABLE) ingest \
		$(ALIGNMENTS)/*.json \
		--metadata-csv $(METADATA_CSV) \
		$(if $(wildcard $(THUMBNAILS_DIR)),--thumbnail-dir $(THUMBNAILS_DIR),) \
		$(if $(MEDIA_BASE_URI),--media-base-uri $(MEDIA_BASE_URI),--audio-root $(AUDIO_DIR)) \
		--fts-language Swedish

# ─── Rebuild only the FTS index (no re-ingest) ──────────────────────────────
FTS_LANGUAGE ?= Swedish
reindex-fts:          ## Rebuild the FTS index with a different language/config (fast).
	uv run raudio --db $(DB) --table $(TABLE) reindex-fts --language $(FTS_LANGUAGE)

# ─── Full pipeline on a single GPU ──────────────────────────────────────────
# Steps: transcribe → thumbnail → ingest-full. Assumes $(AUDIO_DIR) already
# populated (via `make download` or manual copies). Forces $(LANGUAGE) with
# silero VAD for reproducibility — override to use pyannote.
pipeline:             ## Transcribe + thumbnail + ingest for $(LANGUAGE) on GPU $(GPU).
	@echo "── 1/3 transcribe ──────────────────────────────────────────────"
	CUDA_VISIBLE_DEVICES=$(GPU) uv run raudio transcribe \
		--audio-dir $(AUDIO_DIR) --language $(LANGUAGE) --vad $(VAD) \
		--output-root $(OUTPUT_ROOT)
	@echo "── 2/3 thumbnails ──────────────────────────────────────────────"
	$(MAKE) thumbnail
	@echo "── 3/3 ingest ──────────────────────────────────────────────────"
	$(MAKE) ingest-full

# ─── Sharded pipeline across 3 GPUs ────────────────────────────────────────
# Pre-split $(AUDIO_DIR) into shard subdirs, then run transcribe on each in
# parallel. You still need to run thumbnail + ingest-full afterwards.
#   make pipeline-sharded                       (splits automatically)
#   make pipeline-sharded SHARDS=4              (4-way split, 4 GPUs)
SHARDS ?= 3
shards:               ## Split $(AUDIO_DIR)/*.mp4 round-robin into $(AUDIO_DIR)/shard{0..}.
	@for i in $$(seq 0 $$(($(SHARDS)-1))); do mkdir -p $(AUDIO_DIR)/shard$$i; done
	@cd $(AUDIO_DIR) && ls *.mp4 2>/dev/null | awk -v n=$(SHARDS) '{print ((NR-1)%n)"\t"$$0}' \
		| while IFS=$$'\t' read s f; do mv "$$f" "shard$$s/"; done
	@for i in $$(seq 0 $$(($(SHARDS)-1))); do \
		printf "  shard%s: %s files\n" "$$i" "$$(ls $(AUDIO_DIR)/shard$$i/*.mp4 2>/dev/null | wc -l)"; \
	done

pipeline-sharded: shards  ## Transcribe all $(SHARDS) shards in parallel (one GPU each).
	@echo "Launching $(SHARDS) transcribe processes (one per GPU)."
	@echo "Watch progress with: tail -f $(OUTPUT_ROOT)/shard{0..$$(($(SHARDS)-1))}.log"
	@for i in $$(seq 0 $$(($(SHARDS)-1))); do \
		CUDA_VISIBLE_DEVICES=$$i nohup uv run raudio transcribe \
			--audio-dir $(AUDIO_DIR)/shard$$i --language $(LANGUAGE) --vad $(VAD) \
			--output-root $(OUTPUT_ROOT) \
			> $(OUTPUT_ROOT)/shard$$i.log 2>&1 & \
		echo "  started GPU $$i (pid $$!) → $(OUTPUT_ROOT)/shard$$i.log"; \
	done
	@echo "All shards running in background. Run 'make thumbnail && make ingest-full' when done."

# ─── HuggingFace bucket sync ───────────────────────────────────────────────
# Requires `hf auth login` first. Buckets are the s3-like storage layer:
#   https://huggingface.co/docs/huggingface_hub/main/en/guides/buckets
# Set HF_BUCKET=your-namespace/your-bucket.
# ─── Viewer: Python backend + Bun frontend ──────────────────────────────────
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 3000

backend:              ## Run the FastAPI backend (Lance reads, /api/*).
	uv run raudio --db $(DB) serve --host $(BACKEND_HOST) --port $(BACKEND_PORT)

frontend:             ## Run the Bun frontend (static + /api/* → backend).
	cd frontend && bun install --silent && bun run server.ts --api http://$(BACKEND_HOST):$(BACKEND_PORT) --port $(FRONTEND_PORT)

LABELER_PORT    ?= 3999
labeler:              ## Run the manual language relabeler on $(AUDIO_DIR) (port $(LABELER_PORT)).
	cd tools/labeler && bun install --silent && bun run labeler.ts --root ../../$(AUDIO_DIR) --port $(LABELER_PORT)

dev:                  ## Run backend + frontend together (tmux or two terminals).
	@echo "Run these in two terminals:"
	@echo "  1) make backend"
	@echo "  2) make frontend"
	@echo "Then open http://localhost:$(FRONTEND_PORT)"

hf-upload-db:         ## Sync $(DB)/ to hf://buckets/$(HF_BUCKET)/transcripts.lance/
	@test -n "$(HF_BUCKET)" || (echo "Set HF_BUCKET=namespace/bucket"; exit 2)
	hf buckets sync $(DB) hf://buckets/$(HF_BUCKET)/transcripts.lance --delete --verbose

hf-upload-videos:     ## Sync $(AUDIO_DIR)/ to hf://buckets/$(HF_BUCKET)/videos/ (split pattern).
	@test -n "$(HF_BUCKET)" || (echo "Set HF_BUCKET=namespace/bucket"; exit 2)
	hf buckets sync $(AUDIO_DIR) hf://buckets/$(HF_BUCKET)/videos --include "*.mp4" --verbose

hf-upload-alignments: ## Sync ./output/ to hf://buckets/$(HF_BUCKET)/output/ (viewer needs these).
	@test -n "$(HF_BUCKET)" || (echo "Set HF_BUCKET=namespace/bucket"; exit 2)
	hf buckets sync ./output hf://buckets/$(HF_BUCKET)/output --include "*.json" --verbose

hf-upload-thumbnails: ## Sync $(THUMBNAILS_DIR)/ to hf://buckets/$(HF_BUCKET)/thumbnails/
	@test -n "$(HF_BUCKET)" || (echo "Set HF_BUCKET=namespace/bucket"; exit 2)
	hf buckets sync $(THUMBNAILS_DIR) hf://buckets/$(HF_BUCKET)/thumbnails --verbose

hf-upload-all: hf-upload-db hf-upload-videos hf-upload-alignments hf-upload-thumbnails  ## Push everything a viewer needs.

hf-download-db:       ## Pull hf://buckets/$(HF_BUCKET)/transcripts.lance/ → $(DB)/
	@test -n "$(HF_BUCKET)" || (echo "Set HF_BUCKET=namespace/bucket"; exit 2)
	hf buckets sync hf://buckets/$(HF_BUCKET)/transcripts.lance $(DB) --verbose

hf-download-all:      ## Pull Lance + videos + alignments + thumbnails to local.
	@test -n "$(HF_BUCKET)" || (echo "Set HF_BUCKET=namespace/bucket"; exit 2)
	hf buckets sync hf://buckets/$(HF_BUCKET)/transcripts.lance $(DB) --verbose
	hf buckets sync hf://buckets/$(HF_BUCKET)/videos $(AUDIO_DIR) --verbose
	hf buckets sync hf://buckets/$(HF_BUCKET)/output ./output --verbose
	hf buckets sync hf://buckets/$(HF_BUCKET)/thumbnails $(THUMBNAILS_DIR) --verbose

# ─── Search ─────────────────────────────────────────────────────────────────
search:               ## FTS query. Usage: make search Q='best of times'
	@test -n "$(Q)" || (echo "Set Q=...  (e.g. make search Q='best of times')"; exit 2)
	uv run raudio --db $(DB) --table $(TABLE) search '$(Q)'

query: search         ## Alias for `make search`.

# ─── One-shot demo ──────────────────────────────────────────────────────────
# Wipes the DB, ingests the sample, then runs three representative queries so
# you can eyeball the pipeline end-to-end. This is the `make` command that
# exercises everything else.
demo: install reingest  ## Full end-to-end smoke: install + reingest + 3 queries.
	@echo ""
	@echo "── query 1 ── 'best of times' ──────────────────────────────────"
	uv run raudio --db $(DB) --table $(TABLE) search 'best of times'
	@echo ""
	@echo "── query 2 ── exact phrase '\"spring of hope\"' ─────────────────"
	uv run raudio --db $(DB) --table $(TABLE) search '"spring of hope"'
	@echo ""
	@echo "── query 3 ── boolean 'wisdom OR foolishness' ──────────────────"
	uv run raudio --db $(DB) --table $(TABLE) search 'wisdom OR foolishness' -n 3

# ─── REPL ───────────────────────────────────────────────────────────────────
shell:                ## Drop into a Python REPL with raudio imported as `la`.
	uv run python -ic "import raudio as la; print('raudio loaded as `la`')"

# ─── Cleanup ────────────────────────────────────────────────────────────────
clean-db:             ## Delete the Lance database only.
	rm -rf $(DB)

clean-run: clean-db   ## Delete the Lance DB + output/ so the next pipeline is fresh.
	rm -rf ./output ./thumbnails

clean: clean-run      ## clean-run + build artefacts.
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

reset: clean          ## clean + delete .venv.
	rm -rf .venv
