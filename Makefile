# ─── raudio dev helpers ────────────────────────────────────────────────
# All paths are relative to the project root. Override on the CLI:
#   make search Q='spring of hope'
#   make demo AUDIO_ROOT=./examples

DB       ?= ./transcripts_v2.lance
LOG_DIR  ?= ./logs
TABLE    ?= chunks
SAMPLE   ?= examples/taleoftwocities_01_dickens_64kb_trimmed.json

# ─── Corpus defaults (override on CLI: make pipeline LANGUAGE=en) ───────────
# `:=` instead of `?=` for LANGUAGE/AUDIO_DIR because GNU make's `?=` respects
# environment variables, and `LANGUAGE` is a Linux locale env var (e.g.
# "en_US:en") that bleeds in and corrupts our paths. `:=` lets command-line
# overrides still work but ignores the environment.
LANGUAGE        := sv
AUDIO_DIR       := ./input/$(LANGUAGE)
OUTPUT_ROOT     := ./output/$(LANGUAGE)
ALIGNMENTS      := $(OUTPUT_ROOT)/alignments
THUMBNAILS_DIR  ?= ./thumbnails
METADATA_CSV    ?= ./video_batcher.csv
HF_BUCKET       ?=                       # e.g. gborg/raudio-demo
GPU             ?= 2

.PHONY: help check-deps bootstrap install lock \
	transcribe detect-language thumbnail \
	ingest ingest-with-media ingest-full reindex-fts \
	pipeline pipeline-sharded pipeline-multimodal \
	embed-server rerank-server embed-server-docker rerank-server-docker vllm-stop kernels-prepare embed-chunks extract-chunk-frames embed-chunk-frames speaker-turns \
	merge-speaker-turns embed-speaker-turns merge-speaker-embeddings build-speakers cluster-speakers \
	caption-chunk-frames embed-captions captions topics \
	atlas atlas-visual atlas-caption atlas-all features-all stack-up stack-down \
	compact e2e-smoke backend frontend frontend-build frontend-dev labeler dev \
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
# `make frontend` builds + serves the prebuilt SPA on this port via the Bun
# static server (server.ts binds 0.0.0.0 and proxies /api/* → the backend).
# Forward this port over SSH / your editor to reach it as localhost:5274.
# Prefer this over `frontend-dev` for remote use — no Vite recompile spinner.
FRONTEND_PORT ?= 5274

backend:              ## Run the FastAPI backend (Lance reads, /api/*).
	uv run raudio --db $(DB) serve --host $(BACKEND_HOST) --port $(BACKEND_PORT)

FRONTEND_DIR    ?= ./frontend
FRONTEND_BUILD  ?= $(FRONTEND_DIR)/build

frontend-build:       ## Build the SvelteKit static bundle into $(FRONTEND_BUILD).
	cd $(FRONTEND_DIR) && bun install --silent && bun run build

frontend-dev:         ## Run the SvelteKit dev server with HMR (vite, port 5173).
	cd $(FRONTEND_DIR) && bun install --silent && bun run dev

frontend: frontend-build  ## Build then serve the SvelteKit app via the Bun proxy.
	cd $(FRONTEND_DIR) && bun run server.ts \
		--root ./build \
		--api http://$(BACKEND_HOST):$(BACKEND_PORT) \
		--port $(FRONTEND_PORT)

LABELER_PORT    ?= 3999
labeler:              ## Run the manual language relabeler on $(AUDIO_DIR) (port $(LABELER_PORT)).
	cd tools/labeler && bun install --silent && bun run labeler.ts --root ../../$(AUDIO_DIR) --port $(LABELER_PORT)

# ─── Multimodal embeddings (Phase 1+2 of the multimodal plan) ───────────────
# Long-running vLLM HTTP servers serve Qwen3-VL-Embedding-2B + Reranker-2B.
# CLI commands and the FastAPI backend are clients of these servers.
# Online inference: model loads once, stays warm across all uses.
EMBED_BACKEND   ?= vllm
EMBED_URL       ?= http://127.0.0.1:8001
RERANK_URL      ?= http://127.0.0.1:8002
# Both vLLM servers run on the SAME GPU by default (override the card with
# `make embed-server VLLM_GPU=N`). Two 2B models co-locate fine at ~0.45
# mem-frac each (~88 GB on a 96 GB card).
# IMPORTANT: start them SEQUENTIALLY — bring the embed server fully up before
# launching the reranker. Launching both at once trips vLLM's memory-profiling
# race (one server freeing GPU memory mid-init aborts the other's profile_run).
VLLM_GPU        ?= 0
EMBED_GPU       ?= $(VLLM_GPU)
RERANK_GPU      ?= $(VLLM_GPU)
EMBED_MEM_FRAC  ?= 0.45
RERANK_MEM_FRAC ?= 0.45

# Caption model — a generative VLM (Gemma 4) you ALREADY run locally on :8003.
# raudio is only a CLIENT of it: `raudio feature caption` POSTs frames to this
# URL. We do NOT start the server (no Make target spins one up) — point these at
# wherever your Gemma serves. The client also honours RAUDIO_CAPTION_URL/MODEL.
CAPTION_URL      ?= http://127.0.0.1:8003
CAPTION_MODEL    ?= google/gemma-4-31B-it
# Resume sidecar for captioning. A 145k-frame Gemma pass takes hours; this makes
# it crash-resumable (each caption persisted to {CAPTION_CKPT}.values.jsonl as it's
# produced), so a killed run picks up where it left off instead of recomputing.
CAPTION_CKPT     ?= $(DB).caption.ckpt

# vLLM version pin — set to 0.22.0 (one pinned build across both the uvx and
# Docker routes; see docs/INVESTIGATION.md on why a single build matters).
# Caveats to verify on your box (we can't test without the GPU):
#   * vLLM >= 0.20 requires NVIDIA driver >= 575 (CUDA 12.9). If the host driver
#     only supports CUDA 12.8, the native (uvx) server "driver too old"-crashes at
#     engine init — use the Docker variant (bundles its own CUDA) in that case.
#   * Re-check the Qwen3-VL deepstack image-embed path: 0.22.0 may fix or shift
#     the warmup-buffer overflow — confirm the image pixel-pin matches the
#     server's vision-token count (the 448-vs-392 mismatch in INVESTIGATION.md).
# The HF `kernels` package + FA3 cache (see `kernels-prepare`) remains the
# workaround for the Blackwell sm_120 FA2/PTX gap; keep that infrastructure.
VLLM_PIN ?= vllm==0.22.0
VLLM_ENV ?=
VLLM_INDEX ?=

# Docker variant — uses the official vllm/vllm-openai image which bundles
# its own CUDA toolkit + userspace, sidestepping host driver/toolkit
# mismatches. Recommended path on Blackwell with driver 12.8.
VLLM_IMAGE    ?= vllm/vllm-openai:v0.22.0
HF_CACHE      ?= $(HOME)/.cache/huggingface

# Docker 27+ resolves `--gpus all` through CDI for ALL vendors and aborts
# with "AMD CDI spec not found" on NVIDIA-only hosts. Pass the device by
# CDI name directly — only enumerates the requested vendor.

embed-server-docker:  ## Run vLLM embedding server in Docker (no driver pin).
	# Confirmed vLLM 0.20.0 bug: warmup ignores `mm_processor_kwargs` and
	# sizes the deepstack buffer for ITS OWN dummy image (~218 tokens here),
	# while runtime honours the kwargs. Setting min==max=200704 px gave 224
	# runtime tokens which still overflows the 218-token warmup buffer.
	# Workaround: pin the runtime image to a value comfortably *below* the
	# warmup ceiling. 14*28 = 392 px square → 14×14 = 196 vision tokens.
	# 22-token headroom under the typical 218-token warmup buffer.
	docker run --rm -it \
		--device=nvidia.com/gpu=$(EMBED_GPU) --ipc=host \
		-p 8001:8001 \
		-v $(HF_CACHE):/root/.cache/huggingface \
		--name raudio-embed \
		$(VLLM_IMAGE) \
		--model Qwen/Qwen3-VL-Embedding-2B \
		--runner pooling --port 8001 --enable-prefix-caching \
		--dtype bfloat16 --gpu-memory-utilization $(EMBED_MEM_FRAC) \
		--max-model-len 8192 \
		--limit-mm-per-prompt.image 1 \
		--limit-mm-per-prompt.video 0 \
		--mm-processor-kwargs '{"min_pixels": 153664, "max_pixels": 153664}'
		# NOTE (2B + vLLM 0.22.0): the 392px / 153664-px pin above + the client
		# _IMAGE_SIDE were derived for the 8B on 0.20.0 — re-verify the vision-token
		# count for the 2B tower before trusting the image-embed path (INVESTIGATION.md).

rerank-server-docker: ## Run vLLM reranker server in Docker (no driver pin).
	# Reranker is text-only in raudio (cross-encoder over query/doc strings),
	# so we disable image+video profiling — frees ~1 GB and skips the
	# multimodal warmup that would otherwise size a deepstack buffer for
	# inputs we never send.
	docker run --rm -it \
		--device=nvidia.com/gpu=$(RERANK_GPU) --ipc=host \
		-p 8002:8002 \
		-v $(HF_CACHE):/root/.cache/huggingface \
		-v $(PWD)/src/raudio/retrieval/qwen3_vl_reranker.jinja:/templates/qwen3_vl_reranker.jinja:ro \
		--name raudio-rerank \
		$(VLLM_IMAGE) \
		--model Qwen/Qwen3-VL-Reranker-2B \
		--runner pooling --port 8002 \
		--dtype bfloat16 --gpu-memory-utilization $(RERANK_MEM_FRAC) \
		--max-model-len 4096 \
		--limit-mm-per-prompt '{"image": 0, "video": 0}' \
		--hf_overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
		--chat-template /templates/qwen3_vl_reranker.jinja

vllm-stop:            ## Stop the Docker vLLM containers.
	-docker stop raudio-embed raudio-rerank 2>/dev/null

# vLLM runs in a `uvx`-managed ephemeral env so its torch/torchaudio pins
# don't fight our cu128 ones. First launch downloads vLLM into uv's tool
# cache; subsequent launches reuse it.
kernels-prepare:      ## Pre-download FA3 kernels from HF hub (one-time, ~200 MB).
	@echo "→ Fetching FlashAttention-3 prebuilt kernels for sm_120 …"
	uvx --python 3.12 --with "kernels" --with "torch" python -c \
		"from kernels import get_kernel; m = get_kernel('kernels-community/flash-attn3'); print('✓ FA3 cached:', m.__name__)" || \
		(echo "Hint: 'hf auth login' if the repo gates downloads."; exit 1)
	@echo "✓ Kernels cached under ~/.cache/huggingface/hub/. Now run: make embed-server"

embed-server:         ## Start vLLM Qwen3-VL-Embedding-2B (port 8001) on GPU $(EMBED_GPU).
	CUDA_VISIBLE_DEVICES=$(EMBED_GPU) $(VLLM_ENV) \
	uvx --python 3.12 --with "kernels" $(VLLM_INDEX) --from "$(VLLM_PIN)" vllm serve Qwen/Qwen3-VL-Embedding-2B \
		--runner pooling --port 8001 --enable-prefix-caching \
		--dtype bfloat16 --gpu-memory-utilization $(EMBED_MEM_FRAC) \
		--max-model-len 8192 \
		--limit-mm-per-prompt '{"image": 1}' \
		--mm-processor-kwargs '{"min_pixels": 153664, "max_pixels": 153664}'

# Reranker requires an explicit chat template + hf_overrides per the model
# card to wire up the no/yes classification head and /v1/rerank endpoint.
rerank-server:        ## Start vLLM Qwen3-VL-Reranker-2B (port 8002) on GPU $(RERANK_GPU).
	CUDA_VISIBLE_DEVICES=$(RERANK_GPU) $(VLLM_ENV) \
	uvx --python 3.12 --with "kernels" $(VLLM_INDEX) --from "$(VLLM_PIN)" vllm serve Qwen/Qwen3-VL-Reranker-2B \
		--runner pooling --port 8002 \
		--dtype bfloat16 --gpu-memory-utilization $(RERANK_MEM_FRAC) \
		--max-model-len 4096 \
		--limit-mm-per-prompt '{"image": 0, "video": 0}' \
		--hf_overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
		--chat-template ./src/raudio/retrieval/qwen3_vl_reranker.jinja

embed-chunks:         ## Embed chunks.text → text_embedding column + IVF_PQ index.
	uv run --extra multimodal raudio --db $(DB) feature text_embedding --url $(EMBED_URL)

EXTRACT_JOBS    ?= 16
extract-chunk-frames: ## ffmpeg → one JPEG per chunk.start into the chunk_frames table.
	uv run raudio --db $(DB) extract-chunk-frames --audio-root $(AUDIO_DIR) --jobs $(EXTRACT_JOBS)

embed-chunk-frames:   ## Embed each chunk's frame → frame_embedding + IVF_PQ index.
	uv run --extra multimodal raudio --db $(DB) feature frame_embedding --url $(EMBED_URL)

speaker-turns:        ## Diarize each video → speaker_turns.lance (who-spoke-when; needs pyannote + cached HF token).
	uv run raudio --db $(DB) extract-speaker-turns --audio-root $(AUDIO_DIR) $(if $(LIMIT),--limit $(LIMIT),)

merge-speaker-turns:  ## Fold sharded speaker_turns staging tables into the canonical table.
	uv run raudio --db $(DB) merge-speaker-turns

embed-speaker-turns:  ## Per-turn WeSpeaker 256-d voiceprints → speaker_embeddings.lance (GPU; shardable).
	uv run raudio --db $(DB) embed-speaker-turns --audio-root $(AUDIO_DIR) $(if $(LIMIT),--limit $(LIMIT),)

merge-speaker-embeddings: ## Fold speaker_embeddings shards into the canonical table + rebuild indexes.
	uv run raudio --db $(DB) merge-speaker-embeddings

build-speakers:       ## Duration-weighted centroid per (doc, label) → speakers.lance.
	uv run raudio --db $(DB) build-speakers

cluster-speakers:     ## Seeded EVōC over speakers → speaker_cluster (cross-video identities; needs atlas extra).
	uv run --extra atlas raudio --db $(DB) cluster-speakers --seed 42 --validate

caption-chunk-frames: ## Caption EXISTING frames → chunk_frames.caption (resumable; uses your Gemma at $(CAPTION_URL)).
	uv run --extra multimodal raudio --db $(DB) feature caption \
		--url $(CAPTION_URL) --model $(CAPTION_MODEL) --checkpoint $(CAPTION_CKPT)

embed-captions:       ## Embed chunk_frames.caption → caption_embedding + IVF_PQ index (needs embed-server).
	uv run --extra multimodal raudio --db $(DB) feature caption_embedding --url $(EMBED_URL)

# Caption pipeline: write the Swedish captions, then embed them for scene search.
# Reuses existing frames — run `extract-chunk-frames` first if chunk_frames is empty.
captions: caption-chunk-frames embed-captions  ## Caption frames + embed captions (scene search).

# ─── Atlas 2-D projections (EVōC; CPU-only, needs the `atlas` extra) ──────────
# Build the atlas_*_x/y/cluster columns the Atlas view reads. Run each after its
# embedding exists: text ← embed-chunks, visual ← embed-chunk-frames,
# caption ← embed-captions. EVōC is CPU-only (no GPU/embed-server needed).
atlas:                ## Text EVōC map (atlas_x/y/cluster) from chunks.text_embedding.
	uv run --extra atlas raudio --db $(DB) feature atlas
atlas-visual:         ## Visual EVōC map (atlas_img_*) from frame_embedding.
	uv run --extra atlas raudio --db $(DB) feature atlas --space visual
atlas-caption:        ## Caption EVōC map (atlas_cap_*) from caption_embedding.
	uv run --extra atlas raudio --db $(DB) feature atlas --space caption
atlas-all: atlas atlas-visual atlas-caption  ## All three atlas projections.

topics:               ## Build Swedish topic layers (Toponymy, isolated env; needs atlas map + Gemma :8003 + embed :8001).
	uv run raudio --db $(DB) feature topics

# ─── Complete feature DAG (everything the serving DB carries) ─────────────────
# The full multimodal + atlas + topics chain in dependency order. Assumes
# `ingest-full` already ran and the embed server (:8001) + your Gemma (:8003) are
# up. Each stage is resumable (skips populated rows). Hours of GPU for 145k rows.
# This is what `pipeline-multimodal` was missing (captions, atlas, topics).
features-all: embed-chunks extract-chunk-frames embed-chunk-frames speaker-turns captions atlas-all topics compact  ## Build every derived column on $(DB).

# Full multimodal indexing chain. Existing `pipeline` runs first, then the
# three new stages add the multimodal columns + indexes. Resumable: each
# new stage skips already-populated rows via `WHERE … IS NULL`.
pipeline-multimodal: pipeline embed-chunks extract-chunk-frames embed-chunk-frames compact

compact:               ## Compact $(TABLE)'s fragments + rebuild its indexes (run after bulk writes; TABLE=chunk_frames for frames).
	uv run raudio --db $(DB) --table $(TABLE) compact
	@echo "── multimodal indexing complete ────────────────────────────────"

E2E_DOCS        ?= 2
E2E_FRAME_LIMIT ?= 24
e2e-smoke:            ## Full-pipeline e2e on a tiny real sample → throwaway DB (needs embed-server on :8001).
	uv run --extra multimodal python scripts/e2e_smoke.py --docs $(E2E_DOCS) --frame-limit $(E2E_FRAME_LIMIT)

dev:                  ## Run backend + frontend together (tmux or two terminals).
	@echo "Run these in two terminals:"
	@echo "  1) make backend"
	@echo "  2) make frontend"
	@echo "Then open http://localhost:$(FRONTEND_PORT)"

# ─── Whole-stack bring-up (4 services, detached + health-gated) ──────────────
# One command for embed:8001 → rerank:8002 → backend:8000 → frontend:5274.
# Sequential vLLM start + idempotent (skips healthy ports). See docs/REPRODUCE.md.
stack-up:             ## Start the full stack detached (DB=transcripts_v2.lance VLLM_GPU=0).
	DB=$(DB) VLLM_GPU=$(VLLM_GPU) bash scripts/serve-all.sh up
stack-down:           ## Stop the detached stack started by `make stack-up`.
	bash scripts/serve-all.sh down

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
