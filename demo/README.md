---
title: Multimodal Lab
emoji: 🎙️
colorFrom: purple
colorTo: pink
sdk: static
pinned: true
header: mini
license: apache-2.0
short_description: Whisper speech-to-text, 100% in your browser.
---

# Multimodal Lab

A local-first, multimodal playground for browser ML. Runs
[KB-Whisper](https://huggingface.co/KBLab) (Swedish) and
[OpenAI Whisper](https://huggingface.co/openai) (English) directly in the
browser via [Transformers.js v4](https://huggingface.co/docs/transformers.js)
on WebGPU. Nothing leaves the tab — audio, model weights and transcripts all
stay on the user's machine.

## Routes

- `/` — **Realtime speech**: live transcription from microphone or tab/system audio, with on-Stop finalisation for word-accurate timestamps.
- `/batch` — **Batch media**: drop in a stack of audio/video files, chunked transcription of any length, downloads per file.
- `/search` — Multimodal search (WIP).
- `/about` — Privacy/tech overview.

## Stack

- SvelteKit 2 + Svelte 5 (runes) · Tailwind v4 · shadcn-svelte · bits-ui
- `@huggingface/transformers` v4 · ONNX Runtime Web
- `@sveltejs/adapter-static` — pure SPA, no server

## Develop

```bash
bun install
bun --bun run dev
```

## Build

```bash
bun run build
```

Produces a self-contained static site in `build/`.

## Deploy to Hugging Face Spaces

This repo is configured as a **static** Space. First-time setup:

```bash
# 1) Build the site
bun run build

# 2) Create a new Space (one time), then clone its git remote
git clone https://huggingface.co/spaces/<your-org>/<space-name>
cd <space-name>

# 3) Copy the contents of build/ (not the folder itself) into the Space repo
cp -r ../audio-demo/build/* .
cp ../audio-demo/README.md .   # keep the YAML frontmatter above at the top

# 4) Push
git add -A
git commit -m "Initial deploy"
git push
```

Subsequent deploys: re-run the build, copy `build/*` into the Space repo, commit, push.

Model weights are **not** included in the build — Transformers.js fetches them from the HF Hub on first run and the browser caches them for later visits.
