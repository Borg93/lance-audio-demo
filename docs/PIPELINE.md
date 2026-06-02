# From audio to searchable text: the ASR pipeline & models

> How a Swedish press-conference MP4 becomes word-aligned transcript JSON that
> `raudio ingest` can load. This is the **upstream half** of the write side
> sketched in the [Architecture Guide](../GUIDE.md) §5 — everything that
> happens *before* a Lance table exists. For the schema those JSONs land in,
> see [GUIDE.md §4](../GUIDE.md#4-the-data-model--three-lance-tables); for the
> running task list see [TODO.md](../TODO.md).

`raudio` does **not** implement ASR. It is a thin operator wrapper around two
KBLab libraries — [`easytranscriber`](https://github.com/kb-labb/easytranscriber)
(the pipeline) and `easyaligner` (the forced-alignment backend) — pinning the
*models* and *defaults* appropriate for a Swedish video archive. The two repo
modules that matter here are:

| Module | Subcommand | What it does |
|---|---|---|
| [`src/raudio/asr/transcribe.py`](../src/raudio/asr/transcribe.py) | `raudio transcribe` | Runs the 4-stage `easytranscriber` pipeline → alignment JSONs |
| [`src/raudio/asr/detect_language.py`](../src/raudio/asr/detect_language.py) | `raudio detect-language` | Pre-step: classify each file's language, sort into `<lang>/` |
| [`src/raudio/model/datamodel.py`](../src/raudio/model/datamodel.py) | — | The Pydantic v2 models that *describe* the JSON output |

---

## 1. The 4-stage pipeline at a glance

`easytranscriber.pipelines.pipeline(...)` runs four models in sequence. Each
stage writes its intermediate output to its own directory under
`--output-root` (default `output/`), so a crash mid-run is resumable and you
can inspect any stage. `raudio transcribe` wires all four together with one call
in [`run_transcribe`](../src/raudio/asr/transcribe.py).

```mermaid
flowchart TD
    A["input/sv/*.mp4 (16 kHz mono PCM)"] --> S1
    subgraph S1["Stage 1 — VAD (Voice Activity Detection)"]
        V["pyannote (default) or silero<br/>finds speech regions"]
    end
    S1 -->|"SpeechSegment[] → output/vad/"| S2
    subgraph S2["Stage 2 — Transcription (Whisper)"]
        W["KBLab/kb-whisper-large<br/>backend: ct2 (CTranslate2) or hf<br/>beam_size=1"]
    end
    S2 -->|"AudioChunk[] text → output/transcriptions/"| S3
    subgraph S3["Stage 3 — Emission extraction (wav2vec2 CTC)"]
        E["KBLab/wav2vec2-large-voxrex-swedish<br/>per-frame character logits"]
    end
    S3 -->|"CTC emission matrices → output/emissions/"| S4
    subgraph S4["Stage 4 — Forced alignment (easyaligner)"]
        F["Viterbi alignment of Whisper text<br/>against wav2vec2 emissions"]
    end
    S4 -->|"AlignmentSegment[] + WordSegment[]"| OUT["output/sv/alignments/*.json<br/>(AudioMetadata)"]
    OUT --> ING["raudio ingest → Lance"]
```

**Why two acoustic models?** Whisper (Stage 2) is excellent at *what* was said
but its timestamps are coarse and drift. A wav2vec2 CTC model (Stage 3) emits
fine-grained per-frame character probabilities; forced alignment (Stage 4) then
snaps Whisper's words onto that grid via Viterbi, yielding **millisecond word
timestamps**. The Whisper text is the transcript; the wav2vec2 emissions are
only used to *time* it. This is the same split popularized by WhisperX, which
`transcribe.py` cites directly in its emissions-model comment.

---

## 2. The models this repo pins

All values below are the **actual defaults** in
[`src/raudio/cli/`](../src/raudio/cli/) (`cmd_transcribe`) and the
`DEFAULT_EMISSIONS_MODEL` map in
[`src/raudio/asr/transcribe.py`](../src/raudio/asr/transcribe.py) — not invented.

| Stage | Flag | Default | Notes |
|---|---|---|---|
| 1 VAD | `--vad` | `pyannote` | `silero` is the alternative; validated in `cmd_transcribe` |
| 2 Transcribe | `--model` | `KBLab/kb-whisper-large` | Swedish-fine-tuned Whisper-large |
| 2 Backend | `--backend` | `ct2` | CTranslate2 (fast); `hf` = HF transformers |
| 2 Beam | `--beam-size` | `1` | ~3–5× faster than Whisper's default 5; "negligible quality loss on clean audio" |
| 3 Emissions | `--emissions-model` | `None` → resolved per language | see below |
| — Language | `--language` | `sv` | ISO 639-1; drives emissions model + Punkt tokenizer |
| — Device | `--device` | `cuda` | |
| — Throughput | `--batch-size-features` | `64` | "64 fits ~25 GB on a 96 GB GPU"; default upstream is 8 |

### Emissions model is chosen by language

When `--emissions-model` is omitted (the default), `run_transcribe` resolves it
from `DEFAULT_EMISSIONS_MODEL` keyed on `--language`:

```python
# src/raudio/asr/transcribe.py
DEFAULT_EMISSIONS_MODEL = {
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "en": "facebook/wav2vec2-base-960h",
}
# fallback when the language isn't in the map:
emissions_model = emissions_model or DEFAULT_EMISSIONS_MODEL.get(language, "facebook/wav2vec2-base-960h")
```

`easyaligner` also needs a sentence tokenizer; `transcribe.py` maps the ISO code
to a Punkt model name via `PUNKT_LANG = {"sv": "swedish", "en": "english"}` and
passes `load_tokenizer(...)` plus `easytranscriber`'s `text_normalizer` into the
pipeline.

### The torch pin (why install is fussy)

[`pyproject.toml`](../pyproject.toml) hard-pins the GPU stack:

```toml
torch==2.11.0+cu128
torchaudio==2.11.0+cu128   # from the explicit pytorch-cu128 index
```

The comment is load-bearing: *"driver 570.x supports up to CUDA 12.8, and cu130
wheels fail to initialize (driver too old)."* The whole transcribe stack
(`easytranscriber` + torch + pyannote) is an **optional extra** — install it
with `uv sync --extra transcribe`. `transcribe.py` lazily imports
`easytranscriber`/`easyaligner` inside `run_transcribe` and raises a clear
install hint if the extra is missing, so the FTS-only path needs no GPU at all.

---

## 3. The `detect-language` pre-step

Before transcription you usually have a folder of mixed-language files. Stage 3
(emissions) needs the *correct* language-specific wav2vec2 model, so running
the wrong one wrecks alignment. `raudio detect-language` samples each file,
classifies it, and sorts files into `<audio-dir>/<lang>/` subfolders that
`transcribe` then processes one language at a time.

```mermaid
flowchart TD
    F["each file in audio-dir"] --> SMP
    subgraph SMP["Sample 3 windows (robust to silent intros)"]
        O["read_audio_segment @ 16 kHz<br/>offsets = sample_offset × (1, 3, 5)<br/>default 60s → 60s / 180s / 300s<br/>each 30s long"]
    end
    SMP --> CLS{"--model"}
    CLS -->|"facebook/mms-lid-*"| MMS["Wav2Vec2ForSequenceClassification<br/>softmax → ISO 639-3 (eng, swe, ...)"]
    CLS -->|"openai/whisper-*"| WHI["Whisper LID head (ct2)<br/>→ ISO 639-1 (sv, en, ...)"]
    MMS --> VOTE
    WHI --> VOTE
    VOTE["sum probability per language<br/>across the 3 windows; argmax wins"] --> MAP["ISO 639-3 → 639-1 map<br/>(swe→sv, eng→en, ...)"]
    MAP --> MV["move file → audio-dir/&lt;lang&gt;/<br/>(unless --no-move / --dry-run)"]
```

Grounded details from [`detect_language.py`](../src/raudio/asr/detect_language.py):

- **Two backends, picked by `--model`.** `_mms_probe` loads
  `Wav2Vec2ForSequenceClassification` and softmaxes over its 256 language
  labels (MMS-LID emits ISO 639-3); `_whisper_probe` runs a multilingual
  Whisper's LID head through CTranslate2 (emits ISO 639-1 like `<|sv|>`, braces
  stripped). The module docstring calls `facebook/mms-lid-256` *"state-of-the-art
  for this exact task"* and the `--model` help text names it the recommended
  classifier.
- **The CLI default is `openai/whisper-large-v3`** (`cmd_detect_language`'s
  `--model` default in `cli/`) — a multilingual Whisper, *not* a
  language-fine-tuned one. Both the CLI help and the module docstring warn:
  **never** pass `KBLab/kb-whisper-large` here — fine-tuned models over-predict
  their training language so every file comes back `sv`.
- **3-offset voting.** `OFFSET_MULTIPLIERS = (1.0, 3.0, 5.0)` — it samples at
  `sample_offset × each`, sums each language's probability across windows, and
  takes the argmax. One 30 s window might land on silence, a leader tone, or an
  archive voiceover; voting across 60/180/300 s is robust to that.
- **ISO 639-3 → 639-1 mapping.** `ISO_639_3_TO_1` translates MMS output
  (`swe`, `eng`, `nor`, `dan`, …) into the 2-letter codes the transcribe
  pipeline expects (`sv`, `en`, `no`, `da`, …); unmapped codes pass through
  unchanged. A `✓` is printed when the detected language has a default
  emissions model, `!` otherwise.

---

## 4. The data model the pipeline produces

The final alignment JSON deserializes into one `AudioMetadata` per file. The
nesting is the contract between `easytranscriber` and `raudio ingest`;
[`src/raudio/model/datamodel.py`](../src/raudio/model/datamodel.py) vendors the Pydantic v2 models so ingest can `AudioMetadata.model_validate_json(raw)` *without*
importing torch. Field names and shapes match upstream exactly.

```mermaid
erDiagram
    AudioMetadata ||--o{ SpeechSegment : "speeches[]"
    SpeechSegment ||--o{ AudioChunk : "chunks[]"
    SpeechSegment ||--o{ AlignmentSegment : "alignments[]"
    AlignmentSegment ||--o{ WordSegment : "words[]"

    AudioMetadata {
        str audio_path
        int sample_rate
        float duration
        list speeches
    }
    SpeechSegment {
        int speech_id
        float start
        float end
        str text
        list chunks
        list alignments
    }
    AudioChunk {
        float start
        float end
        str text
        int audio_frames
        int num_logits
        str language
        float language_prob
    }
    AlignmentSegment {
        float start
        float end
        str text
        float score
        list words
    }
    WordSegment {
        str text
        float start
        float end
        float score
    }
```

Two parallel children hang off each `SpeechSegment`, and understanding *why* is
the key to ingest:

- **`chunks` (`AudioChunk[]`)** = Whisper's ~15–30 s transcription windows. The
  `text` here is what becomes a Lance `chunks` row and what the FTS index is
  built on. Carries `audio_frames` / `num_logits` (acoustic bookkeeping) and an
  optional per-chunk `language` / `language_prob` from Whisper auto-detect.
- **`alignments` (`AlignmentSegment[]`)** = the forced-alignment output, each
  holding `WordSegment[]` with millisecond `start`/`end` and a confidence
  `score`. These are the word-level timestamps that power "jump to this word in
  the video."

At ingest, [`flatten_chunks`](../src/raudio/ingest/ingest.py) emits one row per
`AudioChunk` and `_pick_alignments` attaches *only the alignments fully
contained in that chunk's `[start, end]` window* (`a.start >= start and a.end <=
end`), serialized into the `alignments_json` JSONB column. So the chunk grain is
the search unit; the words inside it ride along for precise seeking.

---

## 5. The alignment JSON shape (real example)

Below is the actual structure of
`output/sv/alignments/T0000234_00001.json` (values trimmed for length, but the
field names, types, and nesting are verbatim from the file). Note the
millisecond word timings and the per-word confidence `score`.

```jsonc
{
  "audio_path": "T0000234_00001.mp4",
  "sample_rate": 16000,
  "duration": 4825.824,
  "metadata": { },
  "speeches": [
    {
      "speech_id": 0,
      "start": 7.050968750000001,
      "end": 4784.515343749999,
      "text": "Ett annat skäl till att jag vill gå ut med principmodellen nu ...",
      "chunks": [
        {
          "start": 7.050968750000001,
          "end": 22.17096875,
          "text": "Ett annat skäl till att jag vill gå ut med principmodellen nu ...",
          "duration": 15.12,
          "audio_frames": 241920,
          "num_logits": 755,
          "language": null,
          "language_prob": null,
          "id": "0-0"
        }
        // ... 196 chunks in this speech
      ],
      "alignments": [
        {
          "start": 7.091,
          "end": 21.00112,
          "text": "Ett annat skäl till att jag vill gå ut med principmodellen ...",
          "duration": 13.91,
          "score": 0.97,
          "words": [
            { "text": "Ett ",   "start": 7.091,   "end": 7.19107, "score": 0.99854 },
            { "text": "annat ",  "start": 7.2311,  "end": 7.43125, "score": 0.9998  },
            { "text": "skäl ",   "start": 7.47127, "end": 7.69143, "score": 0.95207 }
            // ... 33 words in this alignment segment
          ]
        }
        // ... 633 alignment segments in this speech
      ]
    }
  ]
}
```

For this one ~80-minute press conference: **1 speech → 196 chunks + 633
alignment segments**, each segment carrying its own word list. `sample_rate` is
`16000` because every stage runs on 16 kHz mono PCM (the rate VAD, Whisper, and
wav2vec2 all expect).

---

## 6. How this feeds `raudio ingest`

The alignment JSONs are the *only* input `raudio ingest` consumes. Their path
follows the `--output-root` you pass to `transcribe`: the **Makefile** pipeline
uses `output/<lang>/` (`OUTPUT_ROOT = ./output/$(LANGUAGE)`), so the Swedish run
writes `output/sv/alignments/`; the **bare CLI** default for `--output-root` is
`output`, so it writes `output/alignments/`. Ingest then points at that dir:

```bash
# Makefile pipeline (per-language):
raudio ingest output/sv/alignments/*.json
# bare CLI default (no --output-root):
raudio ingest output/alignments/*.json
```

```mermaid
sequenceDiagram
    participant CLI as raudio transcribe
    participant ET as easytranscriber.pipeline
    participant FS as output/sv/alignments/*.json
    participant ING as raudio ingest
    participant L as transcripts.lance

    CLI->>ET: pipeline(vad, transcription_model, emissions_model, ...)
    ET-->>FS: AudioMetadata JSON per file
    Note over CLI,FS: GPU-heavy, offline, resumable
    ING->>FS: load_transcript() → AudioMetadata.model_validate_json
    ING->>ING: flatten_chunks() — one row / AudioChunk
    ING->>ING: _pick_alignments() — words inside [chunk.start, chunk.end] → alignments_json
    ING->>L: write chunks table + FTS (Swedish) + BTREE indexes
```

The handoff is purely by convention:

- **Language inference.** `cmd_ingest` infers `doc_language` from the directory
  layout — `output/sv/alignments/foo.json` → `parent.parent.name == "sv"` — so
  the `chunks.language` / `documents.language` columns get stamped correctly
  without an extra flag. This is exactly why `detect-language` sorts into
  `<lang>/` subfolders first.
- **FTS language.** `ingest` builds the Tantivy index with `--fts-language`
  (default `English`, but **use `Swedish`** for this corpus — the English
  stemmer can't reduce forms like `ministern` / `vägen` / `ansåg`). See
  [GUIDE.md §4](../GUIDE.md#4-the-data-model--three-lance-tables) and
  `raudio reindex-fts` for fixing the stemmer after the fact.
- **What the words are for.** The per-word `start`/`end` preserved in
  `alignments_json` is what the search API surfaces as exact word timestamps
  (`raudio search --words`) and what the frontend uses to seek the `<video>`
  element. The full read path picks up from
  [GUIDE.md §5 (read side)](../GUIDE.md#5-end-to-end-information-flow).

In short: **`transcribe` (4 models) → alignment JSON (`AudioMetadata`) →
`ingest` (Lance `chunks`/`documents`)**. Everything downstream — embeddings,
frames, FTS, semantic and visual search — operates on the rows `ingest`
materializes from these JSONs.
