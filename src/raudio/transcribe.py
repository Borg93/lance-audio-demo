"""Thin wrapper around ``easytranscriber.pipelines.pipeline`` that produces the
JSON files ``raudio ingest`` reads.

Exposed as ``raudio transcribe …`` via :mod:`raudio.cli`. The
``easytranscriber`` dependency is optional (pulls torch + pyannote + …) and
lives in the ``[transcribe]`` project extra, so we import it lazily and
surface a clear error message if the extra wasn't installed.
"""

from __future__ import annotations

from pathlib import Path


# Suitable (language → wav2vec2 emissions model) defaults. See also:
# https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
DEFAULT_EMISSIONS_MODEL: dict[str, str] = {
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "en": "facebook/wav2vec2-base-960h",
}

# easyaligner ships Punkt tokenizers keyed by language name, not ISO code.
PUNKT_LANG: dict[str, str] = {"sv": "swedish", "en": "english"}


def run_transcribe(
    *,
    audio_dir: Path,
    language: str = "sv",
    model: str = "KBLab/kb-whisper-large",
    emissions_model: str | None = None,
    vad: str = "pyannote",
    backend: str = "ct2",
    device: str = "cuda",
    cache_dir: Path = Path("models"),
    output_root: Path = Path("output"),
    batch_size_features: int = 64,
    num_workers_features: int = 8,
    batch_size_files: int = 1,
    num_workers_files: int = 2,
    beam_size: int = 1,
) -> Path:
    """Run the full VAD → Whisper → emissions → forced-alignment pipeline.

    Returns the directory the final alignment JSONs were written to
    (``output_root/alignments``).
    """
    try:
        from easyaligner.text import load_tokenizer  # type: ignore[import-not-found]
        from easytranscriber.pipelines import pipeline  # type: ignore[import-not-found]
        from easytranscriber.text.normalization import (  # type: ignore[import-not-found]
            text_normalizer,
        )
    except ImportError as e:
        raise SystemExit(
            "The `transcribe` subcommand requires the `[transcribe]` extra.\n"
            "Install with:  uv sync --extra transcribe\n"
            "           or: pip install 'raudio[transcribe]'\n"
            f"(underlying error: {e})"
        ) from e

    if not audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    emissions_model = emissions_model or DEFAULT_EMISSIONS_MODEL.get(
        language, "facebook/wav2vec2-base-960h"
    )
    tokenizer = load_tokenizer(PUNKT_LANG[language]) if language in PUNKT_LANG else None

    audio_files = sorted(
        f.name for f in audio_dir.iterdir() if f.is_file() and not f.name.startswith(".")
    )
    if not audio_files:
        raise SystemExit(f"No audio files found in {audio_dir}")

    print(
        f"→ Transcribing {len(audio_files)} file(s) from {audio_dir} "
        f"with {model} ({language}, {backend}/{device})."
    )

    pipeline(
        vad_model=vad,
        emissions_model=emissions_model,
        transcription_model=model,
        audio_paths=audio_files,
        audio_dir=str(audio_dir),
        backend=backend,
        language=language,
        tokenizer=tokenizer,
        text_normalizer_fn=text_normalizer,
        cache_dir=str(cache_dir),
        device=device,
        # Bump batch + worker counts. Default easytranscriber values (8/4)
        # are tuned for 8–12 GB consumer GPUs. On a 96 GB PRO 6000 we can
        # push features batch to 64 which uses ~25 GB and gives ~3–5× the
        # Whisper + wav2vec2 throughput.
        batch_size_files=batch_size_files,
        num_workers_files=num_workers_files,
        batch_size_features=batch_size_features,
        num_workers_features=num_workers_features,
        # beam_size=1 is ~3-5× faster than default 5, with negligible quality
        # loss on clean audio (press conferences, lectures, interviews).
        beam_size=beam_size,
        output_vad_dir=str(output_root / "vad"),
        output_transcriptions_dir=str(output_root / "transcriptions"),
        output_emissions_dir=str(output_root / "emissions"),
        output_alignments_dir=str(output_root / "alignments"),
    )

    out_dir = output_root / "alignments"
    print(f"✓ Done. Alignment JSONs written to {out_dir}/")
    print(f"  Next: raudio ingest {out_dir}/*.json")
    return out_dir
