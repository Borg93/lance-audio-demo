"""CLI: `raudio {transcribe, detect-language, ingest, search}` — built with Typer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from .ingest import ingest_many, load_transcript, reindex_fts
from .search import extract_query_terms, iter_matching_words, nearest_chunks, timecode


app = typer.Typer(
    name="raudio",
    help="Audio transcription → Lance ingestion → full-text search.",
    no_args_is_help=True,
    add_completion=False,
)


# Global state carried between the root callback and subcommands.
class _Ctx:
    db: Path = Path("./transcripts.lance")
    table: str = "chunks"


@app.callback()
def _root(
    ctx: typer.Context,
    db: Annotated[
        Path,
        typer.Option("--db", help="Path to the Lance database."),
    ] = Path("./transcripts.lance"),
    table: Annotated[
        str,
        typer.Option("--table", help="Table name."),
    ] = "chunks",
) -> None:
    _Ctx.db = db
    _Ctx.table = table


@app.command("transcribe")
def cmd_transcribe(
    audio_dir: Annotated[Path, typer.Option("--audio-dir", exists=True, file_okay=False)],
    language: Annotated[str, typer.Option("--language", help="ISO-639-1 code (sv, en, …).")] = "sv",
    model: Annotated[str, typer.Option("--model")] = "KBLab/kb-whisper-large",
    emissions_model: Annotated[str | None, typer.Option("--emissions-model")] = None,
    vad: Annotated[str, typer.Option("--vad", help="pyannote or silero.")] = "pyannote",
    backend: Annotated[str, typer.Option("--backend", help="ct2 or hf.")] = "ct2",
    device: Annotated[str, typer.Option("--device")] = "cuda",
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("models"),
    output_root: Annotated[Path, typer.Option("--output-root")] = Path("output"),
    batch_size_features: Annotated[
        int,
        typer.Option(
            "--batch-size-features",
            help="Batch size for Whisper/wav2vec2 inference. 64 fits ~25 GB on a 96 GB GPU.",
        ),
    ] = 64,
    num_workers_features: Annotated[int, typer.Option("--num-workers-features")] = 8,
    batch_size_files: Annotated[int, typer.Option("--batch-size-files")] = 1,
    num_workers_files: Annotated[int, typer.Option("--num-workers-files")] = 2,
    beam_size: Annotated[
        int,
        typer.Option(
            "--beam-size",
            help=(
                "Whisper beam size. 1 is ~3-5× faster than the default 5 "
                "with negligible quality loss on clean audio. Bump to 5 if "
                "you see obviously garbled transcripts."
            ),
        ),
    ] = 1,
) -> None:
    """Run easytranscriber on a directory of audio/video files → alignment JSONs."""
    # Lazy import — the `[transcribe]` extra is optional.
    from .transcribe import run_transcribe

    if vad not in {"pyannote", "silero"}:
        raise typer.BadParameter("--vad must be 'pyannote' or 'silero'")
    if backend not in {"ct2", "hf"}:
        raise typer.BadParameter("--backend must be 'ct2' or 'hf'")

    run_transcribe(
        audio_dir=audio_dir,
        language=language,
        model=model,
        emissions_model=emissions_model,
        vad=vad,
        backend=backend,
        device=device,
        cache_dir=cache_dir,
        output_root=output_root,
        batch_size_features=batch_size_features,
        num_workers_features=num_workers_features,
        batch_size_files=batch_size_files,
        num_workers_files=num_workers_files,
        beam_size=beam_size,
    )


@app.command("detect-language")
def cmd_detect_language(
    audio_dir: Annotated[Path, typer.Option("--audio-dir", exists=True, file_okay=False)],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Classifier model. Default facebook/mms-lid-256 (SOTA for "
                "language ID). Also supports multilingual Whisper like "
                "openai/whisper-large-v3. Never use language-fine-tuned "
                "models (e.g. KBLab/kb-whisper-large) — they over-predict."
            ),
        ),
    ] = "openai/whisper-large-v3",
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("models"),
    sample_seconds: Annotated[
        float,
        typer.Option("--sample-seconds", help="Audio clip length fed to Whisper per sample."),
    ] = 30.0,
    sample_offset: Annotated[
        float,
        typer.Option(
            "--sample-offset",
            help=(
                "Base offset (seconds). We actually sample at 1x/3x/5x this "
                "value and vote, so default 60s → samples at 60s/180s/300s."
            ),
        ),
    ] = 60.0,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    no_move: Annotated[
        bool,
        typer.Option("--no-move", help="Report detected languages without moving files."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show planned moves without executing them."),
    ] = False,
) -> None:
    """Detect language per file via Whisper and sort into <audio-dir>/<lang>/ subfolders."""
    from .detect_language import detect_and_sort

    detect_and_sort(
        audio_dir=audio_dir,
        model=model,
        cache_dir=cache_dir,
        sample_seconds=sample_seconds,
        sample_offset=sample_offset,
        device=device,
        move=not no_move,
        dry_run=dry_run,
    )


@app.command("thumbnail")
def cmd_thumbnail(
    input_dir: Annotated[Path, typer.Option("--input-dir", exists=True, file_okay=False)] = Path("input"),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("thumbnails"),
    at_sec: Annotated[
        float,
        typer.Option("--at", help="Seek this many seconds into each video before grabbing a frame."),
    ] = 5.0,
    width: Annotated[int, typer.Option("--width", help="Target thumbnail width in pixels.")] = 480,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Regenerate thumbnails that already exist."),
    ] = False,
) -> None:
    """Extract a JPEG thumbnail per media file (via ffmpeg) into <output-dir>/{stem}.jpg."""
    from .thumbnails import generate_thumbnails

    generate_thumbnails(
        input_dir=input_dir,
        output_dir=output_dir,
        at_sec=at_sec,
        width=width,
        overwrite=overwrite,
    )


@app.command("download")
def cmd_download(
    csv_path: Annotated[Path, typer.Option("--csv", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("input"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Only download the first N rows (for testing)."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Simultaneous downloads."),
    ] = 4,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Per-request timeout (seconds)."),
    ] = 600.0,
) -> None:
    """Bulk-download Riksarkivet media from a ``video_batcher`` CSV into <output-dir>/{bildid}.mp4."""
    from .download import download_manifest

    download_manifest(
        csv_path=csv_path,
        output_dir=output_dir,
        limit=limit,
        concurrency=concurrency,
        timeout=timeout,
    )


@app.command("ingest")
def cmd_ingest(
    json_paths: Annotated[list[Path], typer.Argument(metavar="JSON...", help="AudioMetadata JSON files.")],
    audio_root: Annotated[
        Path | None,
        typer.Option(
            "--audio-root",
            help=(
                "Local directory holding the source media files. When set "
                "(and --media-base-uri is not), each row's media_uri is "
                "generated as file:///abs/path/<filename>."
            ),
        ),
    ] = None,
    media_base_uri: Annotated[
        str | None,
        typer.Option(
            "--media-base-uri",
            help=(
                "Base URI under which to reference videos in the documents "
                "table. Overrides --audio-root. Examples: "
                "'hf://buckets/you/videos/', 's3://bucket/videos/', "
                "'https://cdn.example.com/videos/'."
            ),
        ),
    ] = None,
    metadata_csv: Annotated[
        Path | None,
        typer.Option(
            "--metadata-csv",
            help=(
                "Optional video_batcher CSV (referenskod;namn;extraid;bildid). "
                "Joined to transcripts by bildid == audio_path stem."
            ),
        ),
    ] = None,
    thumbnail_dir: Annotated[
        Path | None,
        typer.Option(
            "--thumbnail-dir",
            help=(
                "Directory of {stem}.jpg thumbnails (see `raudio thumbnail`). "
                "If set, each document row stores the path to its thumbnail; "
                "the viewer can then serve them for a gallery."
            ),
        ),
    ] = None,
    fts_language: Annotated[
        str,
        typer.Option(
            "--fts-language",
            help=(
                "Stemmer/stop-word language for the FTS index. "
                "Use 'Swedish' for Swedish text — default 'English' mis-stems "
                "forms like 'ministern'/'vägen'/'ansåg'. Supported: English, "
                "Swedish, Norwegian, Danish, Finnish, French, German, Spanish, "
                "Italian, Portuguese, Dutch, Russian, and more."
            ),
        ),
    ] = "English",
    doc_language: Annotated[
        str | None,
        typer.Option(
            "--doc-language",
            help=(
                "2-letter ISO 639-1 language code stamped on every ingested "
                "row (documents.language + chunks.language). If omitted, we "
                "infer from the alignments dir: output/sv/alignments → 'sv'."
            ),
        ),
    ] = None,
) -> None:
    """Ingest one or more easytranscriber AudioMetadata JSON files."""
    docs = [load_transcript(p) for p in json_paths]

    # Infer doc_language from the alignments dir if not explicitly passed.
    # `output/sv/alignments/foo.json` → parent.parent.name == 'sv'.
    if doc_language is None and json_paths:
        candidate = json_paths[0].parent.parent.name
        if len(candidate) == 2 and candidate.isalpha():
            doc_language = candidate.lower()

    table = ingest_many(
        _Ctx.db,
        docs,
        audio_root=audio_root,
        media_base_uri=media_base_uri,
        table_name=_Ctx.table,
        metadata_csv=metadata_csv,
        thumbnail_dir=thumbnail_dir,
        fts_language=fts_language,
        doc_language=doc_language,
    )
    suffix = ""
    if doc_language:
        suffix += f" + language={doc_language}"
    if media_base_uri:
        suffix += f" + media URIs under {media_base_uri}"
    elif audio_root:
        suffix += f" + media URIs (file://) from {audio_root}"
    if metadata_csv:
        suffix += f" + metadata from {metadata_csv.name}"
    if thumbnail_dir:
        suffix += f" + thumbnails from {thumbnail_dir}"
    suffix += f" + FTS({fts_language})"
    typer.echo(
        f"Ingested {len(docs)} transcript(s) → '{_Ctx.table}' now has "
        f"{table.count_rows()} chunk row(s){suffix}.",
        err=True,
    )


@app.command("serve")
def cmd_serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    """Launch the API-only FastAPI backend against the Lance DB.

    The Bun frontend in ./frontend/ proxies /api/* to this server.
    """
    from backend import run

    run(db_path=_Ctx.db, host=host, port=port)


@app.command("reindex-fts")
def cmd_reindex_fts(
    language: Annotated[
        str,
        typer.Option(
            "--language",
            help="Stemmer/stop-word language. Use 'Swedish' for Swedish text.",
        ),
    ] = "Swedish",
    with_position: Annotated[
        bool,
        typer.Option("--with-position/--no-with-position", help="Required for phrase queries."),
    ] = True,
    remove_stop_words: Annotated[
        bool,
        typer.Option("--remove-stop-words/--keep-stop-words"),
    ] = False,
    ascii_folding: Annotated[
        bool,
        typer.Option("--ascii-folding/--no-ascii-folding"),
    ] = True,
) -> None:
    """Rebuild only the FTS index on an existing chunks table. No re-ingest."""
    reindex_fts(
        _Ctx.db,
        table_name=_Ctx.table,
        language=language,
        with_position=with_position,
        remove_stop_words=remove_stop_words,
        ascii_folding=ascii_folding,
    )
    typer.echo(
        f"Rebuilt FTS index on '{_Ctx.table}' "
        f"(language={language}, with_position={with_position}, "
        f"remove_stop_words={remove_stop_words}, ascii_folding={ascii_folding}).",
        err=True,
    )


@app.command("search")
def cmd_search(
    query: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option("-n", "--limit")] = 10,
    where: Annotated[
        str | None,
        typer.Option("--where", help="Optional SQL filter, e.g. language = 'en'."),
    ] = None,
    words: Annotated[
        bool,
        typer.Option(
            "--words/--no-words",
            help="List each matching word with its exact timestamp (ms precision).",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json-output", help="Emit raw JSON hits to stdout."),
    ] = False,
) -> None:
    """Run a full-text (Tantivy BM25) query."""
    hits = nearest_chunks(
        _Ctx.db,
        query,
        table_name=_Ctx.table,
        limit=limit,
        where=where,
        include_alignments=words or json_output,
    )

    if json_output:
        json.dump(hits, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if not hits:
        typer.echo("(no hits)", err=True)
        return

    terms = extract_query_terms(query) if words else []
    for i, h in enumerate(hits, 1):
        typer.echo(
            f"{i:>2}. [{timecode(h['start'])}→{timecode(h['end'])}] "
            f"{Path(h['audio_path']).name}"
        )
        typer.echo(f"     {h['text']}")
        if words:
            matches = iter_matching_words(h, terms)
            if matches:
                for w in matches:
                    typer.echo(
                        f"     • [{timecode(w['start'], millis=True)}] "
                        f"{w['text'].strip()}"
                    )
            else:
                typer.echo("     (chunk matched, no exact word hit — phrase/stemming?)")


if __name__ == "__main__":
    app()
