"""Ingest-stage commands: ``ingest`` (JSON → Lance) and ``reindex-fts``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..ingest.ingest import ingest_many, load_transcript, reindex_fts
from ._app import _Ctx, app


@app.command("ingest")
def cmd_ingest(
    json_paths: Annotated[
        list[Path], typer.Argument(metavar="JSON...", help="AudioMetadata JSON files.")
    ],
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
    from tqdm import tqdm

    # Parsing the alignment JSONs is the I/O-heavy part; the subsequent table
    # write + FTS index build log their own progress (see ingest_many).
    docs = [
        load_transcript(p)
        for p in tqdm(json_paths, unit="file", desc="parsing", smoothing=0.05)
    ]

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
