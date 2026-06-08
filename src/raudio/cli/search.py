"""Query-side commands: ``search`` (FTS) and ``serve`` (the API backend)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from ..retrieval.search import extract_query_terms, iter_matching_words, nearest_chunks, timecode
from ._app import CliContext, app


@app.command("search")
def cmd_search(
    ctx: typer.Context,
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
    cfg: CliContext = ctx.obj
    hits = nearest_chunks(
        cfg.db,
        query,
        table_name=cfg.table,
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
            f"{i:>2}. [{timecode(h['start'])}→{timecode(h['end'])}] {Path(h['audio_path']).name}"
        )
        typer.echo(f"     {h['text']}")
        if words:
            matches = iter_matching_words(h, terms)
            if matches:
                for w in matches:
                    typer.echo(f"     • [{timecode(w['start'], millis=True)}] {w['text'].strip()}")
            else:
                typer.echo("     (chunk matched, no exact word hit — phrase/stemming?)")


@app.command("serve")
def cmd_serve(
    ctx: typer.Context,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    """Launch the API-only FastAPI backend against the Lance DB.

    The Bun frontend in ./frontend/ proxies /api/* to this server.
    """
    from backend import run

    cfg: CliContext = ctx.obj
    run(db_path=cfg.db, host=host, port=port)
