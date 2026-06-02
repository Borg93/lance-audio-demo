"""Feature-column command: ``feature`` — derive a column via Lance data evolution."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ._app import _Ctx, _die, _require_table, app


@app.command("feature")
def cmd_feature(
    name: Annotated[str, typer.Argument(help="Feature column to build (see the list below).")],
    url: Annotated[
        str | None,
        typer.Option("--url", help="Model server base URL (default: the feature's own)."),
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Rows per embed batch.")] = 256,
    only_null: Annotated[
        bool,
        typer.Option("--only-null/--all", help="Top up unpopulated rows (resumable) vs rebuild."),
    ] = True,
    create_index: Annotated[
        bool,
        typer.Option("--create-index/--no-create-index", help="Build IVF_PQ index (vector features)."),
    ] = True,
    num_partitions: Annotated[int, typer.Option("--num-partitions")] = 256,
    num_sub_vectors: Annotated[int, typer.Option("--num-sub-vectors")] = 64,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="batch_udf checkpoint file → crash-resumable."),
    ] = None,
) -> None:
    """Build a derived feature column via Lance data evolution (add_columns).

    Available features:

      * text_embedding  — chunks.text  → 2048-d vector (semantic search)
      * frame_embedding — chunk_frames → 2048-d vector (visual search)
      * summary         — chunks.text  → one-line LLM summary
      * caption         — chunk_frames → VLM caption

    Attaches one new column file, no fragment rewrites. `--only-null` (default)
    tops up rows a later ingest added; `--all` drops and rebuilds.
    """
    import lancedb
    from tqdm import tqdm

    from ..features.columns import FEATURES, FeatureRunOptions

    feature = FEATURES.get(name)
    if feature is None:
        _die(f"Unknown feature '{name}'. Available: {', '.join(FEATURES)}.")

    if feature.table == "chunks":
        _require_table(lancedb.connect(str(_Ctx.db)), "chunks")
    elif not (_Ctx.db / f"{feature.table}.lance").exists():
        _die(f"Table '{feature.table}' missing — run `raudio extract-chunk-frames` first.")

    options = FeatureRunOptions(
        url=url,
        batch_rows=batch_size,
        overwrite=not only_null,
        create_index=create_index,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        checkpoint=checkpoint,
    )
    typer.echo(f"Building feature '{name}' on {feature.table} …", err=True)
    with tqdm(unit="row", smoothing=0.05) as pbar:
        written = feature.run(_Ctx.db, options, pbar.update)
    typer.echo(f"  done: {written} row(s) written.", err=True)
