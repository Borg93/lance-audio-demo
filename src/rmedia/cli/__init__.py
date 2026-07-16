"""CLI for raudio — built with Typer.

Exposes the full pipeline as subcommands: ``transcribe``, ``detect-language``,
``ingest``, ``reindex-fts``, ``search``, ``serve``, ``feature``, ``thumbnail``,
``download``, ``extract-chunk-frames``, ``compact``. Run ``raudio --help`` for
the authoritative list.

The Typer ``app`` lives in :mod:`._app` (underscored: a plain ``app.py`` would
make ``rmedia.cli.app`` ambiguously both that module and this re-exported Typer
object). Each command group module registers against it; importing those modules
here is what wires them onto ``app`` so the entry point (``rmedia.__main__``)
sees the full command set.
"""

from . import features, ingest, media, search, speaker, transcribe  # noqa: F401 — register commands
from ._app import app

__all__ = ["app"]
