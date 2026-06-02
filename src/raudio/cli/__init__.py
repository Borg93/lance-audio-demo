"""CLI for raudio — built with Typer.

Exposes the full pipeline as subcommands: ``transcribe``, ``detect-language``,
``ingest``, ``reindex-fts``, ``search``, ``serve``, ``feature``, ``thumbnail``,
``download``, ``extract-chunk-frames``, ``compact``. Run ``raudio --help`` for
the authoritative list.

The Typer ``app`` lives in :mod:`._app`; each command group module registers its
commands against it. Importing those modules here is what wires them onto ``app``
so the ``raudio.cli:app`` console entry point sees the full command set.
"""

from . import features, ingest, media, search, transcribe  # noqa: F401 — register commands
from ._app import app

__all__ = ["app"]
