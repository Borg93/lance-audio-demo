"""FastAPI backend for the raudio Lance-backed viewer.

Lives at repo root as a standalone top-level package, separate from the
``raudio`` ASR/ingest CLI package. Public API:

    from backend import create_app, run
"""

from .app import create_app, run

__all__ = ["create_app", "run"]
