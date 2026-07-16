"""FastAPI backend for the lance-media viewer/search groups.

A standalone top-level package (no pipeline imports — P2.8). Public API:

    from backend import app, run
"""

from .app import app, run

__all__ = ["app", "run"]
