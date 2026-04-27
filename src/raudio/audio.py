"""Resolve source media paths and compute their URIs / MIME types.

Videos are **referenced** from the Lance table via a URI string in the
``documents.media_uri`` column — Lance Blob V2 "External" semantics, minus
the Blob V2 on-disk encoding (which the @lancedb/lancedb TS binding can't
read yet). The Lance table stays a small portable catalog; the actual MP4
bytes live wherever the URI points:

- Local dev: ``file:///abs/path/input/T0001417_00001.mp4``
- HF bucket: ``hf://buckets/you/videos/T0001417_00001.mp4``
- S3:        ``s3://bucket/videos/T0001417_00001.mp4``

If/when the TS binding gains ``takeBlobs``, flipping to true Blob V2
External (``blob_field`` + ``blob_array([uri, …])``) is a schema-only change.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_source(audio_path: str, audio_root: str | Path | None) -> Path | None:
    """Resolve a transcript's ``audio_path`` against an optional root directory.

    Returns ``None`` (and logs a warning) when the file isn't found.
    """
    if audio_root is None or Path(audio_path).is_absolute():
        p = Path(audio_path)
    else:
        p = Path(audio_root) / audio_path

    if not p.exists():
        logger.warning("media source not found: %s", p)
        return None
    return p


def guess_mime(path: str | Path) -> str:
    """Guess MIME from filename extension, falling back to ``application/octet-stream``."""
    mime, _ = mimetypes.guess_type(Path(path).name)
    return mime or "application/octet-stream"


def compose_media_uri(
    *,
    audio_path: str,
    source_path: Path | None,
    base_uri: str | None,
) -> str | None:
    """Compose the external URI for a source media file.

    Precedence:
      1. ``base_uri`` (e.g. ``hf://buckets/you/videos/``) + ``audio_path``
         basename → treat the transcript's filename as the key.
      2. ``source_path`` resolved to an absolute ``file://`` URI.
      3. ``None`` (media not locatable).
    """
    if base_uri:
        sep = "" if base_uri.endswith("/") else "/"
        return f"{base_uri}{sep}{Path(audio_path).name}"
    if source_path is not None:
        return source_path.resolve().as_uri()
    return None
