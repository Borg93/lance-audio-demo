"""Clip excerpts with MP3 audio — the codec workaround for webview hosts.

VS Code's webview Chromium ships without the AAC decoder (microsoft/vscode
#167685), so the archive's H.264+AAC files play silently inside MCP apps.
This module cuts the requested window out of the source (read seekably via
the backend's own Range-streaming ``/api/media`` endpoint, since media lives
in Lance blobs, not files) and re-encodes audio to MP3, which every webview
supports. Results are small complete MP4s served with normal Range support,
cached on disk and evicted oldest-first.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
from pathlib import Path

from backend.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)

CACHE_DIR = Path(tempfile.gettempdir()) / "raudio-clip-cache"
MAX_CLIP_S = 660.0  # 2×(max transcript half-window) + slack
_MAX_CACHED = 50
_FFMPEG_TIMEOUT_S = 120
# One build at a time: clips are seconds-cheap and the box's CPU belongs to
# the model servers; serializing also makes the tmp+rename publish race-free.
_build_lock = threading.Lock()


def clip_cache_path(doc_id: str, lo: float, hi: float) -> Path:
    """Deterministic cache file for one (doc, window). ``doc_id`` is
    hex-validated upstream, so the name can't traverse."""
    return CACHE_DIR / f"{doc_id}_{int(lo * 1000)}_{int(hi * 1000)}.mp4"


def evict_old_clips(cache_dir: Path, keep: int = _MAX_CACHED) -> None:
    """Drop the oldest cached clips beyond ``keep`` (by mtime)."""
    clips = sorted(
        cache_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for stale in clips[keep:]:
        stale.unlink(missing_ok=True)


def build_clip(source_url: str, doc_id: str, lo: float, hi: float) -> Path:
    """Return the cached clip for the window, transcoding it on first request.

    ``-ss`` before ``-i`` on an HTTP input seeks via Range requests; video is
    re-encoded (not copied) so the output starts frame-accurately at ``lo``
    and the viewer's ``media_offset_s`` mapping holds.
    """
    out = clip_cache_path(doc_id, lo, hi)
    if out.exists():
        return out
    with _build_lock:
        if out.exists():  # built while we waited
            return out
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.mp4")
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v", "error",
            "-ss", f"{lo:.3f}",
            "-i", source_url,
            "-t", f"{hi - lo:.3f}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "libmp3lame",
            "-q:a", "4",
            "-movflags", "+faststart",
            "-y", str(tmp),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S
            )
        except subprocess.TimeoutExpired as exc:
            tmp.unlink(missing_ok=True)
            raise ServiceUnavailableError("clip transcode timed out") from exc
        if proc.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            logger.warning("ffmpeg clip build failed: %s", proc.stderr[-500:])
            raise ServiceUnavailableError("clip transcode failed")
        tmp.rename(out)
        evict_old_clips(CACHE_DIR)
    return out
