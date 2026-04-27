"""Single-frame ffmpeg extraction for chunk-level visual indexing.

Used by ``raudio extract-chunk-frames`` to grab one representative JPEG
per transcript chunk at ``chunk.start``. Output goes straight into the
``chunks.frame_blob`` Blob V2 Inline column — no temp files, no manifest
CSV, no separate frames table.

ffmpeg is invoked with ``-ss <t>`` *before* ``-i <src>`` for fast-seek;
that's slightly less accurate than slow-seek (decodes from the previous
keyframe forward, then drops frames) but plenty for press-conf footage
where the speaker barely moves between keyframes. Output is piped to
stdout as MJPEG so we never touch disk.

CPU-bound ffmpeg startup is the bottleneck (~80–120ms/chunk). For ~145k
chunks that's ~3–5 hours single-threaded; the worker-pool helper below
parallelizes 4–8 ways for ~30–60 minutes wall-clock.
"""

from __future__ import annotations

import logging
import shutil
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator

import msgspec


logger = logging.getLogger(__name__)


# Width is a Qwen3-VL-friendly default — 448px wide preserves enough
# detail for visual retrieval, JPEG ~50–80 KB at q=4 fits Lance's Blob V2
# Inline 64 KB threshold most of the time, spilling to sidecar otherwise.
DEFAULT_FRAME_WIDTH: int = 448
DEFAULT_JPEG_QUALITY: int = 4   # ffmpeg `-q:v` (1=best, 31=worst); 4 ~= JPEG q≈85


class ExtractedFrame(msgspec.Struct):
    """Result of one ffmpeg extraction. ``jpeg_bytes`` is empty on failure."""
    doc_id: str
    speech_id: int
    chunk_id: int
    time_sec: float
    jpeg_bytes: bytes
    width: int
    height: int
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────
# JPEG dimensions parser (avoids round-tripping through PIL just for size)
# ─────────────────────────────────────────────────────────────────────


def _jpeg_dimensions(jpeg: bytes) -> tuple[int, int]:
    """Return (width, height) from a JPEG SOF0/SOF2 marker.

    Faster + lighter than `PIL.Image.open(BytesIO(jpeg)).size` because
    we never decode pixel data, and `[multimodal]` may not be installed
    when the *extraction* CLI runs (it's CPU-only, no model needed).
    Returns (0, 0) if parsing fails — non-fatal, dimensions are stored
    only as metadata.
    """
    try:
        i = 2  # skip SOI
        while i < len(jpeg) - 9:
            if jpeg[i] != 0xFF:
                return 0, 0
            marker = jpeg[i + 1]
            # Standalone markers / restart markers
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = struct.unpack(">H", jpeg[i + 2:i + 4])[0]
            # SOF0/SOF1/SOF2 carry the actual frame dimensions
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", jpeg[i + 5:i + 9])
                return w, h
            i += 2 + seg_len
    except Exception:  # noqa: BLE001
        pass
    return 0, 0


# ─────────────────────────────────────────────────────────────────────
# Single-chunk extraction
# ─────────────────────────────────────────────────────────────────────


def extract_chunk_frame(
    *,
    source: Path,
    time_sec: float,
    width: int = DEFAULT_FRAME_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    timeout: float = 30.0,
) -> tuple[bytes, int, int]:
    """Pull a single JPEG frame at ``time_sec`` from ``source`` via ffmpeg.

    Returns (jpeg_bytes, width, height). Raises :class:`RuntimeError`
    with the ffmpeg stderr tail if ffmpeg fails (including missing source).
    """
    # `-ss` before `-i` triggers fast-seek (input-stream demuxer level).
    # `-frames:v 1` exits after writing one frame to stdout.
    # `-f mjpeg` forces the muxer when the destination is `pipe:1`.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{time_sec:.3f}",
        "-i", str(source),
        "-frames:v", "1",
        "-vf", f"scale={width}:-2",
        "-q:v", str(jpeg_quality),
        "-f", "mjpeg",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {timeout}s on {source}") from e

    if proc.returncode != 0 or not proc.stdout:
        tail = (proc.stderr or b"").decode("utf-8", errors="replace").strip().splitlines()[-3:]
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}) on {source} @ {time_sec:.2f}s: "
            + " | ".join(tail)
        )

    jpeg = proc.stdout
    w, h = _jpeg_dimensions(jpeg)
    return jpeg, w, h


# ─────────────────────────────────────────────────────────────────────
# Batch / worker-pool helper
# ─────────────────────────────────────────────────────────────────────


def _extract_one(args: tuple) -> ExtractedFrame:
    """Process-pool worker: unpacks args, calls extract_chunk_frame."""
    (doc_id, speech_id, chunk_id, time_sec, source_str,
     width, jpeg_quality, timeout) = args
    try:
        jpeg, w, h = extract_chunk_frame(
            source=Path(source_str),
            time_sec=time_sec,
            width=width,
            jpeg_quality=jpeg_quality,
            timeout=timeout,
        )
        return ExtractedFrame(
            doc_id=doc_id, speech_id=speech_id, chunk_id=chunk_id,
            time_sec=time_sec, jpeg_bytes=jpeg, width=w, height=h,
        )
    except Exception as e:  # noqa: BLE001
        return ExtractedFrame(
            doc_id=doc_id, speech_id=speech_id, chunk_id=chunk_id,
            time_sec=time_sec, jpeg_bytes=b"", width=0, height=0,
            error=str(e),
        )


def extract_chunk_frames_parallel(
    rows: Iterable[tuple[str, int, int, float, Path]],
    *,
    width: int = DEFAULT_FRAME_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    timeout: float = 30.0,
    jobs: int = 4,
) -> Iterator[ExtractedFrame]:
    """Run extraction across a worker pool.

    ``rows`` yields ``(doc_id, speech_id, chunk_id, time_sec, source_path)``.
    Yields :class:`ExtractedFrame` results in completion order — callers
    are responsible for any buffering / batching before writing to Lance.

    Failures are surfaced as ``ExtractedFrame(jpeg_bytes=b"", error=…)``
    so a single broken video doesn't kill the whole run; the caller
    decides whether to skip-and-warn or raise.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH — install ffmpeg before running extract-chunk-frames")

    payload = [
        (doc_id, speech_id, chunk_id, float(time_sec), str(source),
         width, jpeg_quality, timeout)
        for (doc_id, speech_id, chunk_id, time_sec, source) in rows
    ]
    if not payload:
        return

    if jobs <= 1:
        for args in payload:
            yield _extract_one(args)
        return

    # ffmpeg runs as a subprocess (GIL released during wait), so threads are
    # plenty parallel here. ThreadPool also dodges the `lance is not fork-safe`
    # warning that ProcessPoolExecutor triggers via the default fork start
    # method on Linux.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for fut in as_completed(pool.submit(_extract_one, args) for args in payload):
            yield fut.result()
