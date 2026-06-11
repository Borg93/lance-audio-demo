"""Unit tests for the clip-cache helpers — pure path/eviction logic, no ffmpeg."""

from __future__ import annotations

import os
import time
from pathlib import Path

from backend.media.clips import clip_cache_path, evict_old_clips


def test_cache_path_is_deterministic_and_safe() -> None:
    p1 = clip_cache_path("200aa6de3e6e5826", 12.345, 102.345)
    p2 = clip_cache_path("200aa6de3e6e5826", 12.345, 102.345)
    assert p1 == p2
    assert p1.name == "200aa6de3e6e5826_12345_102345.mp4"
    assert p1.parent.name == "raudio-clip-cache"


def test_evict_keeps_newest(tmp_path: Path) -> None:
    for i in range(6):
        f = tmp_path / f"clip_{i}.mp4"
        f.write_bytes(b"x")
        stamp = time.time() - (100 - i)  # clip_5 is newest
        os.utime(f, (stamp, stamp))
    evict_old_clips(tmp_path, keep=2)
    survivors = sorted(p.name for p in tmp_path.glob("*.mp4"))
    assert survivors == ["clip_4.mp4", "clip_5.mp4"]
