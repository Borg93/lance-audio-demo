"""Shared offline test doubles + Lance dataset builders.

A deterministic stand-in for the vLLM embedding client plus helpers that build
real on-disk Lance tables, so the no-GPU tests exercise the genuine
``LanceTable.search()`` / ``add_columns`` paths. Frame tables go through the
production :func:`rmedia.modalities.av.frames.write_chunk_frames`, so the tests dogfood
that helper instead of hand-rolling the Arrow table.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import lance
import numpy as np

from rmedia.modalities.av.frames import ExtractedFrame, write_chunk_frames
from rmedia.model.datamodel import AudioChunk, AudioMetadata, SpeechSegment
from rmedia.model.schema import EMBED_DIM

# Three topically-distinct chunks for the search tests. The deterministic
# embedder maps each exact string to its own unit vector, so an exact-text query
# is its own nearest neighbour.
TOPICS = {
    "climate": "The minister announced new targets to cut carbon emissions and fight global warming.",
    "sports": "The national football team won the championship final after a penalty shootout.",
    "economy": "Inflation rose as the central bank raised interest rates to cool the economy.",
}


def det_vector(key: bytes) -> np.ndarray:
    """Deterministic, unique, L2-normalized vector for a key (cosine-ready)."""
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


class FakeEmbedClient:
    """Offline drop-in for VLLMEmbeddingClient — deterministic, no network."""

    def embed_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        return np.stack([det_vector(t.encode("utf-8")) for t in texts])

    def embed_image(self, images: list[bytes]) -> np.ndarray:
        if not images:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        return np.stack([det_vector(b) for b in images])


class FakeReranker:
    """Offline drop-in for VLLMReranker — scores by candidate length, no network."""

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        return [float(len(c)) for c in candidates]


class FakeCaptionClient:
    """Offline drop-in for VLLMCaptionClient — derives a caption from the bytes."""

    def caption(self, images: list[bytes]) -> list[str]:
        return [f"caption of {len(img)} bytes" for img in images]


class FakeSummarizeClient:
    """Offline drop-in for VLLMSummarizeClient — echoes a truncated summary."""

    def summarize(self, texts: list[str]) -> list[str]:
        return [f"summary: {t[:20]}" for t in texts]


def make_doc(audio_path: str, texts: list[str]) -> AudioMetadata:
    """One single-speech document with one chunk per text, 5s apart."""
    return AudioMetadata(
        audio_path=audio_path,
        sample_rate=16000,
        duration=float(len(texts) * 5),
        speeches=[
            SpeechSegment(
                speech_id=0,
                chunks=[
                    AudioChunk(start=i * 5, end=i * 5 + 5, text=t) for i, t in enumerate(texts)
                ],
            )
        ],
    )


def write_frames_aligned_to_chunks(db_path: Path) -> None:
    """Write one frame per chunk row, keyed to the chunks table so the
    frame→chunk join in visual/all search resolves."""
    keys = (
        lance.dataset(str(db_path / "chunks.lance"))
        .to_table(columns=["doc_id", "speech_id", "chunk_id"])
        .to_pylist()
    )
    frames = [
        ExtractedFrame(
            doc_id=k["doc_id"],
            speech_id=int(k["speech_id"]),
            chunk_id=int(k["chunk_id"]),
            frame_idx=0,
            time_sec=0.0,
            jpeg_bytes=f"frame-{i}".encode(),
            width=8,
            height=8,
        )
        for i, k in enumerate(keys)
    ]
    write_chunk_frames(db_path / "chunk_frames.lance", frames, create=True)


def write_synthetic_frames(frames_path: Path, n: int) -> list[bytes]:
    """Write ``n`` standalone frames (doc{i}/chunk{i}); returns their jpeg bytes."""
    jpegs = [f"JPEG-FRAME-{i}".encode() for i in range(n)]
    frames = [
        ExtractedFrame(
            doc_id=f"doc{i}",
            speech_id=0,
            chunk_id=i,
            time_sec=0.0,
            jpeg_bytes=jpegs[i],
            width=8,
            height=8,
        )
        for i in range(n)
    ]
    write_chunk_frames(frames_path, frames, create=True)
    return jpegs
