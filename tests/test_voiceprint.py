"""Pure-logic tests for the voiceprint pipeline — no model, network, or GPU.

Covers the math `embed-speaker-turns` / `build-speakers` lean on: the
duration-weighted centroid (weighting, renormalization, degenerate input), the
turn→sample slice clamping, the duration-sorted batching, and the
filter/slice/reassemble orchestration in :func:`embed_turn_slices` — the last
exercised against a fake encoder that encodes each slice's *length*, so row↔turn
alignment is verifiable across the sorted batches.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from runners.voiceprint.voiceprint import (
    TurnSpan,
    clamped_sample_span,
    duration_sorted_batches,
    duration_weighted_centroid,
    embed_turn_slices,
    load_wav_16k_mono,
)

from ratch.model.schema import VOICE_EMBED_DIM

SR = 16_000


def turn(turn_id: int, start: float, end: float) -> TurnSpan:
    return TurnSpan(turn_id=turn_id, speaker_label="SPEAKER_00", start=start, end=end)


class FakeTurnEncoder:
    """Offline stand-in for VoiceEncoder.embed_batch — encodes each slice's length.

    Batch row i is ``[len_i, 1]``; after the orchestrator's L2 normalization the
    length is recoverable as ``out[i, 0] / out[i, 1]``, so tests can pin which
    slice produced which output row. Records every batch it sees.
    """

    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def embed_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        self.batches.append([int(w.shape[0]) for w in waveforms])
        return np.array([[float(w.shape[0]), 1.0] for w in waveforms], dtype=np.float32)


def recovered_lengths(out: np.ndarray) -> list[int]:
    """Invert the fake encoding: L2-normalizing [L, 1] preserves the ratio L/1."""
    return [round(float(row[0] / row[1])) for row in out]


class TestDurationWeightedCentroid:
    def test_weights_by_duration(self) -> None:
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0], dtype=np.float32)
        centroid = duration_weighted_centroid(np.stack([e1, e2]), [3.0, 1.0])
        expected = np.array([3.0, 1.0]) / np.sqrt(10.0)
        np.testing.assert_allclose(centroid, expected, rtol=1e-6)

    def test_result_is_renormalized_to_unit(self) -> None:
        # Equal-weight orthogonal unit vectors average to norm 1/√2 → must be rescaled.
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        centroid = duration_weighted_centroid(vectors, [2.0, 2.0])
        assert float(np.linalg.norm(centroid)) == pytest.approx(1.0)
        np.testing.assert_allclose(centroid, np.array([1.0, 1.0]) / np.sqrt(2.0), rtol=1e-6)

    def test_single_turn_returns_the_normalized_vector(self) -> None:
        centroid = duration_weighted_centroid(np.array([[3.0, 4.0]], dtype=np.float32), [7.5])
        np.testing.assert_allclose(centroid, [0.6, 0.8], rtol=1e-6)
        assert centroid.dtype == np.float32

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            duration_weighted_centroid(np.zeros((0, 4), dtype=np.float32), [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            duration_weighted_centroid(np.eye(2, dtype=np.float32), [1.0])

    def test_non_positive_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            duration_weighted_centroid(np.eye(2, dtype=np.float32), [1.0, 0.0])

    def test_cancelling_embeddings_raise(self) -> None:
        vectors = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="degenerate"):
            duration_weighted_centroid(vectors, [1.0, 1.0])


class TestClampedSampleSpan:
    def test_maps_seconds_to_samples(self) -> None:
        assert clamped_sample_span(1.0, 2.5, sample_rate=SR, n_samples=10 * SR) == (SR, 40_000)

    def test_clamps_negative_start_to_zero(self) -> None:
        assert clamped_sample_span(-0.5, 1.0, sample_rate=SR, n_samples=10 * SR) == (0, SR)

    def test_clamps_end_to_wav_length(self) -> None:
        assert clamped_sample_span(9.0, 12.0, sample_rate=SR, n_samples=10 * SR) == (
            9 * SR,
            10 * SR,
        )


class TestDurationSortedBatches:
    def test_partitions_all_indices_in_length_order(self) -> None:
        batches = duration_sorted_batches([40, 10, 30, 20, 50], batch_size=2)
        assert batches == [[1, 3], [2, 0], [4]]

    def test_batch_size_one(self) -> None:
        assert duration_sorted_batches([2, 1], batch_size=1) == [[1], [0]]

    def test_batch_size_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            duration_sorted_batches([1], batch_size=0)


class TestEmbedTurnSlices:
    def test_skips_turns_shorter_than_min_duration(self) -> None:
        wav = np.zeros(10 * SR, dtype=np.float32)
        turns = [turn(0, 0.0, 2.0), turn(1, 3.0, 3.2), turn(2, 4.0, 7.0)]
        kept, out = embed_turn_slices(FakeTurnEncoder(), wav, turns, min_duration=0.5)
        assert [t.turn_id for t in kept] == [0, 2]
        assert out.shape[0] == 2

    def test_skips_spans_clamped_too_short_by_wav_end(self) -> None:
        # 1.2 s nominal duration, but only 0.2 s of it exists in the wav.
        wav = np.zeros(10 * SR, dtype=np.float32)
        kept, _ = embed_turn_slices(FakeTurnEncoder(), wav, [turn(0, 9.8, 11.0)], min_duration=0.5)
        assert kept == []

    def test_all_short_returns_empty_with_voice_dim(self) -> None:
        wav = np.zeros(SR, dtype=np.float32)
        kept, out = embed_turn_slices(FakeTurnEncoder(), wav, [turn(0, 0.0, 0.1)], min_duration=0.5)
        assert kept == []
        assert out.shape == (0, VOICE_EMBED_DIM)
        assert out.dtype == np.float32

    def test_rows_align_with_kept_turns_across_sorted_batches(self) -> None:
        # Deliberately NOT in duration order, so reassembly is actually exercised.
        wav = np.zeros(10 * SR, dtype=np.float32)
        turns = [
            turn(0, 2.0, 5.0),  # 48000 samples
            turn(1, 0.0, 1.0),  # 16000
            turn(2, 5.0, 7.0),  # 32000
            turn(3, 8.0, 8.25),  # skipped (< 0.5 s)
            turn(4, 7.0, 7.6),  # 9600
        ]
        encoder = FakeTurnEncoder()
        kept, out = embed_turn_slices(encoder, wav, turns, batch_size=2, min_duration=0.5)

        assert [t.turn_id for t in kept] == [0, 1, 2, 4]
        # Row i must encode kept[i]'s slice length, in original turn order.
        assert recovered_lengths(out) == [48000, 16000, 32000, 9600]
        # The encoder saw ≤ batch_size slices per call, shortest-first.
        assert all(len(b) <= 2 for b in encoder.batches)
        flattened = [length for b in encoder.batches for length in b]
        assert flattened == sorted(flattened) == [9600, 16000, 32000, 48000]

    def test_rows_are_l2_normalized(self) -> None:
        wav = np.zeros(4 * SR, dtype=np.float32)
        _, out = embed_turn_slices(FakeTurnEncoder(), wav, [turn(0, 0.0, 1.0), turn(1, 1.0, 4.0)])
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], rtol=1e-6)


class TestLoadWav:
    def write_wav(self, path: Path, *, rate: int, samples: np.ndarray) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(samples.astype("<i2").tobytes())

    def test_round_trip_pcm16_scaling(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.wav"
        self.write_wav(path, rate=SR, samples=np.array([0, 16384, -32768], dtype=np.int16))
        wav = load_wav_16k_mono(path)
        assert wav.dtype == np.float32
        np.testing.assert_allclose(wav, [0.0, 0.5, -1.0])

    def test_rejects_wrong_sample_rate(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.wav"
        self.write_wav(path, rate=8_000, samples=np.zeros(8, dtype=np.int16))
        with pytest.raises(ValueError, match="16000 Hz mono PCM16"):
            load_wav_16k_mono(path)
