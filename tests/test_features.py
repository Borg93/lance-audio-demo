"""Type-agnostic feature engine + the concrete string-valued feature columns.

Proves the engine attaches *non-vector* columns (int, string) the same way it
attaches embeddings, and that the summary/caption features round-trip — no GPU,
deterministic fakes. The vector features are covered in ``test_embed.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from fakes import (
    FakeCaptionClient,
    FakeEmbedClient,
    FakeSummarizeClient,
    det_vector,
    make_doc,
    write_synthetic_frames,
)
from ratch.core.engine import upsert_scan_column
from ratch.features.columns import (
    CAPTION_COLUMN,
    CAPTION_EMBED_COLUMN,
    CHUNK_KEYS,
    FEATURES,
    SUMMARY_COLUMN,
    caption_column,
    embed_caption_column,
    summary_column,
)
from ratch.ingest.ingest import ingest_many


def _chunks(tmp_path: Path) -> Path:
    db = tmp_path / "t.lance"
    ingest_many(db, [make_doc("input/x.mp4", ["alpha beta gamma", "second chunk text here"])])
    return db / "chunks.lance"


class TestEngineTypeAgnostic:
    def test_add_int_scan_column_without_a_model(self, tmp_path: Path) -> None:
        # The engine derives any Arrow type from a pure compute — no client at all.
        path = _chunks(tmp_path)

        def compute(batch: pa.RecordBatch) -> pa.Array:
            return pa.array([len(t) for t in batch.column("text").to_pylist()], pa.int32())

        n = upsert_scan_column(
            path,
            name="text_len",
            output_type=pa.int32(),
            key_columns=CHUNK_KEYS,
            read_columns=["text"],
            compute=compute,
        )
        assert n == 2
        rows = lance.dataset(str(path)).to_table(columns=["text", "text_len"]).to_pylist()
        assert rows and all(r["text_len"] == len(r["text"]) for r in rows)


class TestSummaryColumn:
    def test_attaches_string_summary(self, tmp_path: Path) -> None:
        path = _chunks(tmp_path)
        assert summary_column(path, client=FakeSummarizeClient()) == 2
        rows = lance.dataset(str(path)).to_table(columns=["text", SUMMARY_COLUMN]).to_pylist()
        for r in rows:
            assert r[SUMMARY_COLUMN] == f"summary: {r['text'][:20]}"

    def test_second_pass_is_noop(self, tmp_path: Path) -> None:
        path = _chunks(tmp_path)
        summary_column(path, client=FakeSummarizeClient())
        assert summary_column(path, client=FakeSummarizeClient()) == 0


class TestCaptionColumn:
    def test_attaches_caption_keyed_by_rowid(self, tmp_path: Path) -> None:
        frames_path = tmp_path / "chunk_frames.lance"
        write_synthetic_frames(frames_path, n=4)
        assert caption_column(frames_path, client=FakeCaptionClient()) == 4

        ds = lance.dataset(str(frames_path))
        keyed = ds.to_table(columns=[CAPTION_COLUMN], with_row_id=True)
        for i, rowid in enumerate(keyed.column("_rowid").to_pylist()):
            blob = ds.take_blobs("frame_blob", ids=[rowid])[0]
            with blob as f:
                expected = f"caption of {len(f.read())} bytes"
            assert keyed.column(CAPTION_COLUMN)[i].as_py() == expected

    def test_second_pass_is_noop(self, tmp_path: Path) -> None:
        frames_path = tmp_path / "chunk_frames.lance"
        write_synthetic_frames(frames_path, n=3)
        caption_column(frames_path, client=FakeCaptionClient())
        assert caption_column(frames_path, client=FakeCaptionClient()) == 0

    def test_resumes_from_checkpoint(self, tmp_path: Path) -> None:
        # A killed run leaves a {ckpt}.values.jsonl sidecar; the resume must REUSE
        # those values (not recompute) and clean the sidecar up on success.
        frames_path = tmp_path / "chunk_frames.lance"
        write_synthetic_frames(frames_path, n=3)
        ckpt = tmp_path / "cap.ckpt"

        ds = lance.dataset(str(frames_path))
        rid0 = ds.to_table(columns=[], with_row_id=True).column("_rowid")[0].as_py()
        # Pre-seed row 0 with a sentinel the FakeCaptionClient would never produce.
        (tmp_path / "cap.ckpt.values.jsonl").write_text(
            json.dumps({"id": rid0, "v": "RESUMED"}) + "\n", encoding="utf-8"
        )

        assert caption_column(frames_path, client=FakeCaptionClient(), checkpoint_file=ckpt) == 3
        out = lance.dataset(str(frames_path)).to_table(columns=[CAPTION_COLUMN], with_row_id=True)
        by_id = {
            out.column("_rowid")[i].as_py(): out.column(CAPTION_COLUMN)[i].as_py()
            for i in range(out.num_rows)
        }
        assert by_id[rid0] == "RESUMED"  # taken from the checkpoint, not recomputed
        assert all(v for v in by_id.values())  # the other rows were captioned normally
        assert not (tmp_path / "cap.ckpt.values.jsonl").exists()  # cleaned up on success


class TestCaptionEmbeddingColumn:
    def _captioned_frames(self, tmp_path: Path, n: int) -> Path:
        frames_path = tmp_path / "chunk_frames.lance"
        write_synthetic_frames(frames_path, n=n)
        caption_column(frames_path, client=FakeCaptionClient())  # captions must exist first
        return frames_path

    def test_embeds_existing_captions(self, tmp_path: Path) -> None:
        # Reuses the frames — embeds the `caption` text, never re-reads frame_blob.
        frames_path = self._captioned_frames(tmp_path, n=4)
        assert embed_caption_column(frames_path, client=FakeEmbedClient()) == 4

        ds = lance.dataset(str(frames_path))
        assert CAPTION_EMBED_COLUMN in ds.schema.names
        rows = ds.to_table(columns=[CAPTION_COLUMN, CAPTION_EMBED_COLUMN]).to_pylist()
        for r in rows:
            stored = np.asarray(r[CAPTION_EMBED_COLUMN], dtype=np.float32)
            np.testing.assert_allclose(stored, det_vector(r[CAPTION_COLUMN].encode()), rtol=0, atol=1e-6)

    def test_requires_caption_column(self, tmp_path: Path) -> None:
        # Guard clause: embedding captions before captioning is a loud error.
        frames_path = tmp_path / "chunk_frames.lance"
        write_synthetic_frames(frames_path, n=2)
        with pytest.raises(ValueError, match=CAPTION_COLUMN):
            embed_caption_column(frames_path, client=FakeEmbedClient())

    def test_second_pass_is_noop(self, tmp_path: Path) -> None:
        frames_path = self._captioned_frames(tmp_path, n=3)
        embed_caption_column(frames_path, client=FakeEmbedClient())
        assert embed_caption_column(frames_path, client=FakeEmbedClient()) == 0


class TestUpsertModes:
    def test_only_null_fills_rows_added_by_a_later_ingest(self, tmp_path: Path) -> None:
        # The incremental top-up path: a later ingest appends NULL-summary rows;
        # the next pass fills only those via merge_insert on the chunk key.
        db = tmp_path / "t.lance"
        ingest_many(db, [make_doc("input/a.mp4", ["first text"])])
        path = db / "chunks.lance"
        assert summary_column(path, client=FakeSummarizeClient()) == 1

        ingest_many(db, [make_doc("input/b.mp4", ["second text"])])
        assert summary_column(path, client=FakeSummarizeClient()) == 1  # only the new row

        rows = lance.dataset(str(path)).to_table(columns=["text", SUMMARY_COLUMN]).to_pylist()
        assert all(r[SUMMARY_COLUMN] == f"summary: {r['text'][:20]}" for r in rows)

    def test_overwrite_rebuilds_every_row(self, tmp_path: Path) -> None:
        path = _chunks(tmp_path)
        summary_column(path, client=FakeSummarizeClient())
        assert summary_column(path, client=FakeSummarizeClient(), overwrite=True) == 2


class TestRegistry:
    def test_has_expected_features(self) -> None:
        # Membership, not an exact set: the registry grows over time, so we only
        # assert the core columns are wired (incl. the caption pair).
        assert {
            "text_embedding",
            "frame_embedding",
            "summary",
            "caption",
            "caption_embedding",
        } <= set(FEATURES)

    @pytest.mark.parametrize("feature", FEATURES.values(), ids=list(FEATURES))
    def test_feature_is_well_formed(self, feature) -> None:
        assert feature.table in {"chunks", "chunk_frames"}
        assert callable(feature.run)
        assert feature.description
