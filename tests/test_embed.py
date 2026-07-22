"""No-GPU, no-network tests for the `add_columns` embedding pipeline.

These exercise the *real* Lance data-evolution path (`dataset.add_columns` with
a `batch_udf`) against a tiny on-disk dataset, using a deterministic fake
embedding client. They prove the wiring that the GPU pipeline relies on:

* `embed_text_column` attaches `text_embedding` and the values round-trip.
* `embed_frame_column` keys vectors by `_rowid`, so each row gets *its own*
  frame's embedding regardless of scan order (the bug the refactor fixed).
* a brute-force vector search (no index) returns the planted nearest neighbour.

Run:  uv run pytest tests/test_embed.py -v
"""

from __future__ import annotations

import lance
import numpy as np
import pytest

from fakes import FakeEmbedClient, det_vector, make_doc, write_synthetic_frames
from ratch.features.columns import (
    FRAME_EMBED_COLUMN,
    TEXT_EMBED_COLUMN,
    embed_frame_column,
    embed_text_column,
)
from ratch.ingest.ingest import ingest_many
from ratch.model.schema import EMBED_DIM


class TestEmbedTextColumn:
    def test_add_columns_attaches_and_roundtrips(self, tmp_path):
        db = tmp_path / "te.lance"
        texts = ["the quick brown fox", "a slow green turtle", "klimat och miljö"]
        ingest_many(db, [make_doc("input/te.mp4", texts)])
        chunks_path = db / "chunks.lance"

        # Column absent before embedding.
        assert TEXT_EMBED_COLUMN not in lance.dataset(str(chunks_path)).schema.names

        n = embed_text_column(chunks_path, client=FakeEmbedClient())
        assert n == len(texts)

        ds = lance.dataset(str(chunks_path))
        assert TEXT_EMBED_COLUMN in ds.schema.names
        tbl = ds.to_table(columns=["text", TEXT_EMBED_COLUMN])
        stored = {
            row["text"]: np.asarray(row[TEXT_EMBED_COLUMN], dtype=np.float32)
            for row in tbl.to_pylist()
        }
        assert set(stored) == set(texts)
        for t in texts:
            assert stored[t].shape == (EMBED_DIM,)
            np.testing.assert_allclose(stored[t], det_vector(t.encode()), rtol=0, atol=1e-6)

    def test_only_null_is_noop_once_embedded(self, tmp_path):
        db = tmp_path / "noop.lance"
        ingest_many(db, [make_doc("input/noop.mp4", ["alpha", "beta"])])
        chunks_path = db / "chunks.lance"
        embed_text_column(chunks_path, client=FakeEmbedClient())
        # Second pass with default (overwrite=False) embeds nothing new.
        assert embed_text_column(chunks_path, client=FakeEmbedClient()) == 0

    def test_brute_force_vector_search_finds_planted_nearest(self, tmp_path):
        db = tmp_path / "search.lance"
        texts = ["alpha unique", "beta distinct", "gamma separate"]
        ingest_many(db, [make_doc("input/s.mp4", texts)])
        embed_text_column(db / "chunks.lance", client=FakeEmbedClient())

        # Brute-force (no index) cosine search via pylance's `nearest=` API.
        ds = lance.dataset(str(db / "chunks.lance"))
        query = FakeEmbedClient().embed_text(["beta distinct"])[0].tolist()
        res = ds.to_table(nearest={"column": TEXT_EMBED_COLUMN, "q": query, "k": 1})
        assert res.num_rows == 1
        assert res.column("text")[0].as_py() == "beta distinct"


class TestEmbedFrameColumn:
    def test_frame_embedding_keyed_by_rowid_not_position(self, tmp_path):
        # The whole point of the _rowid-keyed UDF: each row must get the
        # embedding of ITS OWN frame, not whatever happened to be at that
        # scan position. Verify per-row by reading each row's blob back.
        frames_path = tmp_path / "chunk_frames.lance"
        write_synthetic_frames(frames_path, n=5)

        n = embed_frame_column(frames_path, client=FakeEmbedClient(), batch_rows=2)
        assert n == 5

        ds = lance.dataset(str(frames_path))
        assert FRAME_EMBED_COLUMN in ds.schema.names
        keyed = ds.to_table(columns=[FRAME_EMBED_COLUMN], with_row_id=True)
        rowids = keyed.column("_rowid").to_pylist()
        for i, rowid in enumerate(rowids):
            stored = np.asarray(keyed.column(FRAME_EMBED_COLUMN)[i].as_py(), dtype=np.float32)
            blob = ds.take_blobs("frame_blob", ids=[rowid])[0]
            with blob as f:
                expected = det_vector(f.read())
            np.testing.assert_allclose(stored, expected, rtol=0, atol=1e-6)

    def test_overwrite_false_is_noop_when_present(self, tmp_path):
        frames_path = tmp_path / "chunk_frames.lance"
        write_synthetic_frames(frames_path, n=3)
        embed_frame_column(frames_path, client=FakeEmbedClient())
        assert embed_frame_column(frames_path, client=FakeEmbedClient()) == 0


@pytest.mark.parametrize("repeat", [2, 3])
def test_reingest_same_doc_is_idempotent(tmp_path, repeat):
    db = tmp_path / "idem.lance"
    doc = make_doc("input/idem.mp4", ["alpha beta", "gamma delta"])
    for _ in range(repeat):
        table = ingest_many(db, [doc])
    assert table.count_rows() == 2  # re-ingest replaces, never duplicates
