"""P1.3: the stage registry declares shapes/gates correctly and MIME gates skip, never crash."""

from __future__ import annotations

import logging

import lance
import pyarrow as pa
import pytest

from ratch.core.driver import _gate_filter
from ratch.core.registry import (
    AUDIO_VIDEO,
    VIDEO_ONLY,
    ActorConfig,
    MediaGate,
    Stage,
    StageShape,
)
from ratch.features.stages import STAGES


class TestMediaGate:
    def test_admits_by_prefix(self) -> None:
        assert AUDIO_VIDEO.admits("audio/mpeg")
        assert AUDIO_VIDEO.admits("video/mp4")
        assert not AUDIO_VIDEO.admits("image/png")

    def test_none_mime_never_admitted(self) -> None:
        assert not VIDEO_ONLY.admits(None)


class TestStageValidation:
    def test_blob_stage_requires_blob_column(self) -> None:
        with pytest.raises(ValueError, match="blob_column is required"):
            Stage(
                name="x",
                shape=StageShape.BLOB_COLUMN,
                table="t",
                output_columns=("y",),
            )

    def test_append_stage_requires_output_table(self) -> None:
        with pytest.raises(ValueError, match="output_table is required"):
            Stage(name="x", shape=StageShape.APPEND_ROWS, table="t")

    def test_registry_stages_cover_all_shapes(self) -> None:
        shapes = {s.shape for s in STAGES.values()}
        assert shapes == set(StageShape)

    def test_every_per_row_stage_declares_actor_config(self) -> None:
        for stage in STAGES.values():
            assert isinstance(stage.actor, ActorConfig)
            assert stage.actor.max_actors >= stage.actor.min_actors >= 1


class TestMimeGateSkipsMixedBatch:
    """A gated stage against a mixed-MIME documents table skips (logged), never raises."""

    @pytest.fixture
    def mixed_db(self, tmp_path):
        docs = pa.table(
            {
                "doc_id": pa.array(["videodoc00000000", "audiodoc00000000", "imagedoc00000000"]),
                "media_mime": pa.array(["video/mp4", "audio/wav", "image/png"]),
            }
        )
        lance.write_dataset(docs, str(tmp_path / "documents.lance"))
        return tmp_path

    def test_video_gate_admits_only_video_and_logs_skips(self, mixed_db, caplog) -> None:
        stage = Stage(
            name="gated",
            shape=StageShape.APPEND_ROWS,
            table="chunks",
            key_columns=("doc_id",),
            output_table="out",
            media_gate=VIDEO_ONLY,
        )
        with caplog.at_level(logging.INFO):
            gate = _gate_filter(mixed_db, stage)
        assert gate == "doc_id IN ('videodoc00000000')"
        assert "skipped 2/3" in caplog.text

    def test_no_matching_docs_yields_empty_filter_not_crash(self, mixed_db) -> None:
        stage = Stage(
            name="gated",
            shape=StageShape.APPEND_ROWS,
            table="chunks",
            key_columns=("doc_id",),
            output_table="out",
            media_gate=MediaGate(mime_prefixes=("model/",)),
        )
        gate = _gate_filter(mixed_db, stage)
        assert gate == "doc_id IN ('')"

    def test_ungated_stage_has_no_filter(self, mixed_db) -> None:
        assert _gate_filter(mixed_db, STAGES["text_embedding"]) is None
