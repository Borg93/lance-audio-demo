"""Pure search_api tests — spec validation/clamping, descriptor-driven filter
compilation, mode-set derivation, and hit-projection derivation. No Lance, no
vLLM, no FastAPI app; everything here is constructible from plain values, and
every table/column name is deliberately NON-corpus (body / para_id / …) to pin
the schema-agnostic contract.
"""

from __future__ import annotations

import pytest

from common.lancekit.descriptor import (
    Declared,
    Display,
    FtsBinding,
    Identity,
    MetadataField,
    Search,
    TimeBinding,
    VectorBinding,
)
from common.lancekit.introspect import ColumnInfo, TableInfo
from search.services.filters import (
    build_where_clause,
    extract_filters,
    topic_layer_columns,
)
from search.services.spec import SearchMode, SearchSpec, available_modes
from search.services.target import hit_columns

# ── SearchSpec: validation + clamping ─────────────────────────────────────────


class TestSearchMode:
    def test_member_equals_its_string_value(self) -> None:
        # StrEnum members compare equal to their value, so the service layer's
        # `spec.mode == "fts"` comparisons are correct.
        assert SearchMode.FTS == "fts"

    def test_call_returns_member_singleton(self) -> None:
        assert SearchMode("hybrid") is SearchMode.HYBRID

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError):
            SearchMode("bogus")


class TestDefaults:
    def test_defaults(self) -> None:
        spec = SearchSpec()
        assert spec.q == ""
        assert spec.n == 20
        assert spec.mode == SearchMode.FTS  # str value "fts" (mode is str now)
        assert spec.fuzziness == 0
        assert spec.weight is None
        assert spec.rerank is False
        assert spec.phrase is False
        assert spec.rerank_n == 20


class TestClamping:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [(-5, 1), (0, 1), (1, 1), (50, 50), (200, 200), (201, 200), (10_000, 200)],
    )
    def test_n_clamped_to_1_200(self, given: int, expected: int) -> None:
        assert SearchSpec(n=given).n == expected

    @pytest.mark.parametrize(
        ("given", "expected"),
        [(0, 1), (1, 1), (20, 20), (200, 200), (500, 200)],
    )
    def test_rerank_n_clamped_to_1_200(self, given: int, expected: int) -> None:
        assert SearchSpec(rerank_n=given).rerank_n == expected

    @pytest.mark.parametrize(
        ("given", "expected"),
        [(-1, 0), (0, 0), (1, 1), (2, 2), (3, 2), (99, 2)],
    )
    def test_fuzziness_clamped_to_0_2(self, given: int, expected: int) -> None:
        assert SearchSpec(fuzziness=given).fuzziness == expected

    def test_weight_none_passes_through(self) -> None:
        assert SearchSpec(weight=None).weight is None

    @pytest.mark.parametrize(
        ("given", "expected"),
        [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.0)],
    )
    def test_weight_clamped_to_0_1(self, given: float, expected: float) -> None:
        assert SearchSpec(weight=given).weight == expected

    def test_whitespace_query_reads_as_empty(self) -> None:
        spec = SearchSpec(q="   ", q_vec=" \t ")
        assert spec.q == ""
        assert spec.q_vec == ""


class TestModeCoercion:
    def test_mode_is_a_plain_string(self) -> None:
        # `mode` is `str` (not the enum) so a dataset's OWN embedding keys are
        # accepted; the service validates availability + degrades to [].
        assert SearchSpec.model_validate({"mode": "hybrid"}).mode == "hybrid"

    def test_a_declared_role_key_passes_through(self) -> None:
        assert SearchSpec.model_validate({"mode": "scene"}).mode == "scene"

    def test_any_mode_string_accepted_at_the_spec(self) -> None:
        # No 422 boundary anymore — a mode the dataset can't serve yields no
        # results at the service (graceful), not a spec rejection.
        assert SearchSpec.model_validate({"mode": "bogus"}).mode == "bogus"


class TestExtraIgnored:
    def test_unknown_field_is_dropped(self) -> None:
        # Descriptor-declared filter params (and `dataset`) ride in the same
        # query string; the spec model must ignore, not reject, them.
        spec = SearchSpec.model_validate({"q": "x", "lang": "sv", "dataset": "articles"})
        assert spec.q == "x"
        assert not hasattr(spec, "lang")


# ── Mode set derivation from the descriptor ───────────────────────────────────


def _search(vectors: dict[str, VectorBinding] | None = None) -> Search:
    return Search(
        row_table="paras",
        fts=FtsBinding(table="paras", column="body", language="English"),
        vectors=vectors or {},
        filterable=["lang"],
    )


class TestAvailableModes:
    def test_minimal_search_has_fts_hybrid_all(self) -> None:
        assert available_modes(_search()) == [SearchMode.FTS, SearchMode.HYBRID, SearchMode.ALL]

    def test_vector_modes_present_iff_declared(self) -> None:
        search = _search(
            {
                "semantic": VectorBinding(
                    table="paras", column="body_vec", dim=16, query_encoder="text"
                ),
                "visual": VectorBinding(
                    table="shots", column="look_vec", dim=16, query_encoder="image"
                ),
            }
        )
        modes = available_modes(search)
        assert SearchMode.SEMANTIC in modes
        assert SearchMode.VISUAL in modes
        assert SearchMode.SCENE not in modes
        assert SearchMode.SCENE_FTS not in modes

    def test_scene_fts_requires_caption_source(self) -> None:
        without = _search(
            {"scene": VectorBinding(table="shots", column="blurb_vec", dim=16, query_encoder="text")}
        )
        assert SearchMode.SCENE_FTS not in available_modes(without)
        with_caption = _search(
            {
                "scene": VectorBinding(
                    table="shots",
                    column="blurb_vec",
                    dim=16,
                    query_encoder="text",
                    caption_source="blurb",
                )
            }
        )
        modes = available_modes(with_caption)
        assert SearchMode.SCENE in modes
        assert SearchMode.SCENE_FTS in modes


# ── Topic layer discovery ─────────────────────────────────────────────────────


class TestTopicLayerColumns:
    def test_sorted_by_layer_number(self) -> None:
        assert topic_layer_columns(["topic_l10", "topic_l2", "topic_l0", "body"]) == [
            "topic_l0",
            "topic_l2",
            "topic_l10",
        ]

    def test_non_numeric_suffix_skipped(self) -> None:
        assert topic_layer_columns(["topic_label", "topic_l1"]) == ["topic_l1"]

    def test_empty_when_no_layers(self) -> None:
        assert topic_layer_columns(["body", "para_id"]) == []


# ── Filter extraction + WHERE compilation ─────────────────────────────────────


class TestExtractFilters:
    def test_picks_only_declared_names(self) -> None:
        params = {"lang": "sv", "kind": "news", "q": "x", "evil": "1"}
        assert extract_filters(params, ["lang", "kind"]) == {"lang": "sv", "kind": "news"}

    def test_empty_values_read_as_unset(self) -> None:
        assert extract_filters({"lang": ""}, ["lang"]) == {}

    def test_missing_params_are_absent(self) -> None:
        assert extract_filters({}, ["lang", "kind"]) == {}


class TestBuildWhereClause:
    def test_no_filters_is_none(self) -> None:
        assert build_where_clause(filters={}, filterable=["lang", "kind"]) is None

    def test_filterable_field_is_equality(self) -> None:
        where = build_where_clause(filters={"lang": "sv"}, filterable=["lang"])
        assert where == "lang = 'sv'"

    def test_value_with_quote_is_escaped_not_injectable(self) -> None:
        # The injection boundary: a ``'`` in the value is doubled so it stays
        # inside the literal instead of terminating it.
        where = build_where_clause(filters={"source": "O'Brien"}, filterable=["source"])
        assert where == "source = 'O''Brien'"

    def test_multiple_filters_are_anded_in_declared_order(self) -> None:
        where = build_where_clause(
            filters={"kind": "news", "lang": "en"}, filterable=["lang", "kind"]
        )
        assert where == "lang = 'en' AND kind = 'news'"

    def test_undeclared_filter_is_ignored(self) -> None:
        assert build_where_clause(filters={"evil": "x"}, filterable=["lang"]) is None

    def test_topic_matches_any_layer_column(self) -> None:
        where = build_where_clause(
            filters={"topic": "Sport"},
            filterable=["topic"],
            topic_columns=["topic_l0", "topic_l1"],
        )
        assert where == "(topic_l0 = 'Sport' OR topic_l1 = 'Sport')"

    def test_topic_without_layer_columns_is_ignored(self) -> None:
        assert (
            build_where_clause(filters={"topic": "Sport"}, filterable=["topic"], topic_columns=())
            is None
        )

    def test_raw_passes_through_verbatim_in_parens(self) -> None:
        # ``raw`` is user SQL, intentionally NOT quoted — wrapped in parens and ANDed.
        where = build_where_clause(
            filters={"lang": "sv"}, filterable=["lang"], raw="duration > 5"
        )
        assert where == "lang = 'sv' AND (duration > 5)"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_raw_is_dropped(self, blank: str) -> None:
        assert build_where_clause(filters={}, filterable=[], raw=blank) is None


# ── Hit projection derivation ─────────────────────────────────────────────────


def _col(name: str) -> ColumnInfo:
    return ColumnInfo(name=name, arrow_type="string", nullable=True)


def _table(names: list[str]) -> TableInfo:
    return TableInfo(name="paras", row_count=3, version=1, columns=[_col(n) for n in names], indexes=[])


def _declared() -> Declared:
    return Declared(
        identity=Identity(key_fields=["para_id", "seg_id"], doc_key="para_id"),
        time=TimeBinding(start="t0", end="t1"),
        display=Display(
            title=["source"],
            body="body",
            metadata=[
                MetadataField(field="lang", label="Language"),
                MetadataField(field="source", label="Source"),
                MetadataField(field="ghost", label="Not on the table"),
            ],
        ),
        search=_search(),
    )


class TestHitColumns:
    def test_derives_keys_time_duration_body_and_metadata(self) -> None:
        live = ["para_id", "seg_id", "t0", "t1", "duration", "body", "source", "lang", "extra"]
        cols = hit_columns(_declared(), _table(live))
        assert cols == ["para_id", "seg_id", "t0", "t1", "duration", "body", "source", "lang"]

    def test_absent_declared_fields_never_break_the_select(self) -> None:
        # 'ghost' is declared metadata but missing from the live table; 'duration'
        # is a convenience column this table doesn't have. Both must drop out.
        live = ["para_id", "seg_id", "t0", "t1", "body", "lang", "source"]
        cols = hit_columns(_declared(), _table(live))
        assert "ghost" not in cols
        assert "duration" not in cols

    def test_title_and_metadata_overlap_dedupes(self) -> None:
        live = ["para_id", "seg_id", "t0", "t1", "body", "source", "lang"]
        cols = hit_columns(_declared(), _table(live))
        assert cols.count("source") == 1

    def test_no_time_binding_skips_time_fields(self) -> None:
        declared = _declared()
        declared.time = None
        cols = hit_columns(declared, _table(["para_id", "seg_id", "t0", "t1", "body"]))
        assert "t0" not in cols
        assert "t1" not in cols
