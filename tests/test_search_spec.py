"""Validation + clamping for the normalized SearchSpec request model.

Pure, no Lance/vLLM. ``n`` / ``fuzziness`` / ``weight`` clamp (so the frontend's
"load more" can over-shoot without erroring); ``mode`` is a ``StrEnum`` that
rejects unknown values at construction.
"""

from __future__ import annotations

import pytest
from backend.search.spec import SearchMode, SearchSpec
from pydantic import ValidationError


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
        assert spec.mode is SearchMode.FTS
        assert spec.fuzziness == 0
        assert spec.weight is None
        assert spec.rerank is False
        assert spec.phrase is False


class TestClampN:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [(-5, 1), (0, 1), (1, 1), (50, 50), (200, 200), (201, 200), (10_000, 200)],
    )
    def test_clamped_to_1_200(self, given: int, expected: int) -> None:
        assert SearchSpec(n=given).n == expected


class TestClampRerankN:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [(0, 1), (1, 1), (20, 20), (200, 200), (500, 200)],
    )
    def test_clamped_to_1_200(self, given: int, expected: int) -> None:
        assert SearchSpec(rerank_n=given).rerank_n == expected

    def test_default_is_20(self) -> None:
        assert SearchSpec().rerank_n == 20


class TestClampFuzziness:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [(-1, 0), (0, 0), (1, 1), (2, 2), (3, 2), (99, 2)],
    )
    def test_clamped_to_0_2(self, given: int, expected: int) -> None:
        assert SearchSpec(fuzziness=given).fuzziness == expected


class TestClampWeight:
    def test_none_passes_through(self) -> None:
        assert SearchSpec(weight=None).weight is None

    @pytest.mark.parametrize(
        ("given", "expected"),
        [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.0)],
    )
    def test_clamped_to_0_1(self, given: float, expected: float) -> None:
        assert SearchSpec(weight=given).weight == expected


class TestMode:
    def test_string_coerces_to_enum(self) -> None:
        # FastAPI hands the raw query string in; Pydantic coerces it to the enum.
        assert SearchSpec.model_validate({"mode": "hybrid"}).mode is SearchMode.HYBRID

    def test_scene_mode_coerces(self) -> None:
        assert SearchSpec.model_validate({"mode": "scene"}).mode is SearchMode.SCENE

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchSpec.model_validate({"mode": "bogus"})


class TestExtraIgnored:
    def test_unknown_field_is_dropped(self) -> None:
        spec = SearchSpec.model_validate({"q": "x", "surprise": "ignored"})
        assert spec.q == "x"
        assert not hasattr(spec, "surprise")
