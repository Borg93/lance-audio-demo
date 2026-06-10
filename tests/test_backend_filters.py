"""WHERE-clause composition + value escaping — pure, no Lance.

The metadata filters (`language` / `namn` / `referenskod` / `extraid` / `topic`)
are inlined into a SQL string from user-supplied query params, so the quote
escaping is the injection boundary: a value carrying a ``'`` must be doubled, not
left to break out of the literal. ``raw`` is the deliberate exception — it is
user SQL ANDed in verbatim.
"""

from __future__ import annotations

import pytest
from backend.search.filters import _build_where_clause, _sql_quote


class TestSqlQuote:
    def test_doubles_single_quotes(self) -> None:
        assert _sql_quote("O'Brien") == "O''Brien"

    def test_doubles_every_quote(self) -> None:
        assert _sql_quote("a'b'c") == "a''b''c"

    def test_plain_value_unchanged(self) -> None:
        assert _sql_quote("Stockholm") == "Stockholm"


class TestBuildWhereClause:
    def test_no_filters_is_none(self) -> None:
        assert (
            _build_where_clause(language=None, namn=None, referenskod=None, extraid=None) is None
        )

    def test_language_is_exact_match(self) -> None:
        where = _build_where_clause(language="sv", namn=None, referenskod=None, extraid=None)
        assert where == "language = 'sv'"

    def test_namn_is_like_with_wildcards(self) -> None:
        where = _build_where_clause(language=None, namn="Westerberg", referenskod=None, extraid=None)
        assert where == "namn LIKE '%Westerberg%'"

    def test_value_with_quote_is_escaped_not_injectable(self) -> None:
        # The injection boundary: a ``'`` in the value is doubled so it stays
        # inside the literal instead of terminating it.
        where = _build_where_clause(language=None, namn="O'Brien", referenskod=None, extraid=None)
        assert where == "namn LIKE '%O''Brien%'"

    def test_multiple_filters_are_anded(self) -> None:
        where = _build_where_clause(language="en", namn=None, referenskod=None, extraid="T0001")
        assert where == "language = 'en' AND extraid = 'T0001'"

    def test_topic_matches_any_layer_column(self) -> None:
        where = _build_where_clause(
            language=None,
            namn=None,
            referenskod=None,
            extraid=None,
            topic="Sport",
            topic_columns=["topic_l0", "topic_l1"],
        )
        assert where == "(topic_l0 = 'Sport' OR topic_l1 = 'Sport')"

    def test_topic_without_columns_is_ignored(self) -> None:
        assert (
            _build_where_clause(
                language=None, namn=None, referenskod=None, extraid=None, topic="Sport"
            )
            is None
        )

    def test_raw_passes_through_verbatim_in_parens(self) -> None:
        # ``raw`` is user SQL, intentionally NOT quoted — wrapped in parens and ANDed.
        where = _build_where_clause(
            language="sv", namn=None, referenskod=None, extraid=None, raw="duration > 5"
        )
        assert where == "language = 'sv' AND (duration > 5)"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_raw_is_dropped(self, blank: str) -> None:
        assert (
            _build_where_clause(
                language=None, namn=None, referenskod=None, extraid=None, raw=blank
            )
            is None
        )
