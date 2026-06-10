"""SQL WHERE-clause composition + value escaping — pure string logic, no Lance.

No exceptions, no Lance handles: just builds the SQL the retrieval modules push
down as a filter.
"""

from __future__ import annotations


def _sql_quote(value: str) -> str:
    """Escape single quotes for inlining a value in a SQL string literal."""
    return value.replace("'", "''")


def _build_where_clause(
    *,
    language: str | None,
    namn: str | None,
    referenskod: str | None,
    extraid: str | None,
    topic: str | None = None,
    topic_columns: list[str] | None = None,
    raw: str | None = None,
) -> str | None:
    """Compose the SQL WHERE clause for metadata filters.

    ``raw`` is a user-typed SQL expression ANDed in verbatim (wrapped in parens)
    — intentionally *not* quoted, since it is meant to be SQL, not a value.
    ``topic`` exact-matches any of ``topic_columns`` (the nested topic_l* layers),
    so a treemap node at any depth filters the chunks tagged with that name.
    """
    clauses: list[str] = []
    if language:
        clauses.append(f"language = '{_sql_quote(language)}'")
    if namn:
        clauses.append(f"namn LIKE '%{_sql_quote(namn)}%'")
    if referenskod:
        clauses.append(f"referenskod LIKE '%{_sql_quote(referenskod)}%'")
    if extraid:
        clauses.append(f"extraid = '{_sql_quote(extraid)}'")
    if topic and topic_columns:
        ors = " OR ".join(f"{col} = '{_sql_quote(topic)}'" for col in topic_columns)
        clauses.append(f"({ors})")
    if raw and raw.strip():
        clauses.append(f"({raw})")
    return " AND ".join(clauses) if clauses else None
