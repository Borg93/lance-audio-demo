"""Pure data-hygiene helpers in ``scripts/kg/adapter.py``.

These functions encode every cleanup rule derived from the graph audits
(junk-drop, type taxonomy, alias-safe slugging, word-boundary descriptions).
They are deterministic and dependency-light, so they're locked in here as a
regression guard. ``scripts/kg`` is not a package — load the module by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "kg_adapter", Path(__file__).resolve().parent.parent / "scripts" / "kg" / "adapter.py"
)
assert _SPEC and _SPEC.loader
adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapter)  # safe: argparse lives inside main()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("person", "PERSON"),
        ("organization", "ORG"),
        ("location", "GEO"),
        ("event", "EVENT"),
        ("concept", "CONCEPT"),
        ("method", "CONCEPT"),
        ("content", "WORK"),
        ("artifact", "WORK"),
        ("data", "WORK"),
        ("naturalobject", "OTHER"),
        ("creature", "OTHER"),
        ("", "OTHER"),
        (None, "OTHER"),
    ],
)
def test_norm_type_covers_full_taxonomy(raw: str | None, expected: str) -> None:
    assert adapter.norm_type(raw) == expected


@pytest.mark.parametrize(
    "junk",
    [
        "Jag",  # pronoun stopword
        "1998",  # bare year
        "4 %",  # amount
        "55 Miljoner",  # amount w/ unit
        "(2007-2011)",  # no alphabetic char
        "...",  # punctuation only
        "x",  # too short
        "a" * 95,  # absurdly long
        "Om dina tänder är gula eller bleka, är det inget att le om",  # sentence
    ],
)
def test_is_junk_drops_garbage(junk: str) -> None:
    assert adapter.is_junk(junk) is True


@pytest.mark.parametrize(
    "keep",
    [
        "EU",  # acronym
        "BRÅ",  # Swedish acronym
        "Göran Persson",  # real person
        "Sverige",  # place
        "Utredningen Om Allmänhetens Insyn I Partiers Och Valkandidaters Intäkter",  # long but title-case org
    ],
)
def test_is_junk_keeps_real_entities(keep: str) -> None:
    assert adapter.is_junk(keep) is False


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("<Chirac>", "Chirac"),
        (",Kommuner", "Kommuner"),
        ("::Staten", "Staten"),
        ("Energi\\", "Energi"),
        ("  Sverige  ", "Sverige"),
    ],
)
def test_clean_name_strips_artifacts(raw: str, clean: str) -> None:
    assert adapter.clean_name(raw) == clean
    # the cleaned form must slug identically to the already-clean entity
    assert adapter.slug(adapter.clean_name(raw)) == adapter.slug(clean)


def test_slug_is_deterministic_and_case_insensitive() -> None:
    assert adapter.slug("Stockholm") == adapter.slug("  stockholm ")
    assert len(adapter.slug("anything")) == 16


def test_truncate_desc_word_boundary() -> None:
    long = (
        "Relationen mellan EU och Sverige definieras primärt av att Sverige är en "
        "fullvärdig medlem i EU och har en etablerad ansvarsfördelning mellan parterna "
        "som påverkar både inrikes- och utrikespolitiken under hela perioden"
    )
    assert len(long) > 160  # guard: the case is only meaningful when over budget
    out = adapter.truncate_desc(long)
    assert len(out) <= 160
    assert out.endswith("…")
    assert "  " not in out  # <SEP>-join whitespace collapsed
    assert out[:-1] == out[:-1].rstrip()  # no space before the ellipsis


def test_truncate_desc_passthrough_when_short() -> None:
    assert adapter.truncate_desc("Regeringen styr Sverige.") == "Regeringen styr Sverige."


# --- generic-noun classifier (scripts/kg/generic_sv.py), reached via adapter ---


@pytest.mark.parametrize(
    "name",
    [
        # PERSON-typed group/role nouns
        "Politiker", "Invandrare", "Konsumenter", "Unga Människor", "De Anhöriga",
        "Justitieministern", "VD", "ÖB", "Försvarsmakten ÖB", "Utredaren",
    ],
)
def test_generic_person_catches_groups_and_roles(name: str) -> None:
    assert adapter.is_generic_person(name) is True


@pytest.mark.parametrize(
    "name",
    ["Anders", "Lars Eriksson", "Göran Persson", "Eva", "Carl Bildt", "Ingvar Kamprad"],
)
def test_generic_person_keeps_real_names(name: str) -> None:
    assert adapter.is_generic_person(name) is False


@pytest.mark.parametrize(
    "name",
    [  # GEO / WORK / EVENT / ORG category nouns, not named entities
        "Arbetsplats", "Skolan", "Bankkontor", "Rapporten", "Propositionen",
        "Betänkandet", "Möte", "Debatten", "Presskonferens", "Företaget", "Nämnden",
    ],
)
def test_generic_common_catches_categories(name: str) -> None:
    assert adapter.is_generic_common(name) is True


@pytest.mark.parametrize(
    "name",
    [  # real named places/works/events/orgs — incl. definite-form laws/regions
        "Sverige", "Stockholm", "Norrland", "Miljöbalken", "Kyotoprotokollet",
        "Volvo", "Försvarsmakten", "Riksbanken", "Saltsjöbadsavtalet", "EU",
    ],
)
def test_generic_common_keeps_named_entities(name: str) -> None:
    assert adapter.is_generic_common(name) is False


def test_generic_classifiers_are_deterministic() -> None:
    # byte-stable: same input -> same verdict, no LLM, no randomness
    for name in ("Politiker", "Arbetsplats", "Lars Eriksson", "Sverige"):
        assert adapter.is_generic_person(name) == adapter.is_generic_person(name)
        assert adapter.is_generic_common(name) == adapter.is_generic_common(name)
