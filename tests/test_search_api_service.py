"""search_api against real tmp Lance datasets — service routing, degradation,
frame joins, HTTP wire shape, and the lazy client accessors.

Everything here is deliberately built with NON-corpus table/column names
(``paras.body`` / ``para_id`` / ``shots.blurb`` …) and a 2-field identity key,
proving the port is schema-agnostic: the service learns every name from the
descriptor at runtime. Two datasets share one registry root:

- ``articles`` — full bindings (fts + semantic + visual + scene w/ captions)
- ``docs``     — fts only (no vectors, no frame table): the degradation probe

Fakes are local by design (tests/fakes.py belongs to the old rmedia-importing
suite and stays untouched).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar, Never, cast

import lancedb
import numpy as np
import pyarrow as pa
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from common.core.config import Settings
from common.core.exceptions import ServiceUnavailableError
from common.core.handlers import register_handlers
from common.state import AppState, dataset_handle
from search.api.v1.router import router as search_router
from search.services import clients
from search.services.frames import frame_search
from search.services.postprocess import parse_alignments_json, rrf_fuse
from search.services.service import run_search
from search.services.spec import SearchMode, SearchSpec
from search.services.target import resolve_target
from search.services.vector import vector_search

DIM = 16

PARAS = {
    "climate": "The minister announced new targets to cut carbon emissions and fight global warming.",
    "sports": "The national football team won the championship final after a penalty shootout.",
    "economy": "Inflation rose as the central bank raised interest rates to cool the economy.",
}

# ── Local offline fakes ───────────────────────────────────────────────────────


def _det_vec(key: str) -> np.ndarray:
    """Deterministic, unique, L2-normalized vector (cosine-ready)."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


class FakeEmbedClient:
    """Offline embedder: each exact string maps to its own unit vector."""

    def embed_text(self, texts: list[str]) -> np.ndarray:
        return np.stack([_det_vec(t) for t in texts])

    def embed_image(self, images: list[bytes]) -> np.ndarray:
        return np.stack([_det_vec(f"img:{len(b)}") for b in images])


class FakeReranker:
    """Offline reranker: scores by candidate length, no network."""

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        return [float(len(c)) for c in candidates]


def _embedder() -> FakeEmbedClient:
    return FakeEmbedClient()


def _reranker() -> FakeReranker:
    return FakeReranker()


def _no_reranker() -> Never:
    raise AssertionError("reranker must not be constructed on these paths")


def _no_embedder() -> Never:
    raise AssertionError("embedder must not be constructed on keyword-only paths")


# ── Dataset builders (non-corpus names throughout) ────────────────────────────


def _vec_column(vecs: list[np.ndarray]) -> pa.Array:
    return pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32(), DIM))


def _fts_index(tbl, column: str) -> None:
    # Same config the pipeline's ensure_fts_index uses (with_position enables
    # phrase queries; stop words kept so common-word queries stay testable).
    from lancedb.index import FTS

    tbl.create_index(
        column,
        replace=True,
        config=FTS(with_position=True, remove_stop_words=False, language="English"),
    )


ARTICLES_DESCRIPTOR = {
    "identity": {"key_fields": ["para_id", "seg_id"], "doc_key": "para_id"},
    "time": {"start": "t0", "end": "t1"},
    "display": {
        "title": ["source"],
        "body": "body",
        "metadata": [
            {"field": "lang", "label": "Language"},
            {"field": "source", "label": "Source"},
            {"field": "kind", "label": "Kind"},
        ],
    },
    "search": {
        "row_table": "paras",
        "fts": {"table": "paras", "column": "body", "language": "English"},
        "vectors": {
            "semantic": {
                "table": "paras",
                "column": "body_vec",
                "dim": DIM,
                "query_encoder": "text",
            },
            "visual": {
                "table": "shots",
                "column": "look_vec",
                "dim": DIM,
                "query_encoder": "image",
            },
            "scene": {
                "table": "shots",
                "column": "blurb_vec",
                "dim": DIM,
                "query_encoder": "text",
                "caption_source": "blurb",
            },
            # A space the search box CAN'T drive: its query encoder is neither
            # text nor image (an audio voiceprint), so a text/image query has no
            # matching vector. It's a real declared space, but must degrade to
            # empty (not 400) and never abort an 'all' fusion.
            "voiceprint": {
                "table": "paras",
                "column": "body_vec",
                "dim": DIM,
                "query_encoder": "audio",
            },
        },
        "filterable": ["lang", "kind", "topic"],
        "rerank": True,
    },
    "capabilities": {"alignments": "paras.align_json"},
}

DOCS_DESCRIPTOR = {
    "identity": {"key_fields": ["para_id", "seg_id"], "doc_key": "para_id"},
    "time": {"start": "t0", "end": "t1"},
    "display": {"body": "body", "metadata": [{"field": "lang", "label": "Language"}]},
    "search": {
        "row_table": "paras",
        "fts": {"table": "paras", "column": "body", "language": "English"},
        "vectors": {},
        "filterable": ["lang"],
    },
    "capabilities": {"alignments": "paras.align_json"},
}


def _build_articles(root: Path, ddir: Path) -> None:
    conn = lancedb.connect(str(root / "articles.lance"))
    names = list(PARAS)
    para_ids = [f"p-{n}" for n in names]
    bodies = [PARAS[n] for n in names]
    paras = pa.table(
        {
            "para_id": pa.array(para_ids),
            "seg_id": pa.array([0, 0, 0], type=pa.int64()),
            "t0": pa.array([0.0, 5.0, 10.0], type=pa.float64()),
            "t1": pa.array([5.0, 10.0, 15.0], type=pa.float64()),
            "body": pa.array(bodies),
            "source": pa.array([f"{n}.mp4" for n in names]),
            "lang": pa.array(["sv", "sv", "en"]),
            "kind": pa.array(names),
            "topic_l0": pa.array(["Environment", "Leisure", "Money"]),
            "topic_l1": pa.array(["Climate", "Sports", "Economy"]),
            "align_json": pa.array(['[{"word": "hi", "start": 0.0, "end": 0.5}]'] * 3),
            "body_vec": _vec_column([_det_vec(b) for b in bodies]),
        }
    )
    tbl = conn.create_table("paras", data=paras)
    _fts_index(tbl, "body")

    blurbs = [f"blurb showing {n} footage" for n in names]
    shots = pa.table(
        {
            "para_id": pa.array(para_ids),
            "seg_id": pa.array([0, 0, 0], type=pa.int64()),
            "frame_idx": pa.array([0, 0, 0], type=pa.int64()),
            "blurb": pa.array(blurbs),
            "look_vec": _vec_column([_det_vec(f"img:{p}") for p in para_ids]),
            "blurb_vec": _vec_column([_det_vec(b) for b in blurbs]),
        }
    )
    stbl = conn.create_table("shots", data=shots)
    _fts_index(stbl, "blurb")
    (ddir / "articles.json").write_text(json.dumps(ARTICLES_DESCRIPTOR))


def _build_docs(root: Path, ddir: Path) -> None:
    conn = lancedb.connect(str(root / "docs.lance"))
    names = list(PARAS)
    paras = pa.table(
        {
            "para_id": pa.array([f"d-{n}" for n in names]),
            "seg_id": pa.array([0, 0, 0], type=pa.int64()),
            "t0": pa.array([0.0, 5.0, 10.0], type=pa.float64()),
            "t1": pa.array([5.0, 10.0, 15.0], type=pa.float64()),
            "body": pa.array([PARAS[n] for n in names]),
            "lang": pa.array(["sv", "sv", "en"]),
            "align_json": pa.array(["[]"] * 3),
        }
    )
    tbl = conn.create_table("paras", data=paras)
    _fts_index(tbl, "body")
    (ddir / "docs.json").write_text(json.dumps(DOCS_DESCRIPTOR))


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("search_api_corpus")
    ddir = root / "descriptors"
    ddir.mkdir()
    _build_articles(root, ddir)
    _build_docs(root, ddir)
    return root


def _state(root: Path, default: str) -> AppState:
    settings = Settings(
        MEDIA_DB=str(root / f"{default}.lance"),
        MEDIA_DB_ROOT=str(root),
        MEDIA_DESCRIPTOR_DIR=str(root / "descriptors"),
    )
    return AppState(
        db_path=root / f"{default}.lance", names=[], chunks=object(), settings=settings
    )


@pytest.fixture
def articles_state(corpus_root: Path) -> AppState:
    return _state(corpus_root, "articles")


@pytest.fixture
def articles(articles_state: AppState):
    return dataset_handle(articles_state)


@pytest.fixture
def docs(corpus_root: Path):
    return dataset_handle(_state(corpus_root, "docs"))


def _run(handle, spec, *, filters=None, image=None, embedder=_embedder, reranker=_no_reranker):
    return run_search(
        handle,
        get_embedder=embedder,
        get_reranker=reranker,
        spec=spec,
        filters=filters,
        image_bytes=image,
    )


# ── run_search: error branches ────────────────────────────────────────────────


class TestRunSearchErrors:
    def test_semantic_without_binding_returns_empty(self, docs) -> None:
        # Every embedding space is optional + uniform: a dataset without a
        # semantic vector yields NO results for that mode (not an error) — same
        # graceful degradation as visual/scene below.
        assert _run(docs, SearchSpec(q="carbon", mode=SearchMode.SEMANTIC)) == []

    def test_hybrid_without_text_query_is_400(self, articles) -> None:
        # Image-only hybrid: no text vector, so the FTS half can't run.
        spec = SearchSpec(q="", mode=SearchMode.HYBRID)
        with pytest.raises(HTTPException) as exc:
            _run(articles, spec, image=b"jpeg")
        assert exc.value.status_code == 400
        assert "text query" in exc.value.detail

    def test_hybrid_without_binding_is_400(self, docs) -> None:
        spec = SearchSpec(q="carbon", mode=SearchMode.HYBRID)
        with pytest.raises(HTTPException) as exc:
            _run(docs, spec)
        assert exc.value.status_code == 400


# ── run_search: graceful degradation ─────────────────────────────────────────


class TestRunSearchDegradation:
    def test_fts_returns_parsed_alignments_key(self, docs) -> None:
        spec = SearchSpec(q="carbon", mode=SearchMode.FTS)
        hits = _run(docs, spec, embedder=_no_embedder)
        assert hits
        assert hits[0]["alignments"] == []
        assert "align_json" not in hits[0]

    def test_visual_without_binding_returns_empty(self, docs) -> None:
        spec = SearchSpec(q="anything", mode=SearchMode.VISUAL)
        assert _run(docs, spec) == []

    def test_scene_without_binding_returns_empty(self, docs) -> None:
        spec = SearchSpec(q="carbon", mode=SearchMode.SCENE)
        assert _run(docs, spec) == []

    def test_scene_fts_without_binding_returns_empty(self, docs) -> None:
        spec = SearchSpec(q="anything", mode=SearchMode.SCENE_FTS)
        assert _run(docs, spec, embedder=_no_embedder) == []

    def test_all_without_vectors_falls_back_to_fts(self, docs) -> None:
        spec = SearchSpec(q="carbon emissions", mode=SearchMode.ALL)
        assert _run(docs, spec)

    def test_foreign_encoder_mode_returns_empty(self, articles) -> None:
        # A mode naming a space the search box can't drive (audio voiceprint):
        # no query vector matches its encoder, so it degrades to [] — NOT a 400
        # feeding a text vector into a foreign column.
        assert _run(articles, SearchSpec(q="carbon", mode="voiceprint", n=3)) == []

    def test_all_ignores_the_foreign_encoder_leg(self, articles) -> None:
        # The voiceprint space is declared, but 'all' must skip it (not abort)
        # and still fuse the FTS + semantic + visual + scene legs into hits.
        hits = _run(articles, SearchSpec(q=PARAS["economy"], mode=SearchMode.ALL, n=3))
        assert hits


# ── run_search: descriptor-driven retrieval on the full dataset ──────────────


class TestArticlesSearch:
    def test_fts_hits_use_descriptor_projection(self, articles) -> None:
        hits = _run(articles, SearchSpec(q="carbon", mode=SearchMode.FTS), embedder=_no_embedder)
        assert hits and "carbon" in hits[0]["body"]
        # The whole point of the port: identity/time/body come from the
        # descriptor, and no old-corpus field leaks into the hit shape.
        assert {"para_id", "seg_id", "t0", "t1", "body", "alignments"} <= hits[0].keys()
        assert "doc_id" not in hits[0]

    def test_semantic_ranks_planted_nearest(self, articles) -> None:
        # An exact-text query embeds to the same vector as its row → distance 0.
        hits = _run(articles, SearchSpec(q=PARAS["climate"], mode=SearchMode.SEMANTIC, n=3))
        assert hits and hits[0]["body"] == PARAS["climate"]
        assert hits[0]["_distance"] == pytest.approx(0.0, abs=1e-5)

    def test_visual_joins_back_to_rows(self, articles) -> None:
        # Text-query → frame-vector search → join back to paras. Ranking is
        # arbitrary under the fake embedder; the join must resolve.
        hits = _run(articles, SearchSpec(q="anything", mode=SearchMode.VISUAL, n=3))
        assert hits and "body" in hits[0]
        assert "_distance" in hits[0]

    def test_scene_returns_row_hits_with_caption_and_distance(self, articles) -> None:
        hits = _run(articles, SearchSpec(q="anything", mode=SearchMode.SCENE, n=3))
        assert hits
        assert {"para_id", "body", "caption", "_distance"} <= hits[0].keys()

    def test_scene_fts_hit_carries_the_bm25_score(self, articles) -> None:
        # Keyword over the caption column, needing NO embedder; the caption
        # `_score` must survive the join back to the row table.
        hits = _run(
            articles, SearchSpec(q="blurb", mode=SearchMode.SCENE_FTS, n=3), embedder=_no_embedder
        )
        assert hits and "_score" in hits[0]
        assert hits[0]["caption"].startswith("blurb showing")

    def test_caption_rides_along_on_fts_hits(self, articles) -> None:
        # Captions surface on EVERY mode (for the list/table views), not just scene.
        hits = _run(articles, SearchSpec(q="carbon", mode=SearchMode.FTS), embedder=_no_embedder)
        assert hits and hits[0]["caption"] == "blurb showing climate footage"

    def test_hybrid_runs(self, articles) -> None:
        hits = _run(articles, SearchSpec(q=PARAS["sports"], mode=SearchMode.HYBRID, n=3))
        assert hits  # FTS + vector fused via RRF (no rerank server needed)

    def test_all_runs(self, articles) -> None:
        hits = _run(articles, SearchSpec(q=PARAS["economy"], mode=SearchMode.ALL, n=3))
        assert hits

    def test_filterable_field_compiles_to_equality(self, articles) -> None:
        spec = SearchSpec(q="economy", mode=SearchMode.FTS)
        assert _run(articles, spec, filters={"lang": "en"}, embedder=_no_embedder)
        assert _run(articles, spec, filters={"lang": "sv"}, embedder=_no_embedder) == []
        # Equality, not LIKE-contains: a prefix must NOT match.
        assert _run(articles, spec, filters={"kind": "econ"}, embedder=_no_embedder) == []

    def test_rerank_reorders_the_head(self, articles) -> None:
        # "the" hits all three paras; the fake reranker scores by body length.
        spec = SearchSpec(q="the", mode=SearchMode.FTS, rerank=True, n=3)
        hits = _run(articles, spec, embedder=_no_embedder, reranker=_reranker)
        assert [h["body"] for h in hits] == sorted(PARAS.values(), key=len, reverse=True)


class TestFilterOnlyBrowse:
    def test_topic_filter_browses_matching_rows(self, articles) -> None:
        # A topic name matches ANY topic_l* layer column (old expansion kept).
        spec = SearchSpec(q="")
        hits = _run(articles, spec, filters={"topic": "Sports"}, embedder=_no_embedder)
        assert len(hits) == 1
        assert hits[0]["body"] == PARAS["sports"]
        assert hits[0]["alignments"] == []

    def test_topic_matches_the_broad_layer_too(self, articles) -> None:
        hits = _run(articles, SearchSpec(q=""), filters={"topic": "Money"}, embedder=_no_embedder)
        assert len(hits) == 1 and hits[0]["body"] == PARAS["economy"]

    def test_no_filters_and_no_query_is_empty(self, articles) -> None:
        assert _run(articles, SearchSpec(q=""), embedder=_no_embedder) == []

    def test_raw_where_passthrough_scopes_the_browse(self, articles) -> None:
        hits = _run(articles, SearchSpec(q="", where="t0 > 3.0"), embedder=_no_embedder)
        assert {h["body"] for h in hits} == {PARAS["sports"], PARAS["economy"]}

    def test_raw_where_ands_with_topic(self, articles) -> None:
        spec = SearchSpec(q="", where="seg_id = 9")
        assert _run(articles, spec, filters={"topic": "Sports"}, embedder=_no_embedder) == []


# ── Retrieval helpers ─────────────────────────────────────────────────────────


class TestSearchHelpers:
    def test_vector_search_missing_column_returns_empty(self, docs) -> None:
        tbl = resolve_target(docs).row_tbl
        vec = FakeEmbedClient().embed_text(["x"])[0]
        assert vector_search(tbl, vec, "body_vec", payload_columns=["para_id"], n=5, where=None) == []

    def test_vector_search_none_vec_returns_empty(self, docs) -> None:
        tbl = resolve_target(docs).row_tbl
        assert vector_search(tbl, None, "body_vec", payload_columns=["para_id"], n=5, where=None) == []

    def test_frame_search_none_table_returns_empty(self, articles) -> None:
        target = resolve_target(articles)
        vec = FakeEmbedClient().embed_text(["x"])[0]
        assert frame_search(None, target, vec, column="look_vec", n=5, where=None) == []

    def test_parse_alignments_json_decodes_lists(self) -> None:
        assert parse_alignments_json('[{"words": []}]') == [{"words": []}]

    @pytest.mark.parametrize("bad", [None, "", "not json", '"a string"', "{}"])
    def test_parse_alignments_json_degrades_to_empty(self, bad) -> None:
        assert parse_alignments_json(bad) == []


class TestRrfFuse:
    """Reciprocal-rank fusion, keyed on the descriptor's identity fields."""

    KEYS: ClassVar[list[str]] = ["para_id", "seg_id"]

    def test_row_in_both_rankings_outranks_row_in_one(self) -> None:
        a = {"para_id": "A", "seg_id": 0}
        b = {"para_id": "B", "seg_id": 0}
        c = {"para_id": "C", "seg_id": 0}
        # A is top of both lists → its reciprocal ranks sum → it must win.
        fused = rrf_fuse([[a, b], [a, c]], key_fields=self.KEYS)
        assert fused[0]["para_id"] == "A"

    def test_dedups_by_key_keeping_first_occurrence(self) -> None:
        first = {"para_id": "A", "seg_id": 0, "tag": "first"}
        second = {"para_id": "A", "seg_id": 0, "tag": "second"}
        fused = rrf_fuse([[first], [second]], key_fields=self.KEYS)
        assert len(fused) == 1
        assert fused[0]["tag"] == "first"  # the highest-ranked occurrence is canonical

    def test_distinct_seg_ids_are_distinct_rows(self) -> None:
        a0 = {"para_id": "A", "seg_id": 0}
        a1 = {"para_id": "A", "seg_id": 1}
        assert len(rrf_fuse([[a0, a1]], key_fields=self.KEYS)) == 2

    def test_empty_rankings_is_empty(self) -> None:
        assert rrf_fuse([], key_fields=self.KEYS) == []


# ── HTTP wire shape (router) ──────────────────────────────────────────────────


@pytest.fixture
def client(articles_state: AppState) -> TestClient:
    from search.api import dependencies as search_deps

    app = FastAPI()
    register_handlers(app)
    app.include_router(search_router)
    app.state.resources = articles_state
    app.dependency_overrides[search_deps.get_embedder] = lambda: _embedder
    app.dependency_overrides[search_deps.get_reranker] = lambda: _reranker
    return TestClient(app)


def _hits(client: TestClient, **params) -> list[dict]:
    r = client.get("/api/search", params=params)
    assert r.status_code == 200, (params, r.status_code, r.text)
    body = r.json()
    assert isinstance(body, list)
    return body


class TestSearchRoutes:
    def test_fts_runs_with_descriptor_shape(self, client) -> None:
        hits = _hits(client, q="carbon", mode="fts", n=5)
        assert hits and "carbon" in hits[0]["body"]
        assert {"para_id", "seg_id", "alignments"} <= hits[0].keys()

    def test_semantic_ranks_planted_nearest(self, client) -> None:
        hits = _hits(client, q=PARAS["climate"], mode="semantic", n=3)
        assert hits and hits[0]["body"] == PARAS["climate"]

    def test_hybrid_runs(self, client) -> None:
        assert _hits(client, q=PARAS["sports"], mode="hybrid", n=3)

    def test_all_runs(self, client) -> None:
        assert _hits(client, q=PARAS["economy"], mode="all", n=3)

    def test_all_mode_with_rerank_runs(self, client) -> None:
        assert _hits(client, q=PARAS["economy"], mode="all", rerank="true", n=3)

    def test_visual_runs(self, client) -> None:
        hits = _hits(client, q="anything", mode="visual", n=3)
        assert hits and "body" in hits[0]

    def test_empty_query_returns_empty(self, client) -> None:
        assert _hits(client, q="") == []

    def test_unknown_mode_degrades_to_empty(self, client) -> None:
        # `mode` is now `str` (any dataset embedding key is valid), so an
        # unavailable mode is no longer a 422 boundary — it degrades to no
        # results at the service (200 + []).
        r = client.get("/api/search", params={"q": "x", "mode": "bogus"})
        assert r.status_code == 200
        assert r.json() == []

    def test_filter_params_read_from_descriptor_filterable(self, client) -> None:
        # `lang` is not a route parameter — the router learns it from the
        # descriptor and reads it off the raw query string.
        assert _hits(client, q="economy", mode="fts", lang="en")
        assert _hits(client, q="economy", mode="fts", lang="sv") == []

    def test_topic_only_request_browses(self, client) -> None:
        hits = _hits(client, topic="Sports")
        assert len(hits) == 1 and hits[0]["body"] == PARAS["sports"]

    def test_dataset_param_selects_another_dataset(self, client) -> None:
        hits = _hits(client, q="carbon", mode="fts", dataset="docs")
        assert hits and hits[0]["para_id"].startswith("d-")

    def test_explicit_default_dataset_matches_implicit(self, client) -> None:
        implicit = _hits(client, q="carbon", mode="fts")
        explicit = _hits(client, q="carbon", mode="fts", dataset="articles")
        assert [h["para_id"] for h in implicit] == [h["para_id"] for h in explicit]

    def test_unknown_dataset_is_404_problem(self, client) -> None:
        r = client.get("/api/search", params={"q": "x", "dataset": "nope"})
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_post_hybrid_runs(self, client) -> None:
        r = client.post("/api/search", data={"q": PARAS["sports"], "mode": "hybrid", "n": 3})
        assert r.status_code == 200, r.text
        assert isinstance(r.json(), list) and r.json()

    def test_post_empty_query_returns_empty(self, client) -> None:
        r = client.post("/api/search", data={"q": "", "mode": "hybrid"})
        assert r.status_code == 200
        assert r.json() == []

    def test_post_filter_field_read_from_form(self, client) -> None:
        r = client.post("/api/search", data={"q": "economy", "mode": "fts", "lang": "sv"})
        assert r.status_code == 200
        assert r.json() == []

    def test_post_image_upload_drives_visual_search(self, client) -> None:
        r = client.post(
            "/api/search",
            data={"mode": "visual", "n": 3},
            files={"image": ("q.jpg", b"fake-jpeg-bytes", "image/jpeg")},
        )
        assert r.status_code == 200, r.text
        assert r.json()  # frame ranking joined back to row hits


# ── Version-keyed result cache ────────────────────────────────────────────────


class TestResultCache:
    def _spec(self, **kw):
        return SearchSpec(q=kw.pop("q", "carbon"), mode=kw.pop("mode", SearchMode.FTS), **kw)

    def test_hit_skips_recompute_and_returns_same(self, articles) -> None:
        from search.services.result_cache import run_cached

        cache: dict = {}
        calls = {"n": 0}

        def produce():
            calls["n"] += 1
            return [{"para_id": "p-climate", "run": calls["n"]}]

        r1 = run_cached(cache, 8, articles, self._spec(), None, None, produce)
        r2 = run_cached(cache, 8, articles, self._spec(), None, None, produce)
        assert calls["n"] == 1  # second call served from cache, produce not re-run
        assert r1 == r2
        assert len(cache) == 1

    def test_disabled_size_zero_always_recomputes(self, articles) -> None:
        from search.services.result_cache import run_cached

        cache: dict = {}
        calls = {"n": 0}

        def produce():
            calls["n"] += 1
            return []

        run_cached(cache, 0, articles, self._spec(), None, None, produce)
        run_cached(cache, 0, articles, self._spec(), None, None, produce)
        assert calls["n"] == 2  # cache off: no memoization, no version reads
        assert cache == {}

    def test_distinct_queries_get_distinct_entries(self, articles) -> None:
        from search.services.result_cache import run_cached

        cache: dict = {}
        calls = {"n": 0}

        def produce():
            calls["n"] += 1
            return []

        run_cached(cache, 8, articles, self._spec(q="a"), None, None, produce)
        run_cached(cache, 8, articles, self._spec(q="b"), None, None, produce)
        assert calls["n"] == 2 and len(cache) == 2

    def test_lru_eviction_drops_oldest(self, articles) -> None:
        from search.services.result_cache import run_cached

        cache: dict = {}
        for i in range(5):
            run_cached(cache, 3, articles, self._spec(q=f"q{i}"), None, None, lambda: [])
        assert len(cache) == 3  # capped at max_size, oldest evicted

    def test_query_hash_covers_filters_and_image(self) -> None:
        from search.services.result_cache import query_hash

        spec = self._spec(q="a")
        assert query_hash(spec, {"lang": "en"}, None) != query_hash(spec, {"lang": "sv"}, None)
        assert query_hash(spec, None, b"img1") != query_hash(spec, None, b"img2")
        assert query_hash(spec, None, None) != query_hash(spec, None, b"img1")
        # filter ORDER must not matter (canonicalized)
        assert query_hash(spec, {"a": "1", "b": "2"}, None) == query_hash(spec, {"b": "2", "a": "1"}, None)

    def test_version_signature_covers_every_search_table(self, articles) -> None:
        from search.services.result_cache import version_signature

        sig = version_signature(articles)
        # articles reads paras (row + fts) and shots (visual/scene) — both versioned.
        assert "paras:" in sig and "shots:" in sig

    def test_a_version_change_invalidates(self, articles, monkeypatch) -> None:
        # A table write bumps the signature → the key changes → recompute. Driven
        # here by faking the signature so the shared corpus stays untouched.
        import search.services.result_cache as rc

        cache: dict = {}
        calls = {"n": 0}
        versions = iter(["v1", "v1", "v2"])
        monkeypatch.setattr(rc, "version_signature", lambda _handle: next(versions))

        def produce():
            calls["n"] += 1
            return [calls["n"]]

        r1 = rc.run_cached(cache, 8, articles, self._spec(), None, None, produce)  # v1 miss
        r2 = rc.run_cached(cache, 8, articles, self._spec(), None, None, produce)  # v1 hit
        r3 = rc.run_cached(cache, 8, articles, self._spec(), None, None, produce)  # v2 miss
        assert calls["n"] == 2
        assert r1 == r2 and r1 != r3

    def test_route_populates_cache_and_serves_repeat(self, client, articles_state) -> None:
        articles_state.search_cache.clear()
        h1 = _hits(client, q="carbon", mode="fts", n=5)
        assert len(articles_state.search_cache) == 1
        h2 = _hits(client, q="carbon", mode="fts", n=5)
        assert h1 == h2
        assert len(articles_state.search_cache) == 1  # same key → still one entry


# ── Lazy client accessors (503 mapping + descriptor-fed dim) ──────────────────


class TestEnsureClients:
    def _boom(self, *_args, **_kwargs) -> Never:
        raise RuntimeError("server down")

    def test_cached_embedder_short_circuits(self, articles_state, monkeypatch) -> None:
        import search.services.encoders.embedding as embedding_mod

        sentinel = object()
        articles_state.embedder = sentinel
        # If it tried to construct, boom would fire — proving the cache short-circuit.
        monkeypatch.setattr(embedding_mod, "VLLMEmbeddingClient", self._boom)
        assert clients.ensure_embedder(articles_state) is sentinel

    def test_embedder_gets_url_and_descriptor_dim(self, articles_state, monkeypatch) -> None:
        import search.services.encoders.embedding as embedding_mod

        class Recorder:
            def __init__(self, embed_url: str, *, expected_dim: int) -> None:
                self.embed_url = embed_url
                self.expected_dim = expected_dim

        monkeypatch.setattr(embedding_mod, "VLLMEmbeddingClient", Recorder)
        made = cast(Recorder, clients.ensure_embedder(articles_state))
        assert made.embed_url == articles_state.settings.embed_url
        assert made.expected_dim == DIM  # the semantic binding's dim, not a constant
        assert articles_state.embedder is made  # written back for reuse

    def test_no_arg_fake_skips_dim_resolution(self, monkeypatch) -> None:
        import search.services.encoders.embedding as embedding_mod

        # A registry-less state: resolving the dim would fail, but a fake whose
        # signature takes no kwargs must never trigger it.
        state = AppState(db_path=Path("missing.lance"), names=[], chunks=object())
        made = object()
        monkeypatch.setattr(embedding_mod, "VLLMEmbeddingClient", lambda: made)
        assert clients.ensure_embedder(state) is made

    def test_embedder_failure_maps_to_503(self, articles_state, monkeypatch) -> None:
        import search.services.encoders.embedding as embedding_mod

        monkeypatch.setattr(embedding_mod, "VLLMEmbeddingClient", self._boom)
        with pytest.raises(ServiceUnavailableError) as exc:
            clients.ensure_embedder(articles_state)
        assert exc.value.status_code == 503

    def test_reranker_gets_url_and_caches(self, articles_state, monkeypatch) -> None:
        import search.services.encoders.reranker as reranker_mod

        class Recorder:
            def __init__(self, rerank_url: str) -> None:
                self.rerank_url = rerank_url

        monkeypatch.setattr(reranker_mod, "VLLMReranker", Recorder)
        made = cast(Recorder, clients.ensure_reranker(articles_state))
        assert made.rerank_url == articles_state.settings.rerank_url
        assert articles_state.reranker is made

    def test_reranker_failure_maps_to_503(self, articles_state, monkeypatch) -> None:
        import search.services.encoders.reranker as reranker_mod

        monkeypatch.setattr(reranker_mod, "VLLMReranker", self._boom)
        with pytest.raises(ServiceUnavailableError) as exc:
            clients.ensure_reranker(articles_state)
        assert exc.value.status_code == 503


# ── Vendored encoder units (no server) ────────────────────────────────────────


class TestVendoredEncoders:
    def test_l2_normalize_validates_expected_dim(self) -> None:
        from search.services.encoders.image import l2_normalize

        good = l2_normalize(np.ones((2, DIM), dtype=np.float32), expected_dim=DIM)
        assert good.shape == (2, DIM)
        assert np.allclose(np.linalg.norm(good, axis=1), 1.0)
        with pytest.raises(ValueError):
            l2_normalize(np.ones((2, DIM + 1), dtype=np.float32), expected_dim=DIM)

    def test_image_to_data_url_square_crops_to_server_pin(self) -> None:
        import base64
        import io

        from PIL import Image

        from search.services.encoders.image import _IMAGE_SIDE, image_to_data_url

        buf = io.BytesIO()
        Image.new("RGB", (640, 360), (200, 30, 30)).save(buf, format="JPEG")
        url = image_to_data_url(buf.getvalue())
        assert url.startswith("data:image/jpeg;base64,")
        decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert decoded.size == (_IMAGE_SIDE, _IMAGE_SIDE)

    def test_embedding_client_rejects_nonpositive_dim(self) -> None:
        from search.services.encoders.embedding import VLLMEmbeddingClient

        with pytest.raises(ValueError):
            VLLMEmbeddingClient("http://127.0.0.1:1", expected_dim=0)

    def test_embedding_client_empty_input_shapes_by_expected_dim(self) -> None:
        from search.services.encoders.embedding import VLLMEmbeddingClient

        client = VLLMEmbeddingClient("http://127.0.0.1:1", expected_dim=DIM)
        assert client.embed_text([]).shape == (0, DIM)
        assert client.embed_image([]).shape == (0, DIM)
        client.close()
