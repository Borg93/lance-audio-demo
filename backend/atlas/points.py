"""Atlas ``/points`` payload builder — the Arrow IPC serialization behind the map.

Pure data path (no HTTP): scan the projected ``chunks`` columns for a space and
encode them into one Apache Arrow IPC stream (float16 coords + int keys + a few
``DICTIONARY<int32, utf8>`` colour columns). :mod:`backend.atlas.router` wires
this to the ``/api/atlas`` routes; :mod:`backend.warmup` precomputes it on boot.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from backend.core.exceptions import ValidationError
from backend.state import AppState
from raudio.features.topic_tree import topic_layer_columns

#: The three projection spaces and their column triplets. ``text`` is the default
#: (``raudio feature atlas``, from ``text_embedding``); ``visual`` is the
#: frame-embedding map (``raudio feature atlas --space visual``). Each space's
#: X column doubling as the "is this space built?" signal — mirrors how
#: ``search/service.py`` gates semantic search on ``text_embedding`` presence.
_SPACES: dict[str, dict[str, str]] = {
    "text": {"x": "atlas_x", "y": "atlas_y", "cluster": "atlas_cluster"},
    "visual": {"x": "atlas_img_x", "y": "atlas_img_y", "cluster": "atlas_img_cluster"},
    "caption": {"x": "atlas_cap_x", "y": "atlas_cap_y", "cluster": "atlas_cap_cluster"},
}

#: Memoized /points payloads keyed on (space, dataset version). The full 145k-row
#: scan + dictionary-encode is identical for a given dataset version, so we build
#: the Arrow IPC stream once per space and serve the cached *bytes* thereafter. The
#: dataset version bumps on any rewrite (and a backend restart reopens `chunks_ds`
#: anyway), invalidating stale entries. A plain dict suffices: writes are idempotent
#: (same input → same bytes), so a concurrent double-compute overwrites with equal
#: bytes.
_POINTS_CACHE: dict[tuple[str, int], bytes] = {}


def _space_cols(space: str) -> dict[str, str]:
    cols = _SPACES.get(space)
    if cols is None:
        raise ValidationError("unknown space (text|visual|caption)")
    return cols


def _is_projected(state: AppState, space: str = "text") -> bool:
    return _space_cols(space)["x"] in state.chunks.schema.names


def _dictionary(column: pa.ChunkedArray) -> pa.Array:
    """Encode a string column as Arrow ``DICTIONARY<int32, utf8>`` — the codes/
    labels split done natively (indices = per-point colour codes, values =
    labels), replacing the hand-rolled ``_factorize``.

    NULLs are filled to the empty-string label ``""`` first (the frontend
    renders that muted), so the dictionary carries no nulls — matching the old
    contract. ``int32`` indices keep the codes in the typed-array range the JS
    scatter expects.
    """
    filled = pc.fill_null(column.cast(pa.string()), "")
    encoded = filled.combine_chunks().dictionary_encode()
    return encoded.cast(pa.dictionary(pa.int32(), pa.string()))


def _doc_file_stems(doc_dict: pa.Array, doc_ids: list[str], audio_paths: list[str]) -> list[str]:
    """Filename stem per distinct doc, aligned with the ``doc`` dictionary order.

    The map view colours/labels by video using the readable audio stem (e.g.
    ``T0000234_00001``) rather than the hashed doc id. The result is one value
    per dictionary entry, in dictionary order, so the frontend can index it by
    the same code it uses for ``doc``. Shipped in schema metadata (not a column:
    it has one entry per distinct doc, not one per point).
    """
    stem_by_doc: dict[str, str] = {}
    for d, a in zip(doc_ids, audio_paths, strict=True):
        if d not in stem_by_doc:
            stem_by_doc[d] = Path(a).stem if a else d
    return [stem_by_doc.get(d, d) for d in doc_dict.to_pylist()]


def _build_points(state: AppState, space: str) -> bytes:
    """The expensive part of /points: full-table scan → one Arrow IPC stream.

    Builds a single ``pyarrow.Table`` (float16 coords, ~3 sig-digit precision —
    fine for a ~2000px scatter, and it halves both the wire payload and the GPU
    vertex buffer; int32/int64 keys, and a handful of ``DICTIONARY<int32, utf8>``
    colour columns) and serializes it to Arrow IPC **stream** bytes. Pulled out
    so :func:`backend.atlas.router.atlas_points` can memoize the bytes per (space, version).
    """
    cols = _space_cols(space)
    x_col, y_col, cluster_col = cols["x"], cols["y"], cols["cluster"]
    schema = set(state.chunks.schema.names)
    columns = ["doc_id", "audio_path", "speech_id", "chunk_id", x_col, y_col]
    # The broadest topic layer (data-dependent — `topic_l{N-1}`, not always l2)
    # colours the map into named regions; `doc_topic` is the per-video roll-up.
    # Both are low-cardinality (~19), so the dictionary label list stays tiny.
    topic_cols = topic_layer_columns(list(schema))
    broad_topic_col = topic_cols[-1] if topic_cols else None
    for optional in (cluster_col, "language", "namn", broad_topic_col, "doc_topic"):
        if optional and optional in schema:
            columns.append(optional)

    # `with_row_id` ships each point's stable Lance row address (`_rowid`) so the
    # selection table can be fetched with an O(selection) `take` (see /chunks)
    # instead of a per-key filtered full-table scan.
    tbl = state.chunks_ds.scanner(
        columns=columns, filter=f"{x_col} IS NOT NULL", with_row_id=True
    ).to_table()

    def halves(name: str) -> pa.Array:
        # Ship coords as float16 — ~3 sig-digit precision, sub-pixel on the
        # ~2000px scatter — halving the /points payload AND the GPU vertex
        # buffer. The frontend reads the raw f16 bits for the GPU and decodes
        # them to f32 for CPU hover/lasso math (api.ts:f16ToF32).
        arr = tbl.column(name).to_numpy(zero_copy_only=False).astype(np.float16)
        return pa.array(arr, type=pa.float16())

    def ints(name: str, dtype: pa.DataType) -> pa.Array:
        return tbl.column(name).combine_chunks().cast(dtype)

    doc_dict = _dictionary(tbl.column("doc_id"))
    doc_files = _doc_file_stems(
        doc_dict.dictionary,
        tbl.column("doc_id").to_pylist(),
        tbl.column("audio_path").to_pylist(),
    )

    arrays: list[pa.Array] = [
        halves(x_col),
        halves(y_col),
        ints("speech_id", pa.int32()),
        ints("chunk_id", pa.int32()),
        ints("_rowid", pa.int64()),  # stable address for take-based selection fetch
        doc_dict,  # DICTIONARY<int32, utf8>: indices = `doc`, values = `docs`
    ]
    names = ["x", "y", "speech_id", "chunk_id", "rowid", "doc"]

    if cluster_col in columns:
        arrays.append(ints(cluster_col, pa.int32()))
        names.append("cluster")
    if "language" in columns:
        arrays.append(_dictionary(tbl.column("language")))
        names.append("language")
    if "namn" in columns:
        # Archival name dictionary: low-cardinality metadata for the hover popup.
        # A small label list + a per-point int32 — safe to ship for 145k rows
        # (unlike high-cardinality text/caption, which we lazy-fetch per chunk).
        arrays.append(_dictionary(tbl.column("namn")))
        names.append("namn")
    if broad_topic_col and broad_topic_col in columns:
        # Broadest chunk topic — colours the map into named regions. Unclustered
        # chunks are NULL → "" label, rendered muted on the map.
        arrays.append(_dictionary(tbl.column(broad_topic_col)))
        names.append("topic")
    if "doc_topic" in columns:
        # Per-video dominant topic; NULL → "" label when a video's chunks are all
        # noise (so not strictly 100% — most videos do get a topic).
        arrays.append(_dictionary(tbl.column("doc_topic")))
        names.append("doc_topic")

    # count + space + docFiles ride along in the schema metadata. `docFiles` has
    # one entry per distinct doc (not one per point), so it can't be a column in
    # this per-point table — it's JSON-encoded here, aligned with the `doc`
    # dictionary order so the frontend can index it by the same code.
    out_schema = pa.schema(
        [pa.field(n, a.type) for n, a in zip(names, arrays, strict=True)],
        metadata={
            b"count": str(tbl.num_rows).encode(),
            b"space": space.encode(),
            b"docFiles": json.dumps(doc_files).encode(),
        },
    )
    out = pa.table(arrays, schema=out_schema)

    sink = pa.BufferOutputStream()
    with pa.ipc.RecordBatchStreamWriter(sink, out.schema) as writer:
        writer.write_table(out)
    return sink.getvalue().to_pybytes()
