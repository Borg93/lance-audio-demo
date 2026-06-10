"""Shared projection columns + IVF_PQ recall knobs for the search layer.

These are algorithmic constants (not env-varying), so they stay module
constants, not :class:`~backend.core.config.Settings`. Single source of truth
so the vector / frames / fts / postprocess modules don't duplicate them.
"""

from __future__ import annotations

_HIT_COLUMNS = [
    "_score",
    "doc_id",
    "audio_path",
    "speech_id",
    "chunk_id",
    "start",
    "end",
    "duration",
    "text",
    "language",
    "namn",
    "referenskod",
    "bildid",
    "extraid",
]
# `alignments_json` is intentionally NOT projected here: it's a multi-KB per-word
# timing blob (~93% of a search response's bytes) that only the *player* renders,
# for the ONE selected hit. Search hits ship `alignments: []`; the player
# lazy-fetches the real array via GET /api/chunk-alignments/{doc}/{speech}/{chunk}.

# Hit columns without the FTS-only BM25 `_score`. Vector and hybrid searches
# surface `_distance` / `_relevance_score` instead, so selecting `_score` there
# would fail.
_PAYLOAD_COLUMNS = [c for c in _HIT_COLUMNS if c != "_score"]

# Vector columns on chunk_frames that a query can rank against (must match the
# names the feature engine writes — see raudio.features.columns). frame = raw
# image similarity; caption = text-embedding of the Swedish caption (scene).
_FRAME_EMBED_COLUMN = "frame_embedding"
_CAPTION_EMBED_COLUMN = "caption_embedding"
# Plain-text caption column on chunk_frames (the Gemma Swedish caption). Surfaced
# on every hit for the list/table views — see _attach_captions.
_CAPTION_COLUMN = "caption"

# IVF_PQ recall knobs. Lance's default probes too few partitions for good recall;
# ~√(num_partitions) partitions plus a refine pass that re-scores the top
# candidates with full-precision vectors restores it at a small latency cost
# (see docs/INVESTIGATION.md §A3). Ignored when the column has no IVF index.
_VECTOR_NPROBES = 20
# Adaptive probing: scan _VECTOR_NPROBES partitions, extending toward all of them
# only when a *selective* prefilter leaves the first pass short of `limit`. Pinning
# nprobes (min == max) instead makes a selective scope silently drop rows — the
# index probes too few partitions to contain them (a 20-chunk scope returned 1/20
# before this). 0 = uncapped, cheap here (~35 IVF partitions at 145k rows → worst
# case a full-index scan of a few ms); cap it if the corpus grows ~100x.
_VECTOR_MAX_NPROBES = 0
_VECTOR_REFINE_FACTOR = 3
