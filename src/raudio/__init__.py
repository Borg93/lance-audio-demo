"""Compatibility shim — ``raudio`` moved to :mod:`rmedia` (lance-media).

Kept only so the pre-split backend keeps importing until Phase 2 of
``docs/LANCE_MEDIA_MERGE.md`` severs its pipeline dependency (P2.8). New code
imports :mod:`rmedia`; each submodule here re-exports its successor verbatim.
"""
