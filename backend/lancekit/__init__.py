"""Schema-agnostic Lance primitives for the serving layer (LANCE_MEDIA_MERGE §4.4).

Shared by ``backend.media_api`` and ``backend.search_api`` — the only backend
package either may import besides ``backend.core``. Standalone by contract:
nothing here imports the pipeline package (``rmedia``), so the backend can be
lifted into rask without dragging the pipeline along (P2.8).
"""
