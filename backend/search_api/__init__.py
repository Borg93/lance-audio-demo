"""Descriptor-driven search service (LANCE_MEDIA_MERGE §4.4, search-role group).

The schema-agnostic port of ``backend.search``: every table/column name is read
from the dataset's :class:`~backend.lancekit.descriptor.DatasetDescriptor` at
runtime, query-encoder clients are vendored under ``encoders/`` (no ``rmedia``
imports), and datasets are addressed via an optional ``dataset`` query param
resolved through :func:`backend.state.dataset_handle`. Shared primitives come
only from ``backend.core`` / ``backend.lancekit`` / ``backend.state`` /
``backend.deps`` — never from the sibling media-role group.
"""
