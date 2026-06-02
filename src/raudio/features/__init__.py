"""Feature engineering — derive Lance columns via data evolution.

:mod:`raudio.features.engine` is the type-agnostic core (attach any column —
vector, string, struct — via ``add_columns`` without rewriting base data).
:mod:`raudio.features.columns` defines the concrete feature columns
(text/frame embeddings, summaries, captions) and the ``FEATURES`` registry the
``raudio feature <name>`` CLI drives.
"""
