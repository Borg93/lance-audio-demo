"""Shared observability init — mirrors lance-ns ``services/common/obs.py`` exactly.

The SDK itself is activated by the ``opentelemetry-instrument`` launcher (per the
otel skill: installing the packages does nothing without it) — on the cluster the
pod entrypoint runs the launcher, which attaches an SDK ``LoggingHandler(NOTSET)``
to the ROOT logger at process start. But nothing raises the root/app level from
Python's default WARNING, so ``log.info()`` records — the whole INFO
audit/request-lifecycle tier (lance-ns obs audit 2026-07-13: severity 9 events
like save commits, job submissions, stage progress) — would be rejected by
``isEnabledFor()`` and never reach the OTLP exporter.

Fix, same as theirs: each service calls :func:`configure_app_logging` once at
import time (the SDK handler is already on root by then, since the launcher runs
before app import). It raises ONLY the app package loggers to INFO — the INFO
tier flows to the collector WITHOUT dragging every dependency's chatter along.
Without the launcher (local dev) this is just stdlib log-level configuration.

At merge this file DELETES in favour of lance-ns's — the import path
(``from common.obs import configure_app_logging``) and call shape are already
identical; only their ``_APP_LOGGERS`` tuple gains our package names.
"""

from __future__ import annotations

import logging

#: The top-level app packages whose module loggers (``logging.getLogger(__name__)``)
#: must emit INFO. Effective levels inherit from these package loggers, so raising
#: the package raises the whole tree. ``ratch`` covers pipeline runs that log
#: through the same process (e.g. `ratch serve`).
_APP_LOGGERS = ("common", "viewer", "search", "annotator", "ratch")


def configure_app_logging(level: int = logging.INFO) -> None:
    """Raise the app package loggers to ``level`` (default INFO) so their records reach the OTLP handler."""
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(level)
