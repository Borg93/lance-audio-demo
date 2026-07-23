"""The lance-ns observability contract — ``common.obs`` + service activation.

Pins the two facts the OTLP log path depends on (lance-ns obs audit 2026-07-13):
the app package loggers sit at INFO (default WARNING would silently drop the
whole audit/lifecycle tier before the SDK handler sees it), and every service
main activates that at import. The SDK itself is launcher-activated
(``opentelemetry-instrument``) — nothing here asserts exporters.
"""

from __future__ import annotations

import logging

from common.obs import _APP_LOGGERS, configure_app_logging


def test_raises_exactly_the_app_packages_to_info() -> None:
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)
    configure_app_logging()
    for name in _APP_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() == logging.INFO
    # Dependency chatter is NOT dragged along: root/third-party stay untouched.
    assert logging.getLogger("httpx").level == logging.NOTSET


def test_every_service_main_activates_it() -> None:
    # Importing a service main must leave its package logger at INFO — the call
    # sits at module import, before the app serves anything.
    import annotator.main  # noqa: F401
    import search.main  # noqa: F401
    import viewer.main  # noqa: F401

    for pkg in ("viewer", "search", "annotator", "common", "ratch"):
        assert logging.getLogger(pkg).getEffectiveLevel() == logging.INFO, pkg


def test_info_records_pass_the_level_gate(caplog) -> None:
    configure_app_logging()
    with caplog.at_level(logging.INFO, logger="viewer"):
        logging.getLogger("viewer.api").info("save_committed", extra={"rows": 3})
    assert any(r.message == "save_committed" for r in caplog.records)
