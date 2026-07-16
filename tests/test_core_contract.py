"""P1.1 contract: the pipeline core imports no modality or model-client code.

``rmedia.core`` must stay media-agnostic — stages receive their compute
callables injected by the composition root, so importing the core can never
pull ffmpeg/pyannote wrappers (``rmedia.modalities``) or vLLM HTTP clients
(``rmedia.clients``) into the process. Runs in a subprocess so other tests'
imports can't contaminate ``sys.modules``.
"""

import subprocess
import sys

_PROBE = """
import sys
import rmedia.core
import rmedia.core.engine
bad = [m for m in sys.modules if m.startswith(("rmedia.modalities", "rmedia.clients"))]
assert not bad, f"core pulled in: {bad}"
print("CORE-CONTRACT OK")
"""


def test_core_imports_no_modalities_or_clients() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "CORE-CONTRACT OK" in result.stdout
