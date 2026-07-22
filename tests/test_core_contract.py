"""P1.1 contract: the pipeline core imports no modality or model-client code.

``ratch.core`` must stay media-agnostic — stages receive their compute
callables injected by the composition root, so importing the core can never
pull ffmpeg/pyannote wrappers (``ratch.modalities``) or vLLM HTTP clients
(``ratch.clients``) into the process. Runs in a subprocess so other tests'
imports can't contaminate ``sys.modules``.
"""

import subprocess
import sys

_PROBE = """
import sys
import ratch.core
import ratch.core.engine
bad = [m for m in sys.modules if m.startswith(("ratch.modalities", "ratch.clients"))]
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
