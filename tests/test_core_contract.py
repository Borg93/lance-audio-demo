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


_MODEL_FREE_PROBE = """
import sys
import ratch
import ratch.core.jobs
import ratch.core.runners
import ratch.features.columns
models = ("torch", "easytranscriber", "pyannote", "toponymy", "transformers", "lightrag", "ray")
loaded = [m for m in models if m in sys.modules]
assert not loaded, f"importing ratch pulled model/heavy deps: {loaded}"
print("MODEL-FREE OK")
"""


def test_import_ratch_loads_zero_model_deps() -> None:
    """The runners/ architecture headline: ratch is pure orchestration — importing
    it (including the jobs seam and the feature registry) loads no model stack,
    and not even ray (the driver imports it lazily at run time)."""
    result = subprocess.run(
        [sys.executable, "-c", _MODEL_FREE_PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "MODEL-FREE OK" in result.stdout
