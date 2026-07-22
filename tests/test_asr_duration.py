"""Unit tests for ``_probe_duration_s`` — the ffmpeg-seek duration estimator.

``_probe_duration_s`` approximates a media file's duration without a full
decode: it grows an upper bound by exponential doubling (60s -> x2 ...) and then
binary-refines the EOF over 8 iterations. The estimate it returns must be a
slight *under*-estimate (so planned sample windows stay inside the file) and the
``ceiling_s`` guard must keep a pathologically long / unbounded file from
looping forever.

The only collaborator is ``read_audio_segment`` (a real ffmpeg fast-seek). We
patch it AT THE SEAM -- ``ratch.modalities.av.asr.detect_language.read_audio_segment``, the
name the module actually calls (it does ``from easytranscriber.audio import
read_audio_segment`` at module top) -- with a pure fake that simulates a file of
a known true duration: a read at ``start_sec`` returns audio iff
``start_sec < true_duration``. No ffmpeg, no real files, no GPU, no model.

The window-planning math (``_plan_sample_starts``) and the full
``detect_and_sort`` orchestration are out of scope here; this file isolates the
duration probe.

Run:  uv run pytest tests/test_asr_duration.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ratch.modalities.av.asr.detect_language as detect_language
from ratch.modalities.av.asr.detect_language import _probe_duration_s

# Stand-ins for what real ffmpeg fast-seek hands back: a tiny populated buffer
# (audio present, ``.size > 0``) vs an empty buffer (past EOF, ``.size == 0``).
# ``_probe_duration_s`` only ever inspects ``.size``, so a bare numpy array is a
# faithful stand-in for the decoded clip.
_AUDIO = np.zeros(8, dtype=np.float32)
_SILENCE = np.zeros(0, dtype=np.float32)

# ``Path`` is inert here -- every read goes through the faked seam -- so any
# dummy path stands in for a real media file.
_DUMMY = Path("dummy.mp4")


def _fake_file(true_duration: float):
    """A ``read_audio_segment`` replacement simulating a file of ``true_duration`` s.

    A read returns populated audio while its ``start_sec`` lands before the end
    of the file, and an empty buffer once it seeks at or past the true duration --
    exactly the EOF signal ``_probe_duration_s`` binary-searches for.
    """

    def read_audio_segment(*, audio_path, start_sec, duration_sec, sample_rate):
        return _AUDIO if start_sec < true_duration else _SILENCE

    return read_audio_segment


def _counting_unbounded_file():
    """An *infinitely* long file plus a call counter.

    Every read returns audio (the file never ends), so only ``ceiling_s`` can
    stop the doubling loop. The counter lets a test prove termination is real
    (bounded number of reads) rather than relying on the test simply not hanging.
    """
    calls = {"n": 0}

    def read_audio_segment(*, audio_path, start_sec, duration_sec, sample_rate):
        calls["n"] += 1
        return _AUDIO

    return read_audio_segment, calls


def _empty_counting_file():
    """A file with no readable audio plus a call counter.

    Every read is empty, so ``has_audio(0.0)`` fails immediately. The counter
    proves the probe short-circuits on the very first read.
    """
    calls = {"n": 0}

    def read_audio_segment(*, audio_path, start_sec, duration_sec, sample_rate):
        calls["n"] += 1
        return _SILENCE

    return read_audio_segment, calls


# -- (a) boundary: a file with no readable audio at all ----------------------


def test_probe_duration_no_audio_at_zero_returns_zero(monkeypatch):
    monkeypatch.setattr(detect_language, "read_audio_segment", _fake_file(0.0))
    assert _probe_duration_s(_DUMMY) == 0.0


def test_probe_duration_no_audio_short_circuits_before_doubling(monkeypatch):
    # A dead file must cost exactly one probe (the ``has_audio(0.0)`` guard) and
    # never enter the doubling/refine loops.
    fake, calls = _empty_counting_file()
    monkeypatch.setattr(detect_language, "read_audio_segment", fake)

    assert _probe_duration_s(_DUMMY) == 0.0
    assert calls["n"] == 1


# -- (b) convergence: real durations land just under the truth ---------------


@pytest.mark.parametrize(
    "true_duration",
    [
        0.5,      # shorter than one 60s doubling step -- refined inside [0, 60)
        1.0,
        30.0,     # still inside the first [0, 60) bracket
        59.9,
        100.0,    # one doubling: 60 -> 120, then refine
        600.0,    # several doublings into the minutes
        3700.0,   # ~1 hour, deep into the doubling phase
    ],
)
def test_probe_duration_converges_just_under_true_duration(monkeypatch, true_duration):
    monkeypatch.setattr(detect_language, "read_audio_segment", _fake_file(true_duration))

    est = _probe_duration_s(_DUMMY)

    # Under-estimate: the returned value is the last offset that still had audio,
    # so it must never claim more than the file actually holds (windows planned
    # off it stay inside the file).
    assert est < true_duration
    # ...but a *useful* under-estimate: 8 binary-refine steps halve the final
    # [lo, hi] bracket repeatedly. The final bracket is at most one doubling
    # interval, whose width is bounded by ``max(true_duration, 60)`` -- so the
    # leftover gap is at most that width / 2**8.
    bracket = max(true_duration, 60.0)
    assert true_duration - est <= bracket / 2**8 + 1e-9


def test_probe_duration_error_stays_a_small_relative_slice_for_long_files(monkeypatch):
    # The absolute error grows with duration (wider final bracket) but stays a
    # small *relative* slice -- well under 1% for a multi-minute file.
    monkeypatch.setattr(detect_language, "read_audio_segment", _fake_file(600.0))
    est = _probe_duration_s(_DUMMY)
    assert (600.0 - est) / 600.0 < 0.01


# -- (c) exponential doubling reaches large durations ------------------------


def test_probe_duration_doubling_phase_reaches_multi_hour_file(monkeypatch):
    # 5h = 18000s is only reachable if the 60 -> x2 phase keeps growing the upper
    # bound past it; a search that stopped at the first 60s window would cap the
    # answer absurdly low.
    true_duration = 18000.0
    monkeypatch.setattr(detect_language, "read_audio_segment", _fake_file(true_duration))

    est = _probe_duration_s(_DUMMY)

    assert est > 17000.0           # genuinely climbed into multi-hour territory
    assert est < true_duration     # and is still a (now coarser) under-estimate


def test_probe_duration_doubling_is_logarithmic_not_linear(monkeypatch):
    # The whole point of doubling is cheapness: an ~8h file is found in far fewer
    # reads than its second-count. Count the reads to pin that it never linearly
    # scans the file.
    true_duration = 28800.0  # 8 hours
    calls = {"n": 0}

    def counting(*, audio_path, start_sec, duration_sec, sample_rate):
        calls["n"] += 1
        return _AUDIO if start_sec < true_duration else _SILENCE

    monkeypatch.setattr(detect_language, "read_audio_segment", counting)

    est = _probe_duration_s(_DUMMY)

    assert est < true_duration
    # 1 guard + ~log2(28800/60) doubling reads + 8 refine reads -- tens, not
    # hundreds of thousands.
    assert calls["n"] < 30


# -- (d) ceiling clamps a pathologically long / unbounded file ---------------


def test_probe_duration_unbounded_file_terminates_and_is_bounded(monkeypatch):
    # A file that never returns silence would loop the doubling phase forever
    # without ``ceiling_s``. The guard must stop the doubling and yield a finite
    # estimate after a bounded number of reads.
    fake, calls = _counting_unbounded_file()
    monkeypatch.setattr(detect_language, "read_audio_segment", fake)

    est = _probe_duration_s(_DUMMY, ceiling_s=36000.0)

    assert np.isfinite(est)
    assert est > 0.0
    # NOTE: the returned value can land ABOVE ``ceiling_s`` -- the doubling guard
    # (``while hi < ceiling_s``) only stops *further* doubling once ``hi`` has
    # already crossed the ceiling, so the last doubling step may overshoot. The
    # real guarantee is that it can't run away past that last step (< 2x ceiling)
    # and that the read count stays bounded (no infinite loop).
    assert est < 2 * 36000.0
    assert calls["n"] < 40


def test_probe_duration_ceiling_on_doubling_boundary_clamps_below_ceiling(monkeypatch):
    # When the ceiling coincides exactly with a doubling boundary (60 -> 120),
    # the doubling halts right at it and the refine stays inside [60, 120), so
    # the estimate lands just below the ceiling rather than overshooting. This is
    # the one ceiling value for which "below the ceiling" genuinely holds.
    monkeypatch.setattr(detect_language, "read_audio_segment", _counting_unbounded_file()[0])

    est = _probe_duration_s(_DUMMY, ceiling_s=120.0)

    assert est < 120.0
    assert est > 119.0  # refined right up to the clamp


def test_probe_duration_larger_ceiling_never_lowers_the_estimate(monkeypatch):
    # ``ceiling_s`` is a guard, not a target: raising it can only let the search
    # climb at least as high for the same unbounded file (monotonic
    # non-decreasing).
    monkeypatch.setattr(detect_language, "read_audio_segment", _counting_unbounded_file()[0])
    low = _probe_duration_s(_DUMMY, ceiling_s=1000.0)

    monkeypatch.setattr(detect_language, "read_audio_segment", _counting_unbounded_file()[0])
    high = _probe_duration_s(_DUMMY, ceiling_s=36000.0)

    assert high >= low


def test_probe_duration_ceiling_does_not_clip_a_normal_file(monkeypatch):
    # A file comfortably under the ceiling is governed by its real EOF, not the
    # guard -- the ceiling must be invisible in the common case. (A 500s file's
    # final bracket is the [480, 960) doubling interval, so the under-estimate is
    # ~498.75 -- a little under 2s short, hence the 498.0 lower bound.)
    monkeypatch.setattr(detect_language, "read_audio_segment", _fake_file(500.0))

    est = _probe_duration_s(_DUMMY, ceiling_s=36000.0)

    assert 498.0 < est < 500.0


# -- seam contract: probe_window is forwarded, not assumed --------------------


def test_probe_duration_forwards_probe_window_to_each_read(monkeypatch):
    # The configured ``probe_window`` must reach ``read_audio_segment`` as
    # ``duration_sec`` on every seek -- otherwise the EOF reads would use the
    # wrong clip length.
    seen_durations: list[float] = []

    def recording(*, audio_path, start_sec, duration_sec, sample_rate):
        seen_durations.append(duration_sec)
        return _AUDIO if start_sec < 100.0 else _SILENCE

    monkeypatch.setattr(detect_language, "read_audio_segment", recording)

    _probe_duration_s(_DUMMY, probe_window=0.25)

    assert seen_durations  # the probe actually read something
    assert all(d == 0.25 for d in seen_durations)
