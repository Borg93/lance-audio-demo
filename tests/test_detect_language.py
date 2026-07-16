"""Unit tests for the window-planning in language detection.

`_plan_sample_starts` decides where to sample each file for language ID. A
regression here silently degrades grouping — the exact symptom we're fixing
(long files judged by their intro, short files skipped) — so pin the
spread/clamp behavior. Pure math, no audio/GPU.

Run:  uv run pytest tests/test_detect_language.py -v
"""

from __future__ import annotations

import pytest

from rmedia.modalities.av.asr.detect_language import _plan_sample_starts


class TestPlanSampleStarts:
    def test_spreads_windows_across_the_whole_file(self):
        starts = _plan_sample_starts(5400.0, sample_seconds=20.0, num_windows=8)
        assert len(starts) == 8
        assert starts == sorted(starts)              # monotonic
        assert starts[0] >= 0.0
        assert starts[-1] + 20.0 <= 5400.0           # last clip stays inside the file
        assert starts[-1] > 0.5 * 5400.0             # genuinely reaches the back half

    def test_last_window_never_runs_past_eof(self):
        starts = _plan_sample_starts(100.0, sample_seconds=30.0, num_windows=12)
        assert all(s + 30.0 <= 100.0 for s in starts)

    def test_file_shorter_than_one_clip_collapses_to_single_start(self):
        assert _plan_sample_starts(15.0, sample_seconds=30.0, num_windows=8) == [0.0]

    def test_single_window(self):
        assert len(_plan_sample_starts(600.0, sample_seconds=20.0, num_windows=1)) == 1

    @pytest.mark.parametrize("bad_duration", [0.0, -10.0])
    def test_nonpositive_duration_is_safe(self, bad_duration):
        assert _plan_sample_starts(bad_duration, sample_seconds=20.0, num_windows=8) == [0.0]
