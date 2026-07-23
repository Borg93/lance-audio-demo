"""Unit tests for ``detect_and_sort`` orchestration + aggregation.

``detect_and_sort`` samples a few clips per file, runs a language probe over
each window, votes the results into one language per file, maps the ISO 639-3
code to 639-1, and (optionally) moves the file into ``<audio_dir>/<lang>/``.

The window-planning math (``_plan_sample_starts``) is pinned separately in
``tests/test_detect_language.py`` — this module does NOT re-test it. Here we
pin the parts that would *silently mislabel a batch*: vote aggregation, the
639-3→639-1 mapping, file filtering, per-window read resilience, and the
move/dry-run side effects.

Everything that touches a model or ffmpeg is patched at the seam (the symbol as
imported into ``runners.asr.detect_language``), so the suite is fast + offline:

- ``_mms_probe`` / ``_whisper_probe`` → a fake probe ``clip -> (lang_raw, prob)``
  we control, so no model is ever loaded.
- ``read_audio_segment`` → a non-empty array, so no clip is skipped as silent.
- ``_probe_duration_s`` → a fixed positive duration, so windows are planned.

Run:  uv run pytest tests/test_asr_detect.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from runners.asr import detect_language as dl
from runners.asr.detect_language import detect_and_sort

from ratch.errors import RatchError

# Default model id routes to the *whisper* probe branch; an mms-lid id routes to
# the *mms* branch. We patch both so either branch yields our fake probe.
WHISPER_MODEL = "openai/whisper-large-v3"
MMS_MODEL = "facebook/mms-lid-256"


# ──────────────────────────── fixtures / helpers ─────────────────────────────


def _touch(directory: Path, name: str) -> Path:
    """Create a real (empty) file inside ``directory`` and return its path."""
    p = directory / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


@pytest.fixture
def patched_io(monkeypatch):
    """Neutralise the two ffmpeg seams so every planned window is sampled.

    ``read_audio_segment`` returns a non-empty array (clip not skipped) and
    ``_probe_duration_s`` returns a positive, short duration. A short duration
    collapses ``_plan_sample_starts`` to a single window, which keeps the vote
    arithmetic in each test trivial to reason about unless a test overrides it.
    """
    monkeypatch.setattr(
        dl, "read_audio_segment", lambda **kw: np.ones(16000, dtype=np.float32)
    )
    monkeypatch.setattr(dl, "_probe_duration_s", lambda path, **kw: 10.0)


def _set_probe(monkeypatch, probe):
    """Patch BOTH backend builders so they return ``probe`` regardless of model.

    Patching the builders (not just one) means the test is independent of which
    ``model`` arg routes where — no real model is constructed either way.
    """
    builder = lambda *a, **k: probe  # noqa: E731 — tiny seam shim
    monkeypatch.setattr(dl, "_mms_probe", builder)
    monkeypatch.setattr(dl, "_whisper_probe", builder)


def _const_probe(lang_raw, prob):
    """A probe that returns the same ``(lang_raw, prob)`` for every clip."""
    return lambda clip: (lang_raw, prob)


# ────────────────────────────── vote aggregation ────────────────────────────


def test_unanimous_windows_yield_that_language(patched_io, monkeypatch, tmp_path):
    """Every window voting 'swe' → result is ('sv', that probability)."""
    _set_probe(monkeypatch, _const_probe("swe", 0.9))
    _touch(tmp_path, "talk.mp4")

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    assert results == {"talk.mp4": ("sv", pytest.approx(0.9))}


def test_highest_summed_probability_language_wins_when_windows_disagree(
    patched_io, monkeypatch, tmp_path
):
    """Aggregation sums prob per language; the larger SUM wins, not the count.

    Across 3 windows: 'eng' appears twice (0.30 + 0.30 = 0.60) and 'swe' once
    (0.95). 'swe' has fewer votes but the higher summed probability, so it must
    win — and prob_avg averages over all sampled windows (0.95 / 3).
    """
    monkeypatch.setattr(
        dl, "_plan_sample_starts", lambda *a, **k: [0.0, 1.0, 2.0]
    )
    seq = iter([("eng", 0.30), ("swe", 0.95), ("eng", 0.30)])
    _set_probe(monkeypatch, lambda clip: next(seq))
    _touch(tmp_path, "mixed.mp4")

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    lang, prob = results["mixed.mp4"]
    assert lang == "sv"
    assert prob == pytest.approx(0.95 / 3)


def test_prob_avg_divides_summed_probability_by_windows_sampled(
    patched_io, monkeypatch, tmp_path
):
    """prob_avg = sum(prob for winning lang) / number of sampled windows."""
    monkeypatch.setattr(dl, "_plan_sample_starts", lambda *a, **k: [0.0, 1.0])
    seq = iter([("swe", 0.6), ("swe", 0.8)])
    _set_probe(monkeypatch, lambda clip: next(seq))
    _touch(tmp_path, "two.mp4")

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    # (0.6 + 0.8) summed for 'swe', averaged over 2 sampled windows = 0.7.
    assert results["two.mp4"][1] == pytest.approx(0.7)


# ────────────────────────── ISO 639-3 → 639-1 mapping ───────────────────────


@pytest.mark.parametrize(
    ("raw_code", "expected"),
    [
        ("swe", "sv"),  # mapped
        ("eng", "en"),  # mapped
        ("nor", "no"),  # mapped
        ("xyz", "xyz"),  # unmapped → passes through unchanged
        ("sv", "sv"),  # already 639-1 (whisper output) → passes through
    ],
)
def test_language_code_mapping(patched_io, monkeypatch, tmp_path, raw_code, expected):
    """639-3 codes map to 639-1; anything unmapped passes through verbatim."""
    _set_probe(monkeypatch, _const_probe(raw_code, 0.5))
    _touch(tmp_path, "clip.wav")

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    assert results["clip.wav"][0] == expected


# ─────────────────────────────── file filtering ─────────────────────────────


@pytest.mark.parametrize(
    ("name", "kept"),
    [
        ("song.mp3", True),  # audio ext
        ("movie.MP4", True),  # ext match is case-insensitive
        ("clip.flac", True),
        (".hidden.wav", False),  # dotfile ignored
        ("notes.txt", False),  # non-audio ext ignored
        ("poster.jpg", False),  # image ignored
        ("README", False),  # no extension
    ],
)
def test_file_filtering_by_name_and_extension(
    patched_io, monkeypatch, tmp_path, name, kept
):
    """Only top-level, non-dotfile, audio/video-extension files are processed."""
    _set_probe(monkeypatch, _const_probe("eng", 0.5))
    _touch(tmp_path, name)

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    assert (name in results) is kept


def test_subdirectories_are_ignored(patched_io, monkeypatch, tmp_path):
    """Audio files nested in subdirs are not picked up (top-level only)."""
    _set_probe(monkeypatch, _const_probe("eng", 0.5))
    _touch(tmp_path, "top.wav")
    _touch(tmp_path / "nested", "deep.wav")  # one level down — should be skipped

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    assert set(results) == {"top.wav"}


# ───────────────────────────── move / dry-run side effects ──────────────────


def test_move_relocates_file_into_language_subdir(patched_io, monkeypatch, tmp_path):
    """move=True moves the file into ``audio_dir/<lang>/`` and clears the root."""
    _set_probe(monkeypatch, _const_probe("swe", 0.9))
    src = _touch(tmp_path, "talk.mp4")

    detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=True)

    assert not src.exists()
    assert (tmp_path / "sv" / "talk.mp4").is_file()


def test_dry_run_does_not_move_files(patched_io, monkeypatch, tmp_path):
    """dry_run=True still reports results but leaves files where they were."""
    _set_probe(monkeypatch, _const_probe("swe", 0.9))
    src = _touch(tmp_path, "talk.mp4")

    results = detect_and_sort(
        audio_dir=tmp_path, model=WHISPER_MODEL, move=True, dry_run=True
    )

    assert results == {"talk.mp4": ("sv", pytest.approx(0.9))}
    assert src.is_file()
    assert not (tmp_path / "sv").exists()


def test_move_false_does_not_move_files(patched_io, monkeypatch, tmp_path):
    """move=False detects but never relocates."""
    _set_probe(monkeypatch, _const_probe("swe", 0.9))
    src = _touch(tmp_path, "talk.mp4")

    detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    assert src.is_file()
    assert not (tmp_path / "sv").exists()


# ─────────────────────────────── boundaries / errors ────────────────────────


def test_file_with_zero_duration_is_skipped(monkeypatch, tmp_path):
    """A file whose probe reports 0s of audio is dropped (not in results)."""
    monkeypatch.setattr(
        dl, "read_audio_segment", lambda **kw: np.ones(16000, dtype=np.float32)
    )
    monkeypatch.setattr(dl, "_probe_duration_s", lambda path, **kw: 0.0)
    _set_probe(monkeypatch, _const_probe("swe", 0.9))
    _touch(tmp_path, "silent.wav")

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    assert results == {}


def test_window_that_reads_empty_clip_is_not_counted(
    patched_io, monkeypatch, tmp_path
):
    """An empty (size==0) clip is skipped; with no usable windows → no result.

    Duration is positive so windows ARE planned, but every read comes back
    empty, so the file yields zero votes and is dropped.
    """
    monkeypatch.setattr(
        dl, "read_audio_segment", lambda **kw: np.empty(0, dtype=np.float32)
    )
    _set_probe(monkeypatch, _const_probe("swe", 0.9))
    _touch(tmp_path, "empty.wav")

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    assert results == {}


def test_window_read_error_is_skipped_while_other_windows_still_vote(
    patched_io, monkeypatch, tmp_path
):
    """A single window whose read raises is skipped; the rest still vote.

    Pins the per-window ``try/except`` resilience: a regression that let one
    failed ffmpeg seek abort the whole file would drop (or mislabel) it. Three
    windows are planned, the 2nd read raises, so only 2 windows are sampled —
    the result must still resolve, with prob_avg dividing by the 2 that worked
    (sum 1.6 / 2 = 0.8), proving the failed window was excluded from BOTH the
    numerator and the denominator.
    """
    monkeypatch.setattr(dl, "_plan_sample_starts", lambda *a, **k: [0.0, 1.0, 2.0])

    state = {"reads": 0}

    def flaky_read(**kw):
        state["reads"] += 1
        if state["reads"] == 2:  # the middle window fails to decode
            raise RuntimeError("ffmpeg fast-seek failed on this window")
        return np.ones(16000, dtype=np.float32)

    monkeypatch.setattr(dl, "read_audio_segment", flaky_read)
    _set_probe(monkeypatch, _const_probe("swe", 0.8))
    _touch(tmp_path, "flaky.mp4")

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    lang, prob = results["flaky.mp4"]
    assert lang == "sv"
    assert prob == pytest.approx(0.8)  # (0.8 + 0.8) / 2 sampled, not / 3 planned


def test_empty_directory_returns_empty_dict(patched_io, monkeypatch, tmp_path):
    """A directory with no matching files yields ``{}`` (and no probe is built)."""
    def _boom(*a, **k):
        raise AssertionError("probe builder must not run when there are no files")

    monkeypatch.setattr(dl, "_mms_probe", _boom)
    monkeypatch.setattr(dl, "_whisper_probe", _boom)
    _touch(tmp_path, "ignore.txt")  # present, but filtered out

    results = detect_and_sort(audio_dir=tmp_path, model=WHISPER_MODEL, move=False)

    assert results == {}


def test_audio_dir_not_a_directory_raises_domain_error(tmp_path):
    """A non-directory ``audio_dir`` is a domain error (the CLI maps it to exit 1)."""
    missing = tmp_path / "nope"

    with pytest.raises(RatchError):
        detect_and_sort(audio_dir=missing, model=WHISPER_MODEL, move=False)


def test_mms_model_routes_through_the_mms_branch(patched_io, monkeypatch, tmp_path):
    """An mms-lid model id builds via ``_mms_probe`` (whisper builder untouched).

    Guards the model-routing fork: a regression that always called the whisper
    builder would mislabel every MMS batch.
    """
    monkeypatch.setattr(dl, "_mms_probe", lambda *a, **k: _const_probe("swe", 0.8))

    def _whisper_should_not_run(*a, **k):
        raise AssertionError("mms model must not route through the whisper branch")

    monkeypatch.setattr(dl, "_whisper_probe", _whisper_should_not_run)
    _touch(tmp_path, "talk.wav")

    results = detect_and_sort(audio_dir=tmp_path, model=MMS_MODEL, move=False)

    assert results["talk.wav"] == ("sv", pytest.approx(0.8))
