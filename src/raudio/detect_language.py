"""Language detection pre-step.

Samples a short clip from each audio/video file in a directory, runs a
language classifier, and (optionally) moves each file into
``<audio_dir>/<lang>/`` so the transcribe pipeline can be run per-language
with the correct wav2vec2 emissions model.

**Two backends supported** — chosen by the ``model`` arg:

- ``facebook/mms-lid-*`` (default: ``facebook/mms-lid-256``) — Meta's MMS LID
  classifier, a wav2vec2-SequenceClassification model purpose-built for
  language identification across 256 languages. State-of-the-art for this
  exact task.
- ``openai/whisper-*`` / other Whisper models — the built-in Whisper LID head
  via ctranslate2. Lower accuracy than MMS for sv-vs-en edge cases but
  useful as a fallback.

**Never** pass a language-fine-tuned Whisper (``KBLab/kb-whisper-large``) —
those over-predict their training language (every file comes back as `sv`).

Exposed as ``raudio detect-language …`` via :mod:`raudio.cli`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from easytranscriber.audio import read_audio_segment

from .transcribe import DEFAULT_EMISSIONS_MODEL


# Files we actually want to sort. Everything else (images, docs, …) is ignored.
AUDIO_VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus"}
)


# MMS-LID emits ISO 639-3 codes (``eng``, ``swe``, ``nor`` …). We route files
# into directories named with the 2-letter codes used by our transcribe
# pipeline (``en``, ``sv``, …), so map the common ones.
ISO_639_3_TO_1: dict[str, str] = {
    "swe": "sv", "eng": "en", "nor": "no", "dan": "da", "fin": "fi",
    "isl": "is", "deu": "de", "ger": "de", "fra": "fr", "spa": "es",
    "ita": "it", "nld": "nl", "pol": "pl", "por": "pt", "rus": "ru",
    "ara": "ar", "zho": "zh", "cmn": "zh", "jpn": "ja", "kor": "ko",
    "tur": "tr", "ukr": "uk", "ell": "el", "gre": "el", "ces": "cs",
    "cze": "cs", "hun": "hu", "ron": "ro", "rum": "ro", "est": "et",
    "lav": "lv", "lit": "lt", "heb": "he", "hin": "hi", "ind": "id",
    "tha": "th", "vie": "vi",
}


# Sample at three offsets and pick by summed probability. Beats one 30s
# window that might happen to be silence / music / leader tones.
OFFSET_MULTIPLIERS: tuple[float, ...] = (1.0, 3.0, 5.0)


def detect_and_sort(
    *,
    audio_dir: Path,
    model: str = "openai/whisper-large-v3",
    cache_dir: Path = Path("models"),
    sample_seconds: float = 30.0,
    sample_offset: float = 60.0,
    device: str = "cuda",
    move: bool = True,
    dry_run: bool = False,
) -> dict[str, tuple[str, float]]:
    """Detect the spoken language of each top-level file in ``audio_dir``.

    Parameters
    ----------
    audio_dir
        Directory containing mixed-language audio/video files.
    model
        Hugging Face model id. ``facebook/mms-lid-*`` (default) or a
        multilingual Whisper like ``openai/whisper-large-v3``.
    sample_seconds
        Length of each audio clip fed to the classifier (30s default).
    sample_offset
        Base offset. We actually sample at 1x/3x/5x this into the file,
        so default 60s → 60s, 180s, 300s. Skips silent intros / archive
        voiceovers / leader tones.
    move
        If ``True`` (default), move each file into ``audio_dir/<lang>/``.
    dry_run
        Print what would happen without moving files.

    Returns
    -------
    dict
        ``{filename: (lang_code, probability)}``. ``lang_code`` is the
        2-letter ISO 639-1 code whenever we can map it from MMS's 639-3
        output; otherwise the raw code passes through.
    """
    if not audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    files = sorted(
        f for f in audio_dir.iterdir()
        if f.is_file()
        and not f.name.startswith(".")
        and f.suffix.lower() in AUDIO_VIDEO_EXTS
    )
    if not files:
        print(f"No audio/video files found directly in {audio_dir}/ (subdirs ignored).")
        return {}

    if model.startswith("facebook/mms-lid"):
        probe = _mms_probe(model, cache_dir, device)
    else:
        probe = _whisper_probe(model, cache_dir, device)

    print(
        f"→ Detecting language in {len(files)} file(s) with {model} "
        f"(samples: {sample_seconds}s at offsets "
        f"{[sample_offset * m for m in OFFSET_MULTIPLIERS]}):"
    )

    results: dict[str, tuple[str, float]] = {}
    for f in files:
        votes: dict[str, float] = {}
        for mult in OFFSET_MULTIPLIERS:
            off = sample_offset * mult
            try:
                audio = read_audio_segment(
                    audio_path=f,
                    start_sec=off,
                    duration_sec=sample_seconds,
                    sample_rate=16000,
                )
            except Exception:
                continue
            if audio.size == 0:
                continue
            lang_raw, prob = probe(audio)
            votes[lang_raw] = votes.get(lang_raw, 0.0) + prob
        if not votes:
            print(f"  ⚠ {f.name}: no usable audio at any sample offset; skipping")
            continue

        lang_raw = max(votes, key=lambda k: votes[k])
        prob_avg = votes[lang_raw] / len(OFFSET_MULTIPLIERS)
        lang = ISO_639_3_TO_1.get(lang_raw, lang_raw)
        results[f.name] = (lang, prob_avg)

        supported = "✓" if lang in DEFAULT_EMISSIONS_MODEL else "!"
        print(f"  {supported} {f.name}: {lang} (p={prob_avg:.3f})")

        if move and not dry_run:
            target_dir = audio_dir / lang
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(f), str(target_dir / f.name))

    if move and not dry_run:
        print(f"✓ Sorted {len(results)} file(s) into {audio_dir}/<lang>/ subfolders.")
        print("  Next:")
        for lang in sorted({v[0] for v in results.values()}):
            if lang in DEFAULT_EMISSIONS_MODEL:
                print(
                    f"    raudio transcribe --audio-dir {audio_dir}/{lang} "
                    f"--language {lang} --output-root output/{lang}"
                )
            else:
                print(
                    f"    # {lang}: no default wav2vec2 emissions model in "
                    f"DEFAULT_EMISSIONS_MODEL; pass --emissions-model yourself."
                )
    elif dry_run:
        print("(dry run — no files moved)")

    return results


# ──────────────────────── Backend probes ─────────────────────────────────────


def _mms_probe(model_id: str, cache_dir: Path, device: str):
    """Build an ``(audio)→(lang_639_3, prob)`` probe using MMS-LID."""
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    print(f"→ Loading {model_id} on {device}…")
    processor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=str(cache_dir))
    mdl = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_id, cache_dir=str(cache_dir)
    ).to(device).eval()
    id2label = mdl.config.id2label  # int → "eng"/"swe"/…

    def probe(audio):
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = mdl(**inputs).logits  # (1, 256)
        probs = torch.softmax(logits, dim=-1)[0]
        top = int(torch.argmax(probs).item())
        return id2label[top], float(probs[top].item())

    return probe


def _whisper_probe(model_id: str, cache_dir: Path, device: str):
    """Build an ``(audio)→(lang_639_1, prob)`` probe using Whisper's LID head."""
    import ctranslate2
    from easytranscriber.asr.ct2 import detect_language
    from easytranscriber.utils import hf_to_ct2_converter
    from transformers import WhisperProcessor

    print(f"→ Loading {model_id} on {device}…")
    model_path = hf_to_ct2_converter(model_id, cache_dir=str(cache_dir))
    whisper = ctranslate2.models.Whisper(model_path.as_posix(), device=device)
    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))

    def probe(audio):
        features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.numpy()
        features_ct2 = ctranslate2.StorageView.from_array(features)
        top = detect_language(whisper, features_ct2)[0]
        # top["language"] is ``<|sv|>`` — strip braces; it's already 639-1
        return top["language"].strip("<|>"), float(top["probability"])

    return probe
