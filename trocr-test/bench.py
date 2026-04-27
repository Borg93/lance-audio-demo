"""Benchmark TrOCR optimizations on Riksarkivet/eval_htr_out_of_domain_lines.

Compares latency, throughput, CER, WER, and VRAM across several configurations
to verify that bf16 / SDPA / torch.compile do not degrade accuracy.

Usage:
    uv run --no-sync python trocr-test/bench.py --n 64 --bs 8
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Pin physical GPU 2; expose it as cuda:0 to torch. Must happen before `import torch`
# because some codepaths (compile, lazy buffers) materialize tensors on the default
# device and choke on a non-default cuda index with this checkpoint's legacy buffers.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
import torch.nn.functional as F
from contextlib import contextmanager
from datasets import load_dataset
from jiwer import cer, wer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.trocr.modeling_trocr import TrOCRAttention
from transformers.utils import is_flash_attn_2_available

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE_ID = 0  # remapped from physical GPU 2 via CUDA_VISIBLE_DEVICES
DEVICE = torch.device(f"cuda:{DEVICE_ID}")
MODEL_ID = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
DATASET = "Riksarkivet/eval_htr_out_of_domain_lines"
OUT_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    batch_size: int
    dtype: str
    n: int
    total_s: float
    per_image_ms: float
    img_per_s: float
    cer_pct: float
    wer_pct: float
    vram_mib: float
    sample_ref: str
    sample_hyp: str


def free_cuda() -> None:
    gc.collect()
    torch.cuda.synchronize(DEVICE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)


def load_samples(n: int) -> list[tuple]:
    ds = load_dataset(DATASET, split="test", streaming=True)
    out: list[tuple] = []
    for ex in ds:
        out.append((ex["image"].convert("RGB"), ex["transcription"]))
        if len(out) >= n:
            break
    return out


def build_model(dtype: torch.dtype, attn_impl: Any) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation=attn_impl
    ).to(DEVICE).eval()

    # TrOCRSinusoidalPositionalEmbedding stores its sinusoidal table as a plain
    # attribute (not a registered buffer), so `.to(device)` skips it and the
    # meta-device init path leaves it stranded. Rebuild it on the right device.
    from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding
    for module in model.modules():
        if isinstance(module, TrOCRSinusoidalPositionalEmbedding):
            module.weights = module.get_embedding(
                module.weights.size(0), module.embedding_dim, module.padding_idx
            ).to(DEVICE, dtype)
    return processor, model


# ---------------------------------------------------------------------------
# SDPA monkey-patch for TrOCRAttention.forward
#
# Upstream TrOCRAttention uses raw `bmm + softmax + bmm` with no dispatch hook,
# so `attn_implementation="sdpa"` is silently ignored on the decoder. This
# patch replaces the body with one fused `F.scaled_dot_product_attention` call.
# It is mathematically equivalent (within bf16 ULP) — same softmax(QK/√d)V —
# and preserves the upstream KV-cache routing exactly.
# ---------------------------------------------------------------------------


def _trocr_attention_sdpa_forward(
    self,
    hidden_states,
    key_value_states=None,
    past_key_values=None,
    attention_mask=None,
    output_attentions=False,
    **kwargs,
):
    is_cross = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # Project Q. Do not pre-scale — SDPA applies `scale=` internally.
    q = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

    # Mirror upstream cache routing.
    is_updated = False
    curr_pkv = None
    if past_key_values is not None:
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            curr_pkv = past_key_values.cross_attention_cache if is_cross else past_key_values.self_attention_cache
        else:
            curr_pkv = past_key_values

    current_states = key_value_states if is_cross else hidden_states

    if is_cross and past_key_values is not None and is_updated:
        k = curr_pkv.layers[self.layer_idx].keys
        v = curr_pkv.layers[self.layer_idx].values
    else:
        k = self.k_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if past_key_values is not None:
            k, v = curr_pkv.update(k, v, self.layer_idx)
            if is_cross and isinstance(past_key_values, EncoderDecoderCache):
                past_key_values.is_updated[self.layer_idx] = True

    # SDPA: q,k,v in (bsz, num_heads, seq, head_dim). attn_mask is additive (bsz,1,tgt,src).
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
        scale=self.scaling,
    )
    out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
    return self.out_proj(out), None


@contextmanager
def _noop_ctx():
    yield


@contextmanager
def sdpa_decoder_attention():
    """Swap TrOCRAttention.forward with the SDPA implementation for the duration."""
    original = TrOCRAttention.forward
    TrOCRAttention.forward = _trocr_attention_sdpa_forward
    try:
        yield
    finally:
        TrOCRAttention.forward = original


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_config(
    name: str,
    samples: list[tuple],
    *,
    batch_size: int,
    dtype: torch.dtype,
    attn_impl: Any,
    compile_decoder: bool = False,
    use_sdpa_decoder: bool = False,
    cache_static: bool = False,
    max_new_tokens: int = 128,
) -> Result:
    print(f"\n=== {name} ===")
    free_cuda()

    # Patch context: SDPA decoder is per-config so we can A/B test it.
    patch_ctx = sdpa_decoder_attention() if use_sdpa_decoder else _noop_ctx()

    with patch_ctx:
        processor, model = build_model(dtype, attn_impl)

        if cache_static:
            # StaticCache requires the cache size to be known up front. generate()
            # uses (input_seq_len + max_new_tokens) — for TrOCR the prompt is
            # always 1 (decoder_start_token_id), so this is just max_new_tokens+1.
            model.generation_config.cache_implementation = "static"

        if compile_decoder:
            # With static cache the KV shape never changes mid-decode, so the
            # cudagraph is recorded once and replayed every step. Without it,
            # compile keeps recompiling for each new KV length.
            model.decoder.forward = torch.compile(
                model.decoder.forward,
                mode="reduce-overhead",
                dynamic=False if cache_static else True,
                fullgraph=False,
            )

        images = [s[0] for s in samples]
        refs = [s[1] for s in samples]
        preds: list[str] = []

        # Warmup (triggers compile if enabled).
        pv = processor(images[:1], return_tensors="pt").pixel_values.to(DEVICE, dtype)
        model.generate(pv, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize(DEVICE)

        t0 = time.perf_counter()
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            pv = processor(chunk, return_tensors="pt").pixel_values.to(DEVICE, dtype)
            ids = model.generate(pv, max_new_tokens=max_new_tokens)
            preds.extend(processor.batch_decode(ids, skip_special_tokens=True))
        torch.cuda.synchronize(DEVICE)
        elapsed = time.perf_counter() - t0

    res = Result(
        name=name,
        batch_size=batch_size,
        dtype=str(dtype).split(".")[-1],
        n=len(images),
        total_s=elapsed,
        per_image_ms=elapsed / len(images) * 1000,
        img_per_s=len(images) / elapsed,
        cer_pct=cer(refs, preds) * 100,
        wer_pct=wer(refs, preds) * 100,
        vram_mib=torch.cuda.max_memory_allocated(DEVICE) / 1024**2,
        sample_ref=refs[0][:120],
        sample_hyp=preds[0][:120],
    )

    print(
        f"  N={res.n}, bs={res.batch_size}, dtype={res.dtype}\n"
        f"  total      : {res.total_s:6.2f} s\n"
        f"  per-image  : {res.per_image_ms:6.1f} ms  ({res.img_per_s:5.1f} img/s)\n"
        f"  CER        : {res.cer_pct:5.2f} %\n"
        f"  WER        : {res.wer_pct:5.2f} %\n"
        f"  peak VRAM  : {res.vram_mib:6.0f} MiB\n"
        f"  ref[0]     : {res.sample_ref}\n"
        f"  hyp[0]     : {res.sample_hyp}"
    )

    del model, processor
    free_cuda()
    return res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results: list[Result], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True, parents=True)
    labels = [r.name for r in results]
    short = [l.split(" (")[0] for l in labels]
    palette = ["#888888", "#4C9AFF", "#36B37E", "#FF8B00", "#9F7AEA", "#E94B6E"]
    colors = [palette[i % len(palette)] for i in range(len(results))]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"TrOCR optimization sweep — {MODEL_ID}\n"
        f"dataset: {DATASET} (n={results[0].n}), GPU: {torch.cuda.get_device_name(DEVICE)}",
        fontsize=11,
    )

    def bar(ax, values, ylabel, title, fmt="{:.1f}"):
        bars = ax.bar(short, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    bar(axes[0, 0], [r.per_image_ms for r in results], "ms / image", "Latency (lower is better)")
    bar(axes[0, 1], [r.img_per_s for r in results], "images / second", "Throughput (higher is better)")
    bar(axes[1, 0], [r.cer_pct for r in results], "CER %", "Character Error Rate (must not regress)", fmt="{:.2f}")
    bar(axes[1, 1], [r.wer_pct for r in results], "WER %", "Word Error Rate (must not regress)", fmt="{:.2f}")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = out_dir / "benchmark.png"
    fig.savefig(plot_path, dpi=140)
    print(f"\nplot saved: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64, help="number of eval samples")
    p.add_argument("--bs", type=int, default=8, help="batch size for fast/compile configs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}  (cuda:{DEVICE_ID})")
    print(f"Loading {args.n} samples from {DATASET} ...")
    samples = load_samples(args.n)
    print(f"got {len(samples)} samples")

    enc_attn = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    dec_attn = "flash_attention_2" if is_flash_attn_2_available() else "eager"

    fast_attn = {"encoder": enc_attn, "decoder": dec_attn}
    bs = args.bs

    configs = [
        # Reference: pure stock fp32, no batching, eager attention.
        dict(
            name="baseline (fp32, eager, bs=1)",
            batch_size=1, dtype=torch.float32, attn_impl="eager",
        ),
        # Fast path: bf16, ViT-encoder SDPA, batched. Decoder still uses eager
        # `bmm + softmax + bmm` because TrOCRAttention has no SDPA dispatch.
        # bf16 chosen over bf16: identical accuracy, ~14% faster on Blackwell.
        dict(
            name=f"fast (bf16, enc=sdpa, dec=eager, bs={bs})",
            batch_size=bs, dtype=torch.bfloat16, attn_impl=fast_attn,
        ),
        # Adds torch.compile on top of the fast path. With DynamicCache the KV
        # shape changes every step, so cudagraph keeps re-recording — usually
        # neutral or slightly slower at bs>=8.
        dict(
            name=f"fast+compile (bf16, dynamic cache, bs={bs})",
            batch_size=bs, dtype=torch.bfloat16, attn_impl=fast_attn,
            compile_decoder=True,
        ),
        # Adds the SDPA monkey-patch on TrOCRAttention.forward — fuses the
        # bmm/softmax/bmm into one F.scaled_dot_product_attention call. Same
        # math, fewer kernels.
        dict(
            name=f"fast+sdpa-dec (bf16, monkey-patched dec, bs={bs})",
            batch_size=bs, dtype=torch.bfloat16, attn_impl=fast_attn,
            use_sdpa_decoder=True,
        ),
        # SDPA-decoder + StaticCache. Fixed-shape KV buffer means generate has
        # one shape across all decode steps, which is the precondition for
        # compile to actually pay off.
        dict(
            name=f"fast+sdpa-dec+static-cache (bs={bs})",
            batch_size=bs, dtype=torch.bfloat16, attn_impl=fast_attn,
            use_sdpa_decoder=True, cache_static=True,
        ),
        # Everything stacked: SDPA decoder + StaticCache + torch.compile. This
        # is where compile is supposed to actually win.
        dict(
            name=f"fast+sdpa-dec+static-cache+compile (bs={bs})",
            batch_size=bs, dtype=torch.bfloat16, attn_impl=fast_attn,
            use_sdpa_decoder=True, cache_static=True, compile_decoder=True,
        ),
    ]

    results = [run_config(samples=samples, **cfg) for cfg in configs]

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    json_path = OUT_DIR / "benchmark.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\njson saved: {json_path}")

    print("\n\n=== summary ===")
    header = f"{'config':<60} {'ms/img':>8} {'img/s':>8} {'CER%':>7} {'WER%':>7} {'VRAM':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<60} {r.per_image_ms:>8.1f} {r.img_per_s:>8.1f} "
            f"{r.cer_pct:>7.2f} {r.wer_pct:>7.2f} {r.vram_mib:>8.0f}"
        )

    plot_results(results, OUT_DIR)


if __name__ == "__main__":
    main()
