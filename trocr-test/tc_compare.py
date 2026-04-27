"""Tensor Core utilization comparison across optimization configurations.

Profiles a fixed-shape generate() call under several configurations and reports
how much of the device kernel time lands on Tensor Cores at each step of the
optimization ladder. Produces a side-by-side bar plot so the shift is visible.

Configurations profiled:
  1. broken-default       fp32, eager attention, beam=4, use_cache=False
                          (the model card defaults — what `from_pretrained` gives you)
  2. fixed-gen            fp32, eager attention, beam=1, use_cache=True
                          (just fix the broken generation_config)
  3. fp16-sdpa            fp16, encoder-SDPA + eager-decoder, beam=1, use_cache
                          (the dtype + native HF SDPA dispatch)
  4. bf16-sdpa            bf16, encoder-SDPA + eager-decoder, beam=1, use_cache
                          (bf16 over fp16 — same accuracy, faster on Blackwell)
  5. bf16-sdpa-dec-patch  bf16, encoder-SDPA + monkey-patched SDPA decoder
                          (final config — TrOCRAttention.forward routed through
                          F.scaled_dot_product_attention)

Each config runs the same fixed batch + token budget, no compile, so the
numbers are comparable. Output: console table + `results/tc_compare.png`.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import gc
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.trocr.modeling_trocr import (
    TrOCRAttention,
    TrOCRSinusoidalPositionalEmbedding,
)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda:0")
MODEL_ID = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
DATASET = "Riksarkivet/eval_htr_out_of_domain_lines"
BS = 32
MAX_NEW_TOKENS = 64
OUT_DIR = Path(__file__).parent / "results"

# Tensor Core kernel name patterns.
TC_MATMUL_PATTERNS = [
    r"tensorop", r"wmma", r"hmma", r"h884", r"h1688", r"h16816", r"h16832",
    r"hgemm", r"sm\d+_xmma",
]
TC_ATTN_PATTERNS = [r"flash_attn", r"flash_fwd", r"_fmha", r"fmha_cutlass"]
TC_MATMUL_RE = re.compile("|".join(TC_MATMUL_PATTERNS), re.IGNORECASE)
TC_ATTN_RE = re.compile("|".join(TC_ATTN_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# SDPA monkey-patch
# ---------------------------------------------------------------------------

ORIGINAL_TROCR_ATTENTION_FORWARD = TrOCRAttention.forward


def _trocr_sdpa(self, hidden_states, key_value_states=None, past_key_values=None,
                attention_mask=None, output_attentions=False, **kwargs):
    is_cross = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
    is_updated = False; curr_pkv = None
    if past_key_values is not None:
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            curr_pkv = past_key_values.cross_attention_cache if is_cross else past_key_values.self_attention_cache
        else:
            curr_pkv = past_key_values
    cur = key_value_states if is_cross else hidden_states
    if is_cross and past_key_values is not None and is_updated:
        k = curr_pkv.layers[self.layer_idx].keys
        v = curr_pkv.layers[self.layer_idx].values
    else:
        k = self.k_proj(cur).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(cur).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if past_key_values is not None:
            k, v = curr_pkv.update(k, v, self.layer_idx)
            if is_cross and isinstance(past_key_values, EncoderDecoderCache):
                past_key_values.is_updated[self.layer_idx] = True
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                         dropout_p=0.0, is_causal=False, scale=self.scaling)
    return self.out_proj(out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)), None


# ---------------------------------------------------------------------------
# Per-config build/run
# ---------------------------------------------------------------------------


@dataclass
class TCResult:
    name: str
    total_ms: float
    tc_matmul_ms: float
    tc_attn_ms: float
    cuda_ms: float
    tc_pct: float
    end_to_end_ms: float


def free_cuda():
    gc.collect()
    torch.cuda.synchronize(DEVICE)
    torch.cuda.empty_cache()


def build_model(*, dtype, attn_impl, use_cache, num_beams):
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation=attn_impl,
    ).to(DEVICE).eval()
    for m in model.modules():
        if isinstance(m, TrOCRSinusoidalPositionalEmbedding):
            m.weights = m.get_embedding(m.weights.size(0), m.embedding_dim, m.padding_idx).to(DEVICE, dtype)
    model.generation_config.use_cache = use_cache
    model.generation_config.num_beams = num_beams
    return processor, model


@torch.inference_mode()
def profile_config(name: str, *, dtype, attn_impl, use_cache, num_beams,
                   sdpa_decoder_patch: bool, images) -> TCResult:
    print(f"\n=== {name} ===")
    free_cuda()

    if sdpa_decoder_patch:
        TrOCRAttention.forward = _trocr_sdpa
    else:
        TrOCRAttention.forward = ORIGINAL_TROCR_ATTENTION_FORWARD

    processor, model = build_model(
        dtype=dtype, attn_impl=attn_impl, use_cache=use_cache, num_beams=num_beams,
    )

    pv = processor(images, return_tensors="pt").pixel_values.to(DEVICE, dtype)
    pv = pv.contiguous(memory_format=torch.channels_last)

    # Warmup
    model.generate(pv, max_new_tokens=MAX_NEW_TOKENS)
    torch.cuda.synchronize(DEVICE)

    # End-to-end wall time
    import time
    t0 = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    ) as prof:
        model.generate(pv, max_new_tokens=MAX_NEW_TOKENS)
        torch.cuda.synchronize(DEVICE)
    e2e_ms = (time.perf_counter() - t0) * 1000

    tc_matmul_us = tc_attn_us = cuda_us = 0.0
    for ev in prof.key_averages():
        dt = float(getattr(ev, "device_time_total", 0.0) or getattr(ev, "self_cuda_time_total", 0.0))
        if dt <= 0:
            continue
        n = ev.key
        if TC_ATTN_RE.search(n):
            tc_attn_us += dt
        elif TC_MATMUL_RE.search(n):
            tc_matmul_us += dt
        elif "kernel" in n.lower() or "void " in n:
            cuda_us += dt
    total = tc_matmul_us + tc_attn_us + cuda_us
    tc_pct = (tc_matmul_us + tc_attn_us) / total * 100 if total else 0.0

    res = TCResult(
        name=name,
        total_ms=total / 1000,
        tc_matmul_ms=tc_matmul_us / 1000,
        tc_attn_ms=tc_attn_us / 1000,
        cuda_ms=cuda_us / 1000,
        tc_pct=tc_pct,
        end_to_end_ms=e2e_ms,
    )
    print(f"  end-to-end : {res.end_to_end_ms:7.1f} ms")
    print(f"  device sum : {res.total_ms:7.1f} ms  "
          f"(TC matmul {res.tc_matmul_ms:5.1f} | TC attn {res.tc_attn_ms:5.1f} | CUDA {res.cuda_ms:5.1f})")
    print(f"  TC fraction: {res.tc_pct:5.1f} %")

    del model, processor
    free_cuda()
    return res


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot(results: list[TCResult]):
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    names = [r.name for r in results]
    matmul = [r.tc_matmul_ms for r in results]
    attn = [r.tc_attn_ms for r in results]
    cuda = [r.cuda_ms for r in results]
    pct = [r.tc_pct for r in results]
    e2e = [r.end_to_end_ms for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Tensor Core utilization across optimization ladder\n"
        f"TrOCR generate, bs={BS}, max_new_tokens={MAX_NEW_TOKENS}, "
        f"GPU: {torch.cuda.get_device_name(DEVICE)}",
        fontsize=11,
    )

    # Left: stacked bar of device kernel time (TC matmul + TC attn + CUDA core)
    x = list(range(len(names)))
    ax1.bar(x, matmul, label="TC matmul (cutlass/wmma)", color="#36B37E")
    ax1.bar(x, attn, bottom=matmul, label="TC attention (flash/fmha)", color="#4C9AFF")
    ax1.bar(x, cuda, bottom=[a + b for a, b in zip(matmul, attn)],
            label="CUDA core (LayerNorm/GeLU/cat)", color="#888888")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("device kernel time (ms)")
    ax1.set_title("Where the GPU spends its time")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.2, axis="y")
    # Annotate TC % on top of each bar
    for i, (m, a, c, p) in enumerate(zip(matmul, attn, cuda, pct)):
        ax1.text(i, m + a + c, f"TC: {p:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right: end-to-end wall time per config
    bars = ax2.bar(x, e2e, color=["#888888", "#FF8B00", "#9F7AEA", "#4C9AFF", "#36B37E"][: len(results)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("end-to-end wall time (ms)")
    ax2.set_title(f"End-to-end generate() time (bs={BS}, max_new_tokens={MAX_NEW_TOKENS})")
    ax2.grid(alpha=0.2, axis="y")
    for b, v in zip(bars, e2e):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.0f}",
                 ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT_DIR / "tc_compare.png"
    fig.savefig(p, dpi=140)
    print(f"\nplot saved: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)} (sm_{torch.cuda.get_device_capability(DEVICE)[0]}{torch.cuda.get_device_capability(DEVICE)[1]})\n")

    ds = load_dataset(DATASET, split="test", streaming=True)
    images = []
    for ex in ds:
        images.append(ex["image"].convert("RGB"))
        if len(images) >= BS:
            break

    configs = [
        # 1. The model's broken default config (what `from_pretrained` gives you).
        dict(name="1. broken-default\n(fp32, eager, beam=4, no-cache)",
             dtype=torch.float32, attn_impl="eager",
             use_cache=False, num_beams=4, sdpa_decoder_patch=False),
        # 2. Just fix the generation_config issues.
        dict(name="2. fixed-gen\n(fp32, eager, beam=1, +cache)",
             dtype=torch.float32, attn_impl="eager",
             use_cache=True, num_beams=1, sdpa_decoder_patch=False),
        # 3. fp16 + native HF SDPA on encoder.
        dict(name="3. fp16-sdpa\n(fp16, enc=sdpa, dec=eager)",
             dtype=torch.float16,
             attn_impl={"encoder": "sdpa", "decoder": "eager"},
             use_cache=True, num_beams=1, sdpa_decoder_patch=False),
        # 4. bf16 swap.
        dict(name="4. bf16-sdpa\n(bf16, enc=sdpa, dec=eager)",
             dtype=torch.bfloat16,
             attn_impl={"encoder": "sdpa", "decoder": "eager"},
             use_cache=True, num_beams=1, sdpa_decoder_patch=False),
        # 5. + monkey-patched decoder SDPA — the final stack.
        dict(name="5. bf16-sdpa-dec-patch\n(bf16, enc+dec sdpa)",
             dtype=torch.bfloat16,
             attn_impl={"encoder": "sdpa", "decoder": "eager"},
             use_cache=True, num_beams=1, sdpa_decoder_patch=True),
    ]

    results = [profile_config(images=images, **cfg) for cfg in configs]

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    (OUT_DIR / "tc_compare.json").write_text(json.dumps([asdict(r) for r in results], indent=2))

    print("\n\n=== summary ===")
    header = f"{'config':<40} {'e2e(ms)':>9} {'dev(ms)':>9} {'matmul':>8} {'attn':>8} {'cuda':>8} {'TC%':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        first_line = r.name.split("\n")[0]
        print(f"{first_line:<40} {r.end_to_end_ms:>9.1f} {r.total_ms:>9.1f} "
              f"{r.tc_matmul_ms:>8.1f} {r.tc_attn_ms:>8.1f} {r.cuda_ms:>8.1f} {r.tc_pct:>6.1f}")

    plot(results)


if __name__ == "__main__":
    main()
