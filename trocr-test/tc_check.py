"""Tensor Core utilization check.

Profiles one TrOCR generate() call and reports kernel-level time spent in
Tensor Core math vs CUDA-core math, then saves a breakdown plot.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import re
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

# Tensor Core kernel name patterns. Two buckets:
#  - "matmul" — cutlass GEMMs (the classic tensor-op patterns)
#  - "attn"   — Flash-Attention / xformers fmha (also use TC under the hood)
TC_MATMUL_PATTERNS = [
    r"tensorop", r"wmma", r"hmma", r"h884", r"h1688", r"h16816", r"h16832",
    r"hgemm", r"sm\d+_xmma",
]
TC_ATTN_PATTERNS = [r"flash_attn", r"flash_fwd", r"_fmha", r"fmha_cutlass"]
TC_MATMUL_RE = re.compile("|".join(TC_MATMUL_PATTERNS), re.IGNORECASE)
TC_ATTN_RE = re.compile("|".join(TC_ATTN_PATTERNS), re.IGNORECASE)


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


TrOCRAttention.forward = _trocr_sdpa


def main():
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"compute capability: {torch.cuda.get_device_capability(DEVICE)}\n")

    ds = load_dataset(DATASET, split="test", streaming=True)
    images = []
    for ex in ds:
        images.append(ex["image"].convert("RGB"))
        if len(images) >= BS:
            break

    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
        attn_implementation={"encoder": "sdpa", "decoder": "eager"},
    ).to(DEVICE).eval()
    for m in model.modules():
        if isinstance(m, TrOCRSinusoidalPositionalEmbedding):
            m.weights = m.get_embedding(m.weights.size(0), m.embedding_dim, m.padding_idx).to(DEVICE, torch.bfloat16)
    model.generation_config.use_cache = True
    model.generation_config.num_beams = 1

    pv = processor(images, return_tensors="pt").pixel_values.to(DEVICE, torch.bfloat16)
    pv = pv.contiguous(memory_format=torch.channels_last)

    # warmup
    with torch.inference_mode():
        model.generate(pv, max_new_tokens=64)
    torch.cuda.synchronize(DEVICE)

    print(f"profiling 1 generate() call at bs={BS}, max_new_tokens=64 ...\n")
    import pathlib
    trace_dir = pathlib.Path(__file__).parent / "results" / "trace"
    trace_dir.mkdir(exist_ok=True, parents=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as prof:
        with torch.inference_mode():
            model.generate(pv, max_new_tokens=64)
        torch.cuda.synchronize(DEVICE)
        prof.step()
    print(f"tensorboard trace: {trace_dir} — view with `tensorboard --logdir={trace_dir}` then open http://localhost:6006")

    events = prof.key_averages()
    tc_matmul_us = 0.0
    tc_attn_us = 0.0
    cuda_us = 0.0
    buckets: dict[str, dict[str, float]] = {"tc_matmul": {}, "tc_attn": {}, "cuda": {}}
    for ev in events:
        dt = float(getattr(ev, "device_time_total", 0.0) or getattr(ev, "self_cuda_time_total", 0.0))
        if dt <= 0:
            continue
        name = ev.key
        if TC_ATTN_RE.search(name):
            tc_attn_us += dt
            buckets["tc_attn"][name] = buckets["tc_attn"].get(name, 0.0) + dt
        elif TC_MATMUL_RE.search(name):
            tc_matmul_us += dt
            buckets["tc_matmul"][name] = buckets["tc_matmul"].get(name, 0.0) + dt
        elif "kernel" in name.lower() or "void " in name:
            cuda_us += dt
            buckets["cuda"][name] = buckets["cuda"].get(name, 0.0) + dt

    total = tc_matmul_us + tc_attn_us + cuda_us
    if total == 0:
        print("no GPU kernel events captured?")
        return
    print(f"total device kernel time : {total/1000:8.2f} ms")
    print(f"  TC matmul (cutlass/wmma): {tc_matmul_us/1000:8.2f} ms  ({tc_matmul_us/total*100:5.1f} %)")
    print(f"  TC attention (flash/fmha): {tc_attn_us/1000:8.2f} ms  ({tc_attn_us/total*100:5.1f} %)")
    print(f"  CUDA core (norm/elem/cat): {cuda_us/1000:8.2f} ms  ({cuda_us/total*100:5.1f} %)")
    tc_total = tc_matmul_us + tc_attn_us
    print(f"  → Tensor Core total      : {tc_total/1000:8.2f} ms  ({tc_total/total*100:5.1f} %)\n")

    for bucket_name, bucket in buckets.items():
        print(f"top 5 in {bucket_name}:")
        for name, t in sorted(bucket.items(), key=lambda x: -x[1])[:5]:
            print(f"  {t/1000:7.2f} ms  {name[:110]}")
        print()

    # ---- plot ----
    import matplotlib.pyplot as plt
    out_dir = __import__("pathlib").Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True, parents=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"Tensor Core utilization — TrOCR generate (bs={BS}, max_new_tokens=64)\n"
        f"GPU: {torch.cuda.get_device_name(DEVICE)} (sm_{torch.cuda.get_device_capability(DEVICE)[0]}{torch.cuda.get_device_capability(DEVICE)[1]})",
        fontsize=10,
    )

    sizes = [tc_matmul_us, tc_attn_us, cuda_us]
    labels = [
        f"TC matmul\n(cutlass / wmma)\n{tc_matmul_us/1000:.1f} ms",
        f"TC attention\n(flash / fmha)\n{tc_attn_us/1000:.1f} ms",
        f"CUDA core\n(LayerNorm, GeLU,\nKV-cache cat, etc.)\n{cuda_us/1000:.1f} ms",
    ]
    colors = ["#36B37E", "#4C9AFF", "#888888"]
    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.45, edgecolor="white"), textprops={"fontsize": 9})
    ax1.set_title(f"Kernel time breakdown\n(TC total: {(tc_total)/total*100:.0f}%)")

    # Bar chart of top 6 kernels overall
    all_kernels: dict[str, tuple[float, str]] = {}
    for bn, b in buckets.items():
        for k, t in b.items():
            all_kernels[k] = (t, bn)
    top = sorted(all_kernels.items(), key=lambda x: -x[1][0])[:8]
    bar_colors = {"tc_matmul": "#36B37E", "tc_attn": "#4C9AFF", "cuda": "#888888"}
    ms = [t / 1000 for _, (t, _) in top]
    short_names = []
    for k, (_, _) in top:
        s = k
        # Trim verbose cutlass names
        for prefix in ["void cutlass::Kernel2<", "void pytorch_flash::", "void at::native::", "void "]:
            if s.startswith(prefix):
                s = s[len(prefix):]
        s = s.split("(")[0][:55]
        short_names.append(s)
    cs = [bar_colors[bn] for _, (_, bn) in top]
    bars = ax2.barh(range(len(top)), ms, color=cs)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(short_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("device time (ms)")
    ax2.set_title("Top kernels by device time")
    for b, m in zip(bars, ms):
        ax2.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {m:.1f}",
                 va="center", fontsize=8)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=bar_colors[k]) for k in ["tc_matmul", "tc_attn", "cuda"]]
    ax2.legend(legend_handles, ["TC matmul", "TC attention", "CUDA core"], loc="lower right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / "tensor_cores.png"
    fig.savefig(out_path, dpi=140)
    print(f"plot saved: {out_path}")


if __name__ == "__main__":
    main()
