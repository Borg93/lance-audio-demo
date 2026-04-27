"""Batch-size sweep for TrOCR using the SDPA-decoder monkey-patch + Tensor Core knobs.

Pushes batch size up to find the sweet spot for the RTX PRO 6000 Blackwell (96 GB).
Captures real GPU utilization, memory, and power via `nvidia-smi` polled in a
background thread — so we can prove we're actually saturating the device.

Usage:
    uv run --no-sync python trocr-test/sweep.py --n 256 --bs 8,16,32,64,128
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Pin physical GPU 2 → cuda:0 inside torch (avoids edge cases with this checkpoint).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
PHYSICAL_GPU = 2

import torch
import torch.nn.functional as F
from datasets import load_dataset
from jiwer import cer, wer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.trocr.modeling_trocr import (
    TrOCRAttention,
    TrOCRSinusoidalPositionalEmbedding,
)

# ---------------------------------------------------------------------------
# Tensor Core / kernel-tuning knobs (applied globally before any model loads)
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision("high")     # use TF32 for fp32 matmul
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True          # auto-pick fastest conv kernels

DEVICE_ID = 0
DEVICE = torch.device(f"cuda:{DEVICE_ID}")
MODEL_ID = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
DATASET = "Riksarkivet/eval_htr_out_of_domain_lines"
OUT_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# SDPA monkey-patch for TrOCRAttention.forward (verified accuracy-neutral).
# ---------------------------------------------------------------------------


def _trocr_attention_sdpa_forward(
    self, hidden_states, key_value_states=None, past_key_values=None,
    attention_mask=None, output_attentions=False, **kwargs,
):
    is_cross = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

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

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
        scale=self.scaling,
    )
    out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
    return self.out_proj(out), None


TrOCRAttention.forward = _trocr_attention_sdpa_forward


# ---------------------------------------------------------------------------
# Background nvidia-smi poller
# ---------------------------------------------------------------------------


class GPUMonitor:
    """Polls `nvidia-smi` in a daemon thread; exposes mean/peak util + memory + power."""

    def __init__(self, gpu_index: int, interval: float = 0.1):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cmd = [
            "nvidia-smi", "-i", str(self.gpu_index),
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, timeout=2).decode().strip()
                util, mem, pw = (float(x) for x in out.split(", "))
                self.samples.append((util, mem, pw))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def stats(self) -> dict[str, float]:
        if not self.samples:
            return dict(util_mean=0.0, util_p99=0.0, mem_peak_mib=0.0, power_mean_w=0.0, n_samples=0)
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        powers = [s[2] for s in self.samples]
        utils_sorted = sorted(utils)
        return dict(
            util_mean=sum(utils) / len(utils),
            util_p99=utils_sorted[max(0, int(len(utils_sorted) * 0.99) - 1)],
            mem_peak_mib=max(mems),
            power_mean_w=sum(powers) / len(powers),
            n_samples=len(utils),
        )


# ---------------------------------------------------------------------------
# Result + helpers
# ---------------------------------------------------------------------------


@dataclass
class Result:
    batch_size: int
    n: int
    total_s: float
    gpu_only_s: float
    preproc_s: float
    per_image_ms: float
    img_per_s: float
    cer_pct: float
    wer_pct: float
    vram_peak_mib: float       # from torch
    smi_mem_peak_mib: float    # from nvidia-smi
    smi_util_mean: float
    smi_util_p99: float
    smi_power_mean_w: float


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


def build_model(dtype: torch.dtype, *, use_cache: bool, num_beams: int | None,
                cache_static: bool):
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation={"encoder": "sdpa", "decoder": "eager"},
    ).to(DEVICE).eval()
    # Materialize the legacy sinusoidal weights attribute on the right device.
    for module in model.modules():
        if isinstance(module, TrOCRSinusoidalPositionalEmbedding):
            module.weights = module.get_embedding(
                module.weights.size(0), module.embedding_dim, module.padding_idx
            ).to(DEVICE, dtype)

    # CRITICAL: the upstream model card ships generation_config with use_cache=False.
    # That alone makes generate ~Nx slower and silently disables `cache_implementation`.
    # Flip it on; KV cache is just memoization, mathematically identical output.
    model.generation_config.use_cache = use_cache
    if num_beams is not None:
        model.generation_config.num_beams = num_beams
    if cache_static:
        model.generation_config.cache_implementation = "static"
    return processor, model


# ---------------------------------------------------------------------------
# One run at a given batch size
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_batch_size(
    bs: int,
    samples: list[tuple],
    *,
    dtype: torch.dtype,
    max_new_tokens: int,
    use_channels_last: bool,
    monitor: GPUMonitor,
    use_cache: bool,
    num_beams: int | None,
    cache_static: bool,
    compile_decoder: bool,
) -> Result:
    print(f"\n=== bs={bs} ===")
    free_cuda()

    processor, model = build_model(
        dtype, use_cache=use_cache, num_beams=num_beams, cache_static=cache_static
    )
    if compile_decoder:
        # mode="default": inductor kernel fusion + autotuning. No cudagraph capture,
        # so StaticCache's in-place `keys.index_copy_` and `cumulative_length.add_`
        # don't get flagged as cross-run aliasing. Smaller per-step gain than
        # reduce-overhead would give (no graph replay), but it actually runs.
        # dynamic=True lets one compiled artifact handle prefill (seq>1) and
        # decode (seq=1) without recompiling per shape.
        model.decoder.forward = torch.compile(
            model.decoder.forward,
            mode="default",
            dynamic=True,
            fullgraph=False,
        )
    images = [s[0] for s in samples]
    refs = [s[1] for s in samples]

    # Warmup at this batch size — first call hits cuDNN benchmark + cache allocator.
    pv = processor(images[: min(bs, len(images))], return_tensors="pt").pixel_values.to(DEVICE, dtype)
    if use_channels_last:
        pv = pv.contiguous(memory_format=torch.channels_last)
    if compile_decoder:
        torch.compiler.cudagraph_mark_step_begin()
    model.generate(pv, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize(DEVICE)

    # Pre-process all batches up-front so the timed loop measures GPU work only.
    preproc_t0 = time.perf_counter()
    pv_batches: list[torch.Tensor] = []
    for i in range(0, len(images), bs):
        chunk = images[i : i + bs]
        pv = processor(chunk, return_tensors="pt").pixel_values.to(DEVICE, dtype, non_blocking=True)
        if use_channels_last:
            pv = pv.contiguous(memory_format=torch.channels_last)
        pv_batches.append(pv)
    torch.cuda.synchronize(DEVICE)
    preproc_s = time.perf_counter() - preproc_t0

    preds: list[str] = []
    monitor.start()
    t0 = time.perf_counter()
    for pv in pv_batches:
        if compile_decoder:
            # Tell cudagraph trees that a fresh generate() call boundary starts here,
            # so the static-cache in-place KV updates aren't flagged as cross-run aliasing.
            torch.compiler.cudagraph_mark_step_begin()
        ids = model.generate(pv, max_new_tokens=max_new_tokens)
        preds.extend(processor.batch_decode(ids, skip_special_tokens=True))
    torch.cuda.synchronize(DEVICE)
    gpu_only_s = time.perf_counter() - t0
    monitor.stop()

    smi = monitor.stats()
    res = Result(
        batch_size=bs,
        n=len(images),
        total_s=preproc_s + gpu_only_s,
        gpu_only_s=gpu_only_s,
        preproc_s=preproc_s,
        per_image_ms=gpu_only_s / len(images) * 1000,
        img_per_s=len(images) / gpu_only_s,
        cer_pct=cer(refs, preds) * 100,
        wer_pct=wer(refs, preds) * 100,
        vram_peak_mib=torch.cuda.max_memory_allocated(DEVICE) / 1024**2,
        smi_mem_peak_mib=smi["mem_peak_mib"],
        smi_util_mean=smi["util_mean"],
        smi_util_p99=smi["util_p99"],
        smi_power_mean_w=smi["power_mean_w"],
    )

    print(
        f"  N={res.n}\n"
        f"  preproc        : {res.preproc_s:6.2f} s\n"
        f"  gpu-only       : {res.gpu_only_s:6.2f} s\n"
        f"  per-image      : {res.per_image_ms:6.1f} ms  ({res.img_per_s:5.1f} img/s)\n"
        f"  CER            : {res.cer_pct:5.2f} %\n"
        f"  WER            : {res.wer_pct:5.2f} %\n"
        f"  torch peak VRAM: {res.vram_peak_mib:7.0f} MiB\n"
        f"  smi  peak VRAM : {res.smi_mem_peak_mib:7.0f} MiB\n"
        f"  smi  util  mean: {res.smi_util_mean:5.1f} %  (p99 {res.smi_util_p99:5.1f} %)\n"
        f"  smi  power mean: {res.smi_power_mean_w:5.1f} W"
    )
    del model, processor, pv_batches
    free_cuda()
    return res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sweep(results: list[Result], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True, parents=True)
    bs = [r.batch_size for r in results]
    ips = [r.img_per_s for r in results]
    per = [r.per_image_ms for r in results]
    util = [r.smi_util_mean for r in results]
    util_p99 = [r.smi_util_p99 for r in results]
    vram = [r.vram_peak_mib for r in results]
    power = [r.smi_power_mean_w for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"TrOCR batch-size sweep — {MODEL_ID}\n"
        f"GPU: {torch.cuda.get_device_name(DEVICE)}, n={results[0].n}, "
        f"fp16 + SDPA-decoder + TF32 + cuDNN benchmark + channels-last",
        fontsize=10,
    )

    ax = axes[0, 0]
    ax.plot(bs, ips, "o-", color="#36B37E")
    ax.set_xlabel("batch size"); ax.set_ylabel("images / second"); ax.set_title("Throughput")
    ax.set_xscale("log", base=2); ax.grid(alpha=0.3)
    for x, y in zip(bs, ips):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax = axes[0, 1]
    ax.plot(bs, per, "o-", color="#4C9AFF")
    ax.set_xlabel("batch size"); ax.set_ylabel("ms / image"); ax.set_title("Per-image latency")
    ax.set_xscale("log", base=2); ax.grid(alpha=0.3)
    for x, y in zip(bs, per):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax = axes[1, 0]
    ax.plot(bs, util, "o-", label="mean", color="#FF8B00")
    ax.plot(bs, util_p99, "s--", label="p99", color="#9F7AEA")
    ax.set_xlabel("batch size"); ax.set_ylabel("GPU utilization %"); ax.set_title("GPU util (nvidia-smi)")
    ax.set_ylim(0, 105); ax.set_xscale("log", base=2); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1, 1]
    ax.plot(bs, vram, "o-", label="VRAM peak (MiB)", color="#E94B6E")
    ax.set_xlabel("batch size"); ax.set_ylabel("VRAM (MiB)"); ax.set_title("Memory + power")
    ax.set_xscale("log", base=2); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(bs, power, "s--", color="#888888", label="power (W)")
    ax2.set_ylabel("power (W)")
    ax.legend(loc="upper left"); ax2.legend(loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out_dir / "sweep.png"
    fig.savefig(p, dpi=140)
    print(f"\nplot saved: {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256, help="number of eval samples")
    p.add_argument("--bs", type=str, default="1,8,16,32,64,128",
                   help="comma-separated batch sizes to sweep")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-channels-last", action="store_true")
    p.add_argument("--smi-interval", type=float, default=0.1)
    p.add_argument("--num-beams", type=int, default=4,
                   help="generation beam width (model default is 4)")
    p.add_argument("--no-use-cache", action="store_true",
                   help="disable KV cache (matches model's broken default)")
    p.add_argument("--cache-static", action="store_true",
                   help="use StaticCache (precondition for compile to pay off)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the decoder")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                   help="floating-point dtype. bf16 is ~14%% faster on Blackwell with same accuracy.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bs_list = [int(x) for x in args.bs.split(",") if x.strip()]

    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}  (physical {PHYSICAL_GPU} → cuda:{DEVICE_ID})")
    print(f"Tensor Core knobs: TF32 on, cuDNN benchmark on, channels-last={'off' if args.no_channels_last else 'on'}")
    print(f"Loading {args.n} samples from {DATASET} ...")
    samples = load_samples(args.n)
    print(f"got {len(samples)} samples\n")

    print(f"generation: num_beams={args.num_beams}, use_cache={not args.no_use_cache}, "
          f"cache_static={args.cache_static}, compile={args.compile}\n")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"dtype: {args.dtype} ({dtype})\n")

    monitor = GPUMonitor(PHYSICAL_GPU, interval=args.smi_interval)
    results: list[Result] = []
    for bs in bs_list:
        try:
            res = run_batch_size(
                bs, samples,
                dtype=dtype,
                max_new_tokens=args.max_new_tokens,
                use_channels_last=not args.no_channels_last,
                monitor=monitor,
                use_cache=not args.no_use_cache,
                num_beams=args.num_beams,
                cache_static=args.cache_static,
                compile_decoder=args.compile,
            )
            results.append(res)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at bs={bs} — stopping sweep ({e})")
            break

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    (OUT_DIR / "sweep.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\njson saved: {OUT_DIR / 'sweep.json'}")

    print("\n\n=== summary ===")
    header = (f"{'bs':>4} {'gpu-s':>7} {'ms/img':>7} {'img/s':>7} "
              f"{'CER%':>6} {'VRAM':>7} {'util':>6} {'p99':>5} {'pwr':>5}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.batch_size:>4} {r.gpu_only_s:>7.2f} {r.per_image_ms:>7.1f} {r.img_per_s:>7.1f} "
              f"{r.cer_pct:>6.2f} {r.vram_peak_mib:>7.0f} "
              f"{r.smi_util_mean:>6.1f} {r.smi_util_p99:>5.1f} {r.smi_power_mean_w:>5.0f}")

    plot_sweep(results, OUT_DIR)


if __name__ == "__main__":
    main()
