"""TrOCR optimized — fp16 + batching + SDPA/flash-attn-2 (mirrors insanely-fast-whisper tactics)."""
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import is_flash_attn_2_available

DEVICE = "cuda:2"
MODEL_ID = "microsoft/trocr-base-handwritten"
IMG = Path(__file__).parent / "a01-122-02.jpg"
BATCH_SIZES = [1, 8, 24]
DTYPE = torch.float16


def main():
    # TrOCR decoder lacks SDPA support, but the ViT encoder has it. Mix per-submodule.
    enc_attn = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    dec_attn = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    print(f"attn impl: encoder={enc_attn}, decoder={dec_attn}, dtype: {DTYPE}")

    t0 = time.perf_counter()
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        attn_implementation={"encoder": enc_attn, "decoder": dec_attn},
    ).to(DEVICE).eval()
    t_load = time.perf_counter() - t0

    image = Image.open(IMG).convert("RGB")

    pv_single = processor(image, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)

    # warmup
    with torch.inference_mode():
        _ = model.generate(pv_single, max_new_tokens=64)
    torch.cuda.synchronize(DEVICE)

    print(f"\n=== TrOCR fast ({MODEL_ID}) on {DEVICE} ===")
    print(f"load:        {t_load:.2f}s")

    # single-image timing for direct comparison with baseline
    t1 = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(pv_single, max_new_tokens=64)
    torch.cuda.synchronize(DEVICE)
    t_single = time.perf_counter() - t1
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"generate(1): {t_single*1000:.1f} ms  -> {text!r}")

    # batched throughput
    for bs in BATCH_SIZES:
        batch = processor([image] * bs, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)
        torch.cuda.synchronize(DEVICE)
        t2 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(batch, max_new_tokens=64)
        torch.cuda.synchronize(DEVICE)
        dt = time.perf_counter() - t2
        per = dt / bs * 1000
        print(f"batch={bs:>2}: total {dt*1000:7.1f} ms  | per-image {per:6.1f} ms  | {bs/dt:6.1f} img/s")

    mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    print(f"peak VRAM:   {mem:.1f} MiB")


if __name__ == "__main__":
    main()
