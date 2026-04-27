"""TrOCR + torch.compile — eats kernel-launch overhead in the autoregressive loop."""
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

    # Only compile the decoder — the autoregressive loop is the hot path.
    # Compiling the encoder + reduce-overhead causes CUDA graph aliasing because
    # encoder_outputs are held across decode steps. Encoder runs once/batch anyway.
    model.decoder.forward = torch.compile(
        model.decoder.forward, mode="reduce-overhead", dynamic=True, fullgraph=False
    )

    image = Image.open(IMG).convert("RGB")
    pv_single = processor(image, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)

    # Warmup — first generate triggers compile, second confirms cached graph.
    print("warmup (compile)...")
    tw = time.perf_counter()
    with torch.inference_mode():
        _ = model.generate(pv_single, max_new_tokens=64)
    torch.cuda.synchronize(DEVICE)
    print(f"  first call (compile included): {time.perf_counter()-tw:.2f}s")

    tw = time.perf_counter()
    with torch.inference_mode():
        _ = model.generate(pv_single, max_new_tokens=64)
    torch.cuda.synchronize(DEVICE)
    print(f"  second call (warm): {time.perf_counter()-tw:.2f}s")

    print(f"\n=== TrOCR fast+compile ({MODEL_ID}) on {DEVICE} ===")
    print(f"load:        {t_load:.2f}s")

    # bs=1 timing
    t1 = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(pv_single, max_new_tokens=64)
    torch.cuda.synchronize(DEVICE)
    t_single = time.perf_counter() - t1
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"generate(1): {t_single*1000:.1f} ms  -> {text!r}")

    for bs in BATCH_SIZES:
        if bs == 1:
            continue
        batch = processor([image] * bs, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)
        # warmup this batch shape (compile recompiles on shape change)
        with torch.inference_mode():
            _ = model.generate(batch, max_new_tokens=64)
        torch.cuda.synchronize(DEVICE)

        t2 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(batch, max_new_tokens=64)
        torch.cuda.synchronize(DEVICE)
        dt = time.perf_counter() - t2
        print(f"batch={bs:>2}: total {dt*1000:7.1f} ms  | per-image {dt/bs*1000:6.1f} ms  | {bs/dt:6.1f} img/s")

    mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    print(f"peak VRAM:   {mem:.1f} MiB")


if __name__ == "__main__":
    main()
