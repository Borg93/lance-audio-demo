"""TrOCR baseline — single image, fp32, default attention. Mirrors the HF snippet."""
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = "cuda:2"
MODEL_ID = "microsoft/trocr-base-handwritten"
IMG = Path(__file__).parent / "a01-122-02.jpg"


def main():
    t0 = time.perf_counter()
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    t_load = time.perf_counter() - t0

    image = Image.open(IMG).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    # warmup
    with torch.inference_mode():
        _ = model.generate(pixel_values, max_new_tokens=64)
    torch.cuda.synchronize(DEVICE)

    t1 = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
    torch.cuda.synchronize(DEVICE)
    t_gen = time.perf_counter() - t1

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2

    print(f"=== TrOCR baseline ({MODEL_ID}) on {DEVICE} ===")
    print(f"load:        {t_load:.2f}s")
    print(f"generate(1): {t_gen*1000:.1f} ms")
    print(f"peak VRAM:   {mem:.1f} MiB")
    print(f"output:      {text!r}")


if __name__ == "__main__":
    main()
