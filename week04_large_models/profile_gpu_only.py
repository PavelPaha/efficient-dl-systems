import argparse
from pathlib import Path

import torch
import torch.profiler

from offloaded_model import OffloadedModel

MODEL_NAME = "Qwen/Qwen2.5-32B"
LAYERS_DIR = Path("qwen2_5_32b_layers")
GPU_MEM_LIMIT_MB = 4096


def main():
    parser = argparse.ArgumentParser(description="Profile GPU-only prefetch")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="trace_gpu_only.json")
    parser.add_argument("--prompt", default="скажи слово хуй.")
    args = parser.parse_args()

    out_path = Path(args.out)

    model = OffloadedModel(
        model_name=MODEL_NAME,
        layers_dir=LAYERS_DIR,
        device=args.device,
        gpu_memory_limit_mb=GPU_MEM_LIMIT_MB,
    )

    input_ids = torch.tensor(
        model.tokenizer(args.prompt)["input_ids"], dtype=torch.long,
    )[None, ...]
    print(f"Prompt: {args.prompt!r}")

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    torch.cuda.memory._record_memory_history(max_entries=100_000)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        logits = model(input_ids)

    mem_path = out_path.with_name(out_path.stem + "_memory.pickle")
    torch.cuda.memory._dump_snapshot(str(mem_path))
    torch.cuda.memory._record_memory_history(enabled=None)
    prof.export_chrome_trace(str(out_path))
    print(f"Trace saved to {out_path}, memory snapshot to {mem_path}")


if __name__ == "__main__":
    main()
