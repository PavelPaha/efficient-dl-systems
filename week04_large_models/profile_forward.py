import argparse
import threading
from pathlib import Path

import torch
import torch.profiler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weight_loading import CPULoader, GPULoader
from prefetcher import Prefetcher


MODEL_NAME = "Qwen/Qwen2.5-32B"
LAYERS_DIR = Path("qwen2_5_32b_layers")
GPU_MEM_LIMIT_MB = 4096
CPU_MEM_LIMIT_MB = 400_000




def create_causal_mask(seq_len, device, dtype):
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


def forward_profiled(prefetcher: Prefetcher, inputs: torch.Tensor, device: str):

    n_layers = prefetcher.forwarder.n_layers
    n_files  = len(prefetcher.loader)

    gpu_loader = prefetcher._get_gpu_loader()

    def cpu_loader_job():
        for i in range(n_files):
            with torch.profiler.record_function(f"cpu_load_{i:03d}"):
                prefetcher.loader.load_next()

    def gpu_loader_job():
        for i in range(n_files):
            with torch.profiler.record_function(f"h2d_copy_{i:03d}"):
                gpu_loader.load_next()

    prefetcher.loader.prefetch(from_idx=0)

    t_cpu = threading.Thread(target=cpu_loader_job, daemon=True, name="cpu_loader")
    t_gpu = threading.Thread(target=gpu_loader_job, daemon=True, name="gpu_loader")
    t_cpu.start()
    t_gpu.start()

    with torch.inference_mode():
        inputs = inputs.to(device)

        with torch.profiler.record_function("embed"):
            embed = gpu_loader.release()
            prefetcher.model.model.embed_tokens = embed
            out = prefetcher.forwarder.forward_block(-1, inputs)
            prefetcher.model.model.embed_tokens = prefetcher.model.model.embed_tokens.to('meta')
            del embed

        mask = create_causal_mask(out.shape[1], out.device, out.dtype)

        with torch.profiler.record_function("rotary_emb"):
            prefetcher._materialize_rotary_emb()
            cos, sin = prefetcher.forwarder.build_rotary(out)

        for i in range(n_layers):
            block = gpu_loader.release()
            out = prefetcher.forwarder.forward_block(
                i, out,
                position_embeddings=(cos, sin),
                attention_mask=mask,
                block=block,
            )
            del block

        t_cpu.join()
        t_gpu.join()

        with torch.profiler.record_function("norm"):
            norm = gpu_loader.release()
            prefetcher.model.model.norm = norm
            out = prefetcher.forwarder.forward_block(n_layers, out)
            prefetcher.model.model.norm = prefetcher.model.model.norm.to('meta')
            del norm

        with torch.profiler.record_function("lm_head"):
            lm_head = gpu_loader.release()
            prefetcher.model.lm_head = lm_head
            logits = prefetcher.forwarder.forward_block(n_layers + 1, out)
            prefetcher.model.lm_head = prefetcher.model.lm_head.to('meta')
            del lm_head

    return logits



def parse_args():
    p = argparse.ArgumentParser(description="Profile a single Prefetcher forward pass")
    p.add_argument("--device",  default="cuda:2",  help="CUDA device, e.g. cuda:0")
    p.add_argument("--out",     default="trace.json", help="Output Chrome trace path")
    p.add_argument("--prompt",  default="скажи слово хуй.", help="Input prompt")
    p.add_argument("--no-cuda-activity", action="store_true",
                   help="Profile CPU only (smaller trace, faster to open in Perfetto)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config    = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with torch.device("meta"):
        model_meta = AutoModelForCausalLM.from_config(config)

    block_paths = (
        [str(LAYERS_DIR / "model.embed_tokens.weight.pt")]
        + sorted(
            [str(p) for p in LAYERS_DIR.glob("block_*.pt")],
            key=lambda x: int(Path(x).stem.split("_")[1]),
        )
        + [str(LAYERS_DIR / "model.norm.weight.pt"),
           str(LAYERS_DIR / "lm_head.weight.pt")]
    )
    print(f"Layer files: {len(block_paths)}")

    loader = CPULoader(
        paths_to_layers=block_paths,
        memory_limit_mb=CPU_MEM_LIMIT_MB,
        verbose=False,
    )
    p = Prefetcher(
        loader=loader,
        model=model_meta,
        device=args.device,
        gpu_memory_limit_mb=GPU_MEM_LIMIT_MB,
    )

    input_ids = torch.tensor(
        tokenizer(args.prompt)["input_ids"], dtype=torch.long
    )[None, ...]
    activities = [torch.profiler.ProfilerActivity.CPU]
    if not args.no_cuda_activity:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    torch.cuda.memory._record_memory_history(max_entries=100_000)

    print(f"\nRunning forward pass under profiler ...")
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        logits = forward_profiled(p, input_ids, args.device)

    mem_path = out_path.with_name(out_path.stem + "_memory.pickle")
    torch.cuda.memory._dump_snapshot(str(mem_path))
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"Memory snapshot → {mem_path}")

    prof.export_chrome_trace(str(out_path))
    print(f"\nTrace saved → {out_path}  ({out_path.stat().st_size // 1024 // 1024} MB)")


if __name__ == "__main__":
    main()
