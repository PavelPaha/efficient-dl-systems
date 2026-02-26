import torch
from transformers import AutoModelForCausalLM
from pathlib import Path

MODEL = "Qwen/Qwen2.5-32B"
OUTDIR = Path("qwen2_5_32b_layers")
OUTDIR.mkdir(exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)

sd = model.state_dict()
emb_sd = {k: v.cpu() for k, v in sd.items() if k.startswith("model.embed_tokens.")}
torch.save(emb_sd, OUTDIR / "model.embed_tokens.weight.pt")

num_blocks = len(model.model.layers)
for i in range(num_blocks):
    block_prefix = f"model.layers.{i}."
    block_sd = {k: v.cpu() for k, v in sd.items() if k.startswith(block_prefix)}
    save_path = OUTDIR / f"block_{i}.pt"
    torch.save(block_sd, save_path)

norm_sd = {k: v.cpu() for k, v in sd.items() if k.startswith("model.norm.")}
torch.save(norm_sd, OUTDIR / "model.norm.weight.pt")
lm_sd = {k: v.cpu() for k, v in sd.items() if k.startswith("lm_head.")}
torch.save(lm_sd, OUTDIR / "lm_head.weight.pt")