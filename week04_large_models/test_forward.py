from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from offloaded_model import OffloadedModel

MODEL_NAME = "Qwen/Qwen2.5-32B"
LAYERS_DIR = Path("qwen2_5_32b_layers")
GPU_MEM_LIMIT_MB = 4096
DEVICE = "cuda:2"
PROMPT = "скажи слово хуй."


@pytest.fixture()
def offloaded():
    return OffloadedModel(
        model_name=MODEL_NAME,
        layers_dir=LAYERS_DIR,
        device=DEVICE,
        gpu_memory_limit_mb=GPU_MEM_LIMIT_MB,
    )


@pytest.fixture()
def input_ids(offloaded):
    return torch.tensor(
        offloaded.tokenizer(PROMPT)["input_ids"], dtype=torch.long,
    )[None, ...]


@pytest.fixture()
def blockwise_logits(offloaded, input_ids):
    logits = offloaded(input_ids)
    torch.cuda.empty_cache()
    return logits


@pytest.fixture()
def reference_logits(input_ids):
    orig = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    with torch.inference_mode():
        logits = orig(input_ids.to(DEVICE)).logits
    del orig
    torch.cuda.empty_cache()
    return logits


def test_shapes_match(blockwise_logits, reference_logits):
    assert blockwise_logits.shape == reference_logits.shape


def test_logits_match(blockwise_logits, reference_logits):
    max_diff = torch.max(torch.abs(reference_logits - blockwise_logits)).item()
    mean_diff = torch.mean(torch.abs(reference_logits - blockwise_logits)).item()
    print(f"\nmax  diff: {max_diff}")
    print(f"mean diff: {mean_diff}")
    assert max_diff < 1e-6, f"max abs diff {max_diff} exceeds 1e-6"


def test_argmax_match(blockwise_logits, reference_logits):
    assert torch.equal(
        blockwise_logits.argmax(dim=-1),
        reference_logits.argmax(dim=-1),
    )
