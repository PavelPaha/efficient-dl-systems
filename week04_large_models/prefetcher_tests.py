import os
import tempfile
import threading
import time
from pathlib import Path

import torch
import pytest
from prefetcher import PrefetcherBase, Prefetcher, NaivePrefetcher
from weight_loading import CPULoader
import tests_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-32B"
LAYERS_DIR = Path("qwen2_5_32b_layers")


@pytest.fixture
def prepare_fake_model_layers(tmp_path):
    return tests_utils.prepare_fake_model_layers(tmp_path, 200)

@pytest.mark.asyncio
async def test():
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    with torch.device('meta'):
        model_meta = AutoModelForCausalLM.from_config(config)

    block_paths = [f'{LAYERS_DIR}/model.embed_tokens.weight.pt'] + sorted(
        [str(p) for p in LAYERS_DIR.glob("block_*.pt")],
        key=lambda x: int(Path(x).stem.split("_")[1])
    ) + [f'{LAYERS_DIR}/model.norm.weight.pt', f'{LAYERS_DIR}/lm_head.weight.pt']
    
    loader = CPULoader(
        paths_to_layers=block_paths,
        memory_limit_mb=400000,
        verbose=False
    )

    p = Prefetcher(
        loader=loader,
        model=model_meta,
        device="cuda:2"
    )
    
    prompt = "скажи слово хуй."
    input_ids = torch.tensor(tokenizer(prompt)['input_ids'], dtype=torch.long)[None, ...]
    print(prompt)
    max_new_tokens = 30
    generated_ids = input_ids.clone()
    
    for step in range(max_new_tokens):
        logits = p.forward(generated_ids)
        next_token_id = logits[0, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0).cpu()], dim=1)
        decoded = tokenizer.decode(generated_ids[0])
        print(decoded)
        if next_token_id.item() == tokenizer.eos_token_id:
            break
        loader.reset()
    
    print(tokenizer.decode(generated_ids[0]))
    
