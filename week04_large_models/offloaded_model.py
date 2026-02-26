from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from prefetcher import Prefetcher
from weight_loading import CPULoader, GPULoader, PreloadedGPULoader


class OffloadedModel:
    def __init__(
        self,
        model_name: str,
        layers_dir: str | Path,
        device: str = "cuda:0",
        gpu_memory_limit_mb: int = 4096,
        cpu_memory_limit_mb: int = 400_000,
        preload_to_cpu: bool = True,
        num_io_workers: int = 8,
    ):
        self.device = device
        self.gpu_memory_limit_mb = gpu_memory_limit_mb
        self.cpu_memory_limit_mb = cpu_memory_limit_mb
        self.num_io_workers = num_io_workers
        self.layers_dir = Path(layers_dir)

        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n_layers = self.config.num_hidden_layers

        self._block_paths = self._discover_block_paths()

        with torch.device("meta"):
            self._model = AutoModelForCausalLM.from_config(self.config)

        self._blocks_metadata = self._build_blocks_metadata()

        if preload_to_cpu:
            self._cpu_state_dicts = self._preload_cpu(num_io_workers)
        else:
            self._cpu_state_dicts = None

        loader = self._build_loader()
        self._prefetcher = Prefetcher(
            model=self._model,
            device=self.device,
            loader=loader,
        )

    def _build_blocks_metadata(self):
        return (
            [(self._model.model.embed_tokens, "model.embed_tokens.")]
            + [(self._model.model.layers[i], f"model.layers.{i}.")
               for i in range(self.n_layers)]
            + [(self._model.model.norm, "model.norm.")]
            + [(self._model.lm_head, "lm_head.")]
        )

    def _discover_block_paths(self) -> List[str]:
        d = self.layers_dir
        return (
            [str(d / "model.embed_tokens.weight.pt")]
            + sorted(
                [str(p) for p in d.glob("block_*.pt")],
                key=lambda x: int(Path(x).stem.split("_")[1]),
            )
            + [str(d / "model.norm.weight.pt"),
               str(d / "lm_head.weight.pt")]
        )

    def _preload_cpu(self, num_workers: int) -> list:
        def load_one(path):
            sd = torch.load(path, map_location="cpu")
            return {k: v.to(torch.bfloat16) for k, v in sd.items()}

        result = []
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for sd in tqdm(pool.map(load_one, self._block_paths),
                           total=len(self._block_paths), desc="Loading to CPU"):
                result.append(sd)
        return result

    def _build_loader(self):
        if self._cpu_state_dicts is not None:
            return PreloadedGPULoader(
                cpu_state_dicts=self._cpu_state_dicts,
                blocks_metadata=self._blocks_metadata,
                device=self.device,
                gpu_memory_limit_mb=self.gpu_memory_limit_mb,
            )
        cpu_loader = CPULoader(
            paths_to_layers=self._block_paths,
            memory_limit_mb=self.cpu_memory_limit_mb,
            num_io_workers=self.num_io_workers,
        )
        return GPULoader(
            cpu_loader=cpu_loader,
            blocks_metadata=self._blocks_metadata,
            device=self.device,
            gpu_memory_limit_mb=self.gpu_memory_limit_mb,
        )

    def enable_embedding_training(self):
        if self._cpu_state_dicts is not None:
            embed_sd = self._cpu_state_dicts[0]
            self._cpu_state_dicts = self._cpu_state_dicts[1:]
        else:
            embed_sd = torch.load(self._block_paths[0], map_location="cpu")
            embed_sd = {k: v.to(torch.bfloat16) for k, v in embed_sd.items()}

        self._block_paths = self._block_paths[1:]
        self._blocks_metadata = self._blocks_metadata[1:]
        self._prefetcher.loader = self._build_loader()
        self._prefetcher.enable_embedding_training(embed_sd)

    def trainable_parameters(self):
        return self._prefetcher.trainable_parameters()

    def forward(self, input_ids: torch.Tensor, training: bool = False) -> torch.Tensor:
        return self._prefetcher.forward(input_ids, training=training)

    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        input_ids = torch.tensor(
            self.tokenizer(prompt)["input_ids"], dtype=torch.long,
        )[None, ...]
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat(
                [generated, next_id.unsqueeze(0).cpu()], dim=1,
            )
            if next_id.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(input_ids, **kwargs)
