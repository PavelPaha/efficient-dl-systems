import copy
import threading
import time

import torch

from weight_loading import BlockLoader, CPULoader, GPULoader


class BlockForwarder:
    def __init__(self, model, device="cuda:0"):
        self.model = model
        self.device = device
        self.n_layers = len(model.model.layers)

    def forward_block(self, layer_idx, inputs, position_embeddings=None,
                      attention_mask=None, block=None):
        if layer_idx == -1:
            return self.model.model.embed_tokens(inputs)

        if 0 <= layer_idx < self.n_layers:
            layer = block if block is not None else self.model.model.layers[layer_idx]
            outputs = layer(
                inputs,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return outputs[0] if not isinstance(outputs, torch.Tensor) else outputs

        if layer_idx == self.n_layers:
            return self.model.model.norm(inputs)

        if layer_idx == self.n_layers + 1:
            return self.model.lm_head(inputs)

        raise ValueError(f"Invalid layer_idx: {layer_idx}")

    def build_rotary(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        return self.model.model.rotary_emb(hidden_states, position_ids)


def unbind_parameters(module):
    for name, param in list(module._parameters.items()):
        if param is None:
            continue
        del module._parameters[name]
        setattr(module, name, param.view_as(param))
    for child in module.children():
        unbind_parameters(child)


def create_causal_mask(seq_len, device, dtype):
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]






class PrefetcherBase:
    def __init__(self, loader: CPULoader, device='cuda:0'):
        self.loader = loader
        self.device = device

    async def forward(self, inputs):
        raise NotImplementedError()

    def reset(self):
        self.loader.reset()


class NaivePrefetcher(PrefetcherBase):
    def __init__(self, loader: CPULoader, device='cuda:0'):
        super().__init__(loader, device)

    async def forward(self, inputs):
        out = inputs.to(self.device)
        n = len(self.loader)
        self.loader.load_next()
        for i in range(-1, n + 2):
            load_thread = threading.Thread(target=self.loader.load_next)
            load_thread.start()
            layer = self.loader.release()
            layer = layer['weight'].to(self.device)
            time.sleep(0.3)
            load_thread.join()



class Prefetcher:
    def __init__(self, model=None, device: str = "cuda:0",
                 loader=None,
                 
                 gpu_memory_limit_mb: int = 4096 * 4,
                 verbose: bool = False):
        if model is None:
            raise ValueError("model is required")
        if loader is None:
            raise ValueError("loader is required")

        if isinstance(loader, CPULoader):
            n = len(model.model.layers)
            meta = (
                [(model.model.embed_tokens, "model.embed_tokens.")]
                + [(model.model.layers[i], f"model.layers.{i}.") for i in range(n)]
                + [(model.model.norm, "model.norm.")]
                + [(model.lm_head, "lm_head.")]
            )
            loader = GPULoader(
                cpu_loader=loader, blocks_metadata=meta,
                device=device, gpu_memory_limit_mb=gpu_memory_limit_mb,
                verbose=verbose,
            )

        self.model = model
        self.device = device
        self.loader = loader
        self.forwarder = BlockForwarder(model, device)
        self._rotary_ready = False
        self._trainable_embed = False

    def _materialize_rotary(self):
        if self._rotary_ready:
            return
        rotary = self.model.model.rotary_emb
        rotary.to_empty(device=self.device)
        inv_freq, attention_scaling = rotary.compute_default_rope_parameters(
            rotary.config, device=self.device,
        )
        rotary.inv_freq.copy_(inv_freq)
        rotary.original_inv_freq.copy_(inv_freq)
        rotary.attention_scaling = attention_scaling
        self._rotary_ready = True

    def enable_embedding_training(self, embed_state_dict: dict,
                                  prefix: str = "model.embed_tokens."):
        embed = copy.deepcopy(self.model.model.embed_tokens).to_empty(device=self.device)
        local_sd = embed.state_dict()
        for k, v in embed_state_dict.items():
            if k.startswith(prefix):
                local_sd[k[len(prefix):]].copy_(v)
        embed.weight.requires_grad_(True)
        self.model.model.embed_tokens = embed
        self._trainable_embed = True

    def trainable_parameters(self):
        if self._trainable_embed:
            return [self.model.model.embed_tokens.weight]
        return []

    @staticmethod
    def _freeze_and_unbind(module):
        for p in module.parameters():
            p.requires_grad = False
        unbind_parameters(module)

    def forward(self, input_ids: torch.Tensor, training: bool = False) -> torch.Tensor:
        loader = self.loader
        loader.start()
        n_layers = self.forwarder.n_layers

        ctx = (torch.autograd.graph.save_on_cpu(pin_memory=True)
               if training else torch.inference_mode())

        with ctx:
            inputs = input_ids.to(self.device)

            if self._trainable_embed:
                out = self.forwarder.forward_block(-1, inputs)
            else:
                embed = loader.release()
                if training:
                    self._freeze_and_unbind(embed)
                self.model.model.embed_tokens = embed
                out = self.forwarder.forward_block(-1, inputs)
                if training:
                    out.requires_grad_(True)
                self.model.model.embed_tokens = self.model.model.embed_tokens.to("meta")
                del embed

            self._materialize_rotary()
            cos, sin = self.forwarder.build_rotary(out)

            for i in range(n_layers):
                block = loader.release()
                if training:
                    self._freeze_and_unbind(block)
                out = self.forwarder.forward_block(
                    i, out,
                    position_embeddings=(cos, sin),
                    attention_mask=None,
                    block=block,
                )
                del block

            loader.join()

            norm = loader.release()
            if training:
                self._freeze_and_unbind(norm)
            self.model.model.norm = norm
            out = self.forwarder.forward_block(n_layers, out)
            self.model.model.norm = self.model.model.norm.to("meta")
            del norm

            lm_head = loader.release()
            if training:
                self._freeze_and_unbind(lm_head)
            self.model.lm_head = lm_head
            logits = self.forwarder.forward_block(n_layers + 1, out)
            self.model.lm_head = self.model.lm_head.to("meta")
            del lm_head

        return logits
