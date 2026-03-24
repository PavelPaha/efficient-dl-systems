import torch
from typing import List, Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from dataclasses import dataclass

from edlang.entrypoints.config import EngineConfig
from torch.nn.utils.rnn import pad_sequence

@dataclass
class Request:
    request_id: int
    prompt: str
    max_new_tokens: int
    current_len: int = 0
    sampling_params: Optional[Dict[str, Any]] = None  # Bonus Part

    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    generated_tokens: Optional[List[int]] = None
    generated_text: Optional[str] = None
    num_generated: int = 0
    is_finished: bool = False


@dataclass
class BatchResult:
    request_ids: List[int]
    new_tokens: List[List[int]]
    finished: List[bool]


class InferenceEngine:
    def __init__(self, engine_config: EngineConfig):
        self.model_config = engine_config.model_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=self.model_config.torch_dtype,
            device_map=self.model_config.device,
        )
        self.model.eval()

    @torch.no_grad()
    def prefill(self, requests: List[Request]) -> BatchResult:
        """
        Prefill phase: tokenize prompts, run through model, generate first token.
        
        Steps:
        1. Tokenize prompts and create batch
        2. Forward pass with use_cache=True to get logits and KV cache
        3. Generate first token for each request (greedy: argmax)
        4. Save request state (input_ids, attention_mask, past_key_values)
        5. Check if finished (EOS token or max_new_tokens reached)
        
        Note: Use attention_mask to get real prompt length (without padding).
        """
        if not requests:
            return BatchResult(request_ids=[], new_tokens=[], finished=[])


        batch = self.tokenizer(
            [r.prompt for r in requests],
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        
        for i, r in enumerate(requests):
            r.input_ids = input_ids[i]
            r.attention_mask = attention_mask[i]
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )

        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        generated_tokens = []
        
        for i, (ii, am, l, pkv, r) in enumerate(zip(input_ids, attention_mask, logits, past_key_values, requests)):
            length = am.sum().item()
            r.attention_mask = am
            r.input_ids = ii[:length]
            # print(pkv)
            r.past_key_values = self._get_past_for_request(
                past_key_values,
                request_idx=i,
                real_seq_len=length
            )
            # r.num_generated += 1
            next_token_idx = int(l[length-1].argmax(dim=-1))
            r.generated_tokens = [next_token_idx]
            generated_tokens.append(next_token_idx)
            
            
        return BatchResult(request_ids=[r.request_id for r in requests], new_tokens=generated_tokens, finished=[False for i in range(len(requests))])
        
        
        
        # TODO: Tokenize prompts and create batch (use self.tokenizer with padding=True)
        # TODO: Forward pass through model
        # TODO: For each request:
        #   - Get real prompt length from attention_mask
        #   - Generate next token (greedy: argmax from logits[i, real_prompt_len - 1, :])
        #   - Get past_key_values for the request with self._get_past_for_request
        #   - Save state: current_len, input_ids (real part only), attention_mask, past_key_values
        #   - Set generated_tokens, num_generated, is_finished
        raise NotImplementedError("TODO: Implement prefill method")

    @torch.no_grad()
    def decode(self, requests: List[Request]) -> BatchResult:
        """
        Decode phase: generate next token for each active request using KV cache.
        
        Steps:
        1. Filter active (non-finished) requests
        2. Prepare batched KV cache with RIGHT padding
        3. Create batch from last generated tokens
        4. Build attention_mask accounting for different sequence lengths
        5. Forward pass with past_key_values and cache_position
        6. Generate next token (greedy: argmax)
        7. Update request state
        
        Note: Use RIGHT padding for KV cache. Handle finished requests separately.
        """
        # TODO: Filter active requests (if none, return empty results for all)
        # TODO: Prepare batched KV cache using _prepare_past_key_values_batch
        # TODO: Create batch from last generated tokens [batch_size, 1]
        active = [r for r in requests if not r.is_finished]
        finished = [r for r in requests if r.is_finished]
        results_tokens = []
        results_finished = []
        results_ids = []

        for r in finished:
            results_ids.append(r.request_id)
            results_tokens.append([])
            results_finished.append(True)
        
        if len(active) > 0:
            cache = self._prepare_past_key_values_batch(active)
            batch = torch.tensor([[r.generated_tokens[-1]] for r in active], device=self.model.device)
            lengths = [r.past_key_values.key_cache[0].shape[2] + 1 for r in active]

            attention_mask = torch.zeros((len(active), max(lengths)), device=self.model.device, dtype=torch.long)
            for i, l in enumerate(lengths):
                attention_mask[i, :l] = 1

            outputs = self.model(input_ids=batch, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
            
            logits = outputs.logits[:, -1, :]
            next_tokens = logits.argmax(dim=-1)
                
            for idx, r in enumerate(active):
                if r.is_finished:
                    results_tokens.append([])
                    results_finished.append(True)
                    continue

                token = int(next_tokens[idx])
                r.generated_tokens.append(token)
                r.past_key_values = self._get_past_for_request(outputs.past_key_values, idx)
                r.num_generated += 1

                if r.num_generated >= r.max_new_tokens or token == self.tokenizer.eos_token_id:
                    r.is_finished = True

                results_tokens.append([token])
                results_finished.append(r.is_finished)
                results_ids.append(r.request_id)

        return BatchResult(
            results_ids,
            results_tokens,
            results_finished
        )
        
        # TODO: Build attention_mask for each active request
        # TODO: Forward pass with past_key_values
        # TODO: Get next tokens (greedy: argmax from last logit)
        # TODO: Update each request state (generated_tokens, num_generated, past_key_values, etc.)
        
        
        

    def _get_past_for_request(
        self,
        past_key_values,
        request_idx: int,
        real_seq_len: Optional[int] = None,
    ):
        if past_key_values is None:
            return None

        new_cache = DynamicCache()
        for layer_idx in range(self.model.config.num_hidden_layers):
            key   = past_key_values.key_cache[layer_idx][request_idx:request_idx+1]
            value = past_key_values.value_cache[layer_idx][request_idx:request_idx+1]

            if real_seq_len is not None and key.shape[2] > real_seq_len:
                key   = key[:, :, :real_seq_len, :]
                value = value[:, :, :real_seq_len, :]

            new_cache.update(key, value, layer_idx)
        return new_cache

    def _prepare_past_key_values_batch(self, requests: List[Request]):
        """
        Prepare batched KV cache from requests with RIGHT padding.
        
        Combines KV cache from different requests into one batch. Since requests
        may have different sequence lengths, add RIGHT padding to max_seq_len.
        """
        if not requests:
            return None
        
        active = [r for r in requests if r.past_key_values is not None]
        num_layers = self.model.config.num_hidden_layers
        batch_size = len(active)
        
        cache = DynamicCache()
        for layer in range(num_layers):
            layer_keys = []
            layer_values = []

            max_len = max(
                r.past_key_values.key_cache[layer].shape[2]
                for r in active
            )

            for r in active:
                k = r.past_key_values.key_cache[layer]
                v = r.past_key_values.value_cache[layer]

                pad_len = max_len - k.shape[2]
                if pad_len > 0:
                    k = torch.nn.functional.pad(k, (0,0,0,pad_len))
                    v = torch.nn.functional.pad(v, (0,0,0,pad_len))

                layer_keys.append(k)
                layer_values.append(v)

            cache.update(
                torch.cat(layer_keys, dim=0),
                torch.cat(layer_values, dim=0),
                layer
            )
        
        return cache
        # TODO: Create new DynamicCache for batch
        raise NotImplementedError("TODO: Implement _prepare_past_key_values_batch method")

    def _sample(self, tokens_dist: torch.Tensor, request: Request) -> int:
        # BOUNS PART - Implement sampling logic with sampling_params
        logits = tokens_dist.clone().float()
        params = request.sampling_params

        if params is None or not params.get("do_sample", False):
            return int(logits.argmax(dim=-1))

        eos_token_id = params.get("eos_token_id", None)
        if params.get("ignore_eos_token", False) and eos_token_id is not None:
            logits[eos_token_id] = float("-inf")

        temp = params.get("temperature", 1.0)
        logits = logits / temp

        top_k = params.get("top_k", None)
        if top_k is not None and top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k)
            logits[logits < top_k_values[-1]] = float("-inf")

        top_p = params.get("top_p", None)
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            m = (cumsum - sorted_probs) > top_p
            sorted_logits[m] = float("-inf")
            mask = m.scatter(0, sorted_indices, m)
            logits[mask] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return int(token.item())

    def get_generated_text(self, request: Request) -> str:
        if not request.generated_tokens:
            return request.prompt
        
        print(f'{request.input_ids.tolist()=},\n{request.generated_tokens=}')

        full_ids = request.input_ids.tolist() + request.generated_tokens
        return self.tokenizer.decode(full_ids, skip_special_tokens=True)