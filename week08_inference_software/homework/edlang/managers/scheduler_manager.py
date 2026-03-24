from typing import List, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time

import sys
import os

from edlang.entrypoints.engine import Request, InferenceEngine, BatchResult
from edlang.managers.metric_manager import MetricManager


class PrefillPolicy(str, Enum):
    SERIAL = "serial"
    BATCH_WHEN_IDLE = "batch_when_idle"
    INTERLEAVE = "interleave"


@dataclass
class SchedulerConfig:
    max_batch_size: int = 8
    max_waiting_requests: int = 100
    prefill_timeout_ms: float = 50.0
    enable_metrics: bool = False
    prefill_policy: PrefillPolicy = PrefillPolicy.SERIAL


class EDLangScheduler:

    def __init__(
        self,
        engine: InferenceEngine,
        config: Optional[SchedulerConfig] = None,
    ):
        self.engine = engine
        self.config = config or SchedulerConfig()

        self.waiting_queue = deque()
        self.active_requests = []

        self.next_request_id = 0
        self.metrics_manager = MetricManager(enable_metrics=self.config.enable_metrics)

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ):
        request = Request(
            request_id=self.next_request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        self.waiting_queue.append(request)
        self.next_request_id += 1

        self.metrics_manager.update_waiting_queue_num(len(self.waiting_queue))
        
        return request.request_id
    
    def _get_active_requests(self):
        return  [req for req in self.active_requests if not req.is_finished]
    
    def step(self):
        decode_result = None
        prefill_result = None

        # TODO: Implement step method
        # TODO: First decide how many requests to prefill
        # TODO: Then do decode
        # TODO: Update metrics and inner state
        # active_n = len(self._get_active_requests())
        prefill_result = self._prefill_step()
        active_n = len(self.active_requests)

        decode_result = self._decode_step()
        self.metrics_manager.update_waiting_queue_num(len(self.waiting_queue))

        cur_time = time.time()
        if decode_result:
            tokens_generated = sum(len(tokens) for tokens in decode_result.new_tokens)
            self.metrics_manager.calculate_throughtput_tokens_per_second(tokens_generated,  cur_time - self.metrics_manager.time)
            self.metrics_manager.time = cur_time

        # print(f'{active_n=}')

        self.metrics_manager.update_active_requests_num(active_n)
        # raise NotImplementedError("Implement step method")
    
    def _decode_step(self):        
        if not self._get_active_requests():
            return None
        
        batch_result = self.engine.decode(self.active_requests)
        assert len(self.active_requests) == len(batch_result.finished)
        return batch_result
        # TODO: Do decode for all active requests
        raise NotImplementedError("Implement decode step")
    
    def _prefill_step(self):
        if not self.waiting_queue:
            return None
         
        bs = self._decide_prefill_batch_size()
        added = 0
        while bs > 0 and len(self.waiting_queue) > 0:
            r = self.waiting_queue.popleft()
            self.active_requests.append(r)
            bs -= 1
            added += 1
        if added == 0:
            return None

        self.engine.prefill(self.active_requests)

    def _decide_prefill_batch_size(self) -> int:
        n_wait = len(self.waiting_queue)
        num_active = len(self._get_active_requests())
        if num_active > 0:
            return 0

        policy = self.config.prefill_policy
        if policy == PrefillPolicy.SERIAL:
            return 1 if n_wait > 0 else 0

        if policy == PrefillPolicy.BATCH_WHEN_IDLE:
            return min(self.config.max_batch_size, n_wait) # берм столько, сколько влазит в батч

        if policy == PrefillPolicy.INTERLEAVE:
            return min(self.config.max_batch_size, max(1, (n_wait + 1) // 2)) # берём примерно половину батча в префилл.

        return 1 if n_wait > 0 else 0
    
    def get_finished_requests(self) -> List[Request]:
        finished = [req for req in self.active_requests if req.is_finished]
        self.active_requests = self._get_active_requests()
        return finished
    
    def get_metric_manager(self):
        return self.metrics_manager

    def clear(self):
        self.waiting_queue = deque()
        self.active_requests = []
