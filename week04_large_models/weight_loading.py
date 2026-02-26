import abc
import copy
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import torch


class BlockLoader(abc.ABC):
    @abc.abstractmethod
    def load_next(self):
        ...

    @abc.abstractmethod
    def release(self):
        ...

    @abc.abstractmethod
    def reset(self):
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    def start(self):
        self.reset()
        n = len(self)
        self._thread = threading.Thread(
            target=lambda: [self.load_next() for _ in range(n)],
            daemon=True,
        )
        self._thread.start()

    def join(self):
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join()
            self._thread = None


class GPULoader(BlockLoader):
    def __init__(self, cpu_loader: 'CPULoader', blocks_metadata: list,
                 device: str, gpu_memory_limit_mb: int, verbose: bool = False):
        self.cpu_loader = cpu_loader
        self.blocks_metadata = blocks_metadata
        self.device = device
        self.gpu_memory_limit_mb = gpu_memory_limit_mb
        self.verbose = verbose
        self._copy_stream = torch.cuda.Stream(device=device)
        self._thread = None
        self._cpu_thread = None
        self._init_state()

    def _init_state(self):
        self.left_index = 0
        self.right_index = 0
        self.loaded_layers = {}
        self.current_memory_mb = 0.0
        self.release_event = threading.Event()
        self.load_event = threading.Event()

    def __len__(self):
        return len(self.blocks_metadata)

    def start(self):
        self.reset()
        cpu = self.cpu_loader
        cpu.reset()
        cpu.prefetch(from_idx=0)
        n_cpu = len(cpu)

        self._cpu_thread = threading.Thread(
            target=lambda: [cpu.load_next() for _ in range(n_cpu)],
            daemon=True,
        )
        self._cpu_thread.start()

        n = len(self)
        self._thread = threading.Thread(
            target=lambda: [self.load_next() for _ in range(n)],
            daemon=True,
        )
        self._thread.start()

    def join(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._cpu_thread is not None:
            self._cpu_thread.join()
            self._cpu_thread = None

    def _block_size_mb(self, cpu_sd: dict) -> float:
        return sum(v.numel() * v.element_size() for v in cpu_sd.values()) / (1024 * 1024)

    def load_next(self):
        if self.right_index >= len(self):
            return

        cpu_sd = self.cpu_loader.release()
        if cpu_sd is None:
            return

        size_mb = self._block_size_mb(cpu_sd)

        while self.current_memory_mb + size_mb > self.gpu_memory_limit_mb:
            self.release_event.clear()
            self.release_event.wait()

        block_template, prefix = self.blocks_metadata[self.right_index]
        block = copy.deepcopy(block_template).to_empty(device=self.device)

        local_sd = block.state_dict()
        with torch.cuda.stream(self._copy_stream):
            for k, v in cpu_sd.items():
                if not k.startswith(prefix):
                    continue
                local_key = k[len(prefix):]
                local_sd[local_key].copy_(v, non_blocking=True)

        copy_done = torch.cuda.Event()
        copy_done.record(self._copy_stream)

        self.loaded_layers[self.right_index] = (block, copy_done, size_mb)
        self.current_memory_mb += size_mb
        if self.verbose:
            print(f'GPULoader _load {self.right_index}, '
                  f'gpu_mb: {self.current_memory_mb - size_mb:.0f} -> {self.current_memory_mb:.0f}, '
                  f'limit={self.gpu_memory_limit_mb}')
        self.right_index += 1
        self.load_event.set()

    def release(self):
        if self.left_index >= len(self):
            return None

        idx = self.left_index

        if idx == self.right_index:
            self.load_event.clear()
            self.load_event.wait()

        block, copy_done, size_mb = self.loaded_layers.pop(idx)

        torch.cuda.current_stream(self.device).wait_event(copy_done)

        self.current_memory_mb -= size_mb
        self.left_index += 1
        if self.verbose:
            print(f'GPULoader _release {idx}, gpu_mb: {self.current_memory_mb:.0f}')

        self.release_event.set()
        return block

    def reset(self):
        for block, _, _ in self.loaded_layers.values():
            del block
        self._init_state()


class CPULoader:
    def __init__(self, paths_to_layers, memory_limit_mb=80, verbose=False,
                 num_io_workers: int = 4, dtype=torch.bfloat16):
        self.paths_to_layers = paths_to_layers
        self.memory_limit_mb = memory_limit_mb
        self.verbose = verbose
        self._executor = ThreadPoolExecutor(max_workers=num_io_workers)
        self.dtype = dtype
        self._init()

    def _init(self):
        self.left_index = 0
        self.right_index = 0
        self.loaded_layers = {}
        self.layer_sizes_mb = {}
        self.current_memory_mb = 0.0
        self._reserved_mb = 0.0
        self.release_event = threading.Event()
        self.load_event = threading.Event()
        self._prefetch_futures: dict[int, Future] = {}
        self._prefetch_sizes: dict[int, float] = {}

    def __len__(self):
        return len(self.paths_to_layers)

    @staticmethod
    def _read(path: str, dtype: torch.dtype):
        loaded_dict = torch.load(path, map_location='cpu')
        for k, v in loaded_dict.items():
            loaded_dict[k] = loaded_dict[k].to(dtype)
        return loaded_dict

    def _get_layer_size_mb(self, path):
        size_bytes = os.path.getsize(path)
        return size_bytes // (1024 * 1024)

    def prefetch(self, from_idx: int):
        for idx in range(from_idx, len(self)):
            if idx in self._prefetch_futures:
                continue
            size_mb = self._get_layer_size_mb(self.paths_to_layers[idx])
            if self.current_memory_mb + self._reserved_mb + size_mb > self.memory_limit_mb:
                break
            self._reserved_mb += size_mb
            self._prefetch_sizes[idx] = size_mb
            self._prefetch_futures[idx] = self._executor.submit(
                self._read, self.paths_to_layers[idx], self.dtype
            )

    def _load_layer(self):
        path = self.paths_to_layers[self.right_index]
        size_mb = self._get_layer_size_mb(path)

        if self.right_index in self._prefetch_futures:
            layer = self._prefetch_futures.pop(self.right_index).result()
            reserved = self._prefetch_sizes.pop(self.right_index, 0.0)
            self._reserved_mb -= reserved
        else:
            layer = self._read(path, self.dtype)

        self.loaded_layers[self.right_index] = layer
        self.layer_sizes_mb[self.right_index] = size_mb
        if self.verbose:
            print(f'_load_layer {self.right_index}, '
                  f'current_memory_mb: {self.current_memory_mb} -> '
                  f'{self.current_memory_mb + size_mb}, {self.memory_limit_mb=}')
        self.current_memory_mb += size_mb
        self.right_index += 1

        self.prefetch(self.right_index)

    def _release_layer(self):
        idx = self.left_index

        if idx == self.right_index:
            self.load_event.clear()
            self.load_event.wait()

        if idx in self.loaded_layers:
            size_mb = self.layer_sizes_mb[idx]
            layer = self.loaded_layers.pop(idx)
            del self.layer_sizes_mb[idx]
            self.current_memory_mb -= size_mb
            self.left_index += 1
            if self.verbose:
                print(f'_release_layer {idx=}: {self.current_memory_mb=}')
            return layer

        raise KeyError(f"layer {idx} not found in loaded_layers")

    def release(self):
        if self.left_index >= len(self):
            return
        layer = self._release_layer()
        self.release_event.set()
        next_prefetch_idx = max(self.right_index, max(self._prefetch_futures.keys(), default=0) + 1)
        self.prefetch(next_prefetch_idx)
        return layer

    def load_next(self):
        if self.right_index >= len(self):
            return
        path = self.paths_to_layers[self.right_index]
        size_mb = self._get_layer_size_mb(path)

        while self.current_memory_mb + size_mb > self.memory_limit_mb:
            self.release_event.clear()
            self.release_event.wait()

        self._load_layer()
        self.load_event.set()

    def reset(self):
        for fut in self._prefetch_futures.values():
            fut.cancel()
        self._init()


class PreloadedGPULoader(BlockLoader):
    def __init__(self, cpu_state_dicts: list, blocks_metadata: list,
                 device: str, gpu_memory_limit_mb: int):
        self.cpu_state_dicts = cpu_state_dicts
        self.blocks_metadata = blocks_metadata
        self.device = device
        self.gpu_memory_limit_mb = gpu_memory_limit_mb
        self._copy_stream = torch.cuda.Stream(device=device)
        self._thread = None
        self._init_state()

    def _init_state(self):
        self.left_index = 0
        self.right_index = 0
        self.loaded_layers = {}
        self.current_memory_mb = 0.0
        self.release_event = threading.Event()
        self.load_event = threading.Event()

    def __len__(self):
        return len(self.blocks_metadata)

    def load_next(self):
        if self.right_index >= len(self):
            return

        cpu_sd = self.cpu_state_dicts[self.right_index]
        size_mb = sum(v.numel() * v.element_size() for v in cpu_sd.values()) / (1024 * 1024)

        while self.current_memory_mb + size_mb > self.gpu_memory_limit_mb:
            self.release_event.clear()
            self.release_event.wait()

        block_template, prefix = self.blocks_metadata[self.right_index]
        block = copy.deepcopy(block_template).to_empty(device=self.device)

        local_sd = block.state_dict()
        with torch.cuda.stream(self._copy_stream):
            for k, v in cpu_sd.items():
                if not k.startswith(prefix):
                    continue
                local_key = k[len(prefix):]
                local_sd[local_key].copy_(v, non_blocking=True)

        copy_done = torch.cuda.Event()
        copy_done.record(self._copy_stream)

        self.loaded_layers[self.right_index] = (block, copy_done, size_mb)
        self.current_memory_mb += size_mb
        self.right_index += 1
        self.load_event.set()

    def release(self):
        if self.left_index >= len(self):
            return None
        idx = self.left_index
        if idx == self.right_index:
            self.load_event.clear()
            self.load_event.wait()

        block, copy_done, size_mb = self.loaded_layers.pop(idx)
        torch.cuda.current_stream(self.device).wait_event(copy_done)
        self.current_memory_mb -= size_mb
        self.left_index += 1
        self.release_event.set()
        return block

    def reset(self):
        for block, _, _ in self.loaded_layers.values():
            del block
        self._init_state()
