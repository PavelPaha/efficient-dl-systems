import os
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from weight_loading import CPULoader
import tests_utils

@pytest.fixture
def prepare_fake_model_layers(tmp_path):
    return tests_utils.prepare_fake_model_layers(tmp_path, 80)


def test_prefetch_loader_simple(prepare_fake_model_layers):
    paths = [str(p) for p in prepare_fake_model_layers.iterdir()]
    loader = CPULoader(
        paths_to_layers=paths,
        memory_limit_mb=80
    )

    for i in range(len(paths)):
        loader.load_next()
        loader.release()



def test_loader_with_threads(prepare_fake_model_layers):
    paths = [str(p) for p in prepare_fake_model_layers.iterdir()]
    loader = CPULoader(paths_to_layers=paths, memory_limit_mb=240)
    
    print('start')

    def loader_job():
        # time.sleep(0.2)
        for i in range(len(paths)):
            loader.load_next()
            # if (i+1) % 3 == 0:
                # time.sleep(0.5)
            print(f'loaded {len(loader.loaded_layers)}')

    def releaser_job():
        for i in range(len(paths)):
            loader.release()
            if (i+1) % 3 == 0:
                time.sleep(0.5)
            print(f'loaded {len(loader.loaded_layers)}')

    t1 = threading.Thread(target=loader_job)
    t2 = threading.Thread(target=releaser_job)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert loader.current_memory_mb <= loader.memory_limit_mb
