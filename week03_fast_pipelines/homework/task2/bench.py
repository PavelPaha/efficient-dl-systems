from run_epoch import run_epoch, get_gpt2_model
from dataset import DataMode
import torch
import torchvision.models as models
from torch.profiler import ProfilerActivity, profile, record_function

if __name__ == '__main__':
    model = get_gpt2_model()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            run_epoch(DataMode.ULTRA_BIG_BRAIN)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
