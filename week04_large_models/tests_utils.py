import os
import torch


def prepare_fake_model_layers(tmp_path, size_mb):
    folder = tmp_path / "fake_model"
    if os.path.isdir(folder):
        return folder

    folder.mkdir()
    size = size_mb * 1024 * 1024 // 4
    for i in range(20):
        tensor = torch.randn(size, dtype=torch.float32)
        torch.save({'weight': tensor}, folder / f"layer_{i}.pt")
    return folder