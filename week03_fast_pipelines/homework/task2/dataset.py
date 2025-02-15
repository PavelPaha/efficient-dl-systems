from typing import Optional
from enum import Enum
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader
import transformer


MAX_LENGTH = 640

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_wikitext_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.data = load_wikitext_data(data_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        encoded = tokenizer(
            self.data[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # print(encoded)
        return encoded["input_ids"].squeeze(0), transformer.generate_square_subsequent_mask(self.max_length)
        #
        # transformer.generate_square_subsequent_mask(self.max_length)


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __getitem__(self, idx: int):
        pass


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        pass

    def __getitem__(self, idx: int):
        pass


class UltraDuperBigBrainDataset(IterableDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __iter__(self):
        pass


def collate_fn(
    batch: list[tuple[str, torch.Tensor]], max_length: Optional[int] = MAX_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    pass


class UltraBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass
