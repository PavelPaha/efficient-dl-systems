from typing import Optional, Dict
from enum import Enum
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader
import transformer
from collections import Counter


MAX_LENGTH = 640
PAD = 0
UNK = 1

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def yield_tokens(data_iter):
    for text, label in data_iter:
        yield tokenizer(text)


def load_wikitext_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def get_vocab(dataset, tokenizer) -> Dict:
    counter = Counter()
    for line in dataset:
        counter.update(tokenizer(line)["input_ids"])

    vocab = {token: idx + 2 for idx, (token, _) in enumerate(counter.most_common())}
    # 2, because PAD = 0, UNK = 1
    return vocab


wiki_lines = load_wikitext_data('wikitext-103-raw-v1/test-00000-of-00001.txt')
vocab = get_vocab(wiki_lines, tokenizer)


def get_tgt(encoded_input, pad_token_id):
    tgt = torch.roll(encoded_input, -1)
    tgt[-1] = pad_token_id
    tgt = tgt.to(torch.int)
    return tgt


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
        src = encoded["input_ids"].squeeze(0).to(torch.int)
        tgt = torch.roll(src, -1)
        tgt[-1] = tokenizer.pad_token_id
        tgt = tgt.to(torch.int)
        # raise Exception((src, tgt))
        return src, get_tgt(src, tokenizer.pad_token_id)


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.data = load_wikitext_data(data_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        src = tokenizer(
            self.data[idx],
            return_tensors="pt"
        )["input_ids"].squeeze(0).to(torch.int)
        return src, get_tgt(src, tokenizer.pad_token_id)


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = -1):
        self.data = load_wikitext_data(data_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        src = tokenizer(
            self.data[idx],
            return_tensors="pt"
        )["input_ids"].squeeze(0).to(torch.int)
        return src, get_tgt(src, tokenizer.pad_token_id)


class UltraDuperBigBrainDataset(IterableDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __iter__(self):
        pass


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
    src_list = []
    tgt_list = []
    for src, tgt in batch:
        processed_src = [vocab.get(token, UNK) for token in src]
        src_list.append(torch.tensor(processed_src, dtype=torch.int64))
        processed_tgt = [vocab.get(token, UNK) for token in tgt]
        tgt_list.append(torch.tensor(processed_tgt, dtype=torch.int64))
    src_list = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD)
    tgt_list = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD)
    return tgt_list, src_list


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, data: torch.Tensor, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        self.batch_size = batch_size
        self.length = len(data)
        self.len_to_idx = {}
        for i, (src, tgt) in enumerate(data):
            length = src.shape[0]
            if length not in self.len_to_idx:
                self.len_to_idx[length] = set()
            self.len_to_idx[length].add(i)
        self.random_len = next(iter(self.len_to_idx.keys()))

    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        while len(self.len_to_idx.keys()):
            if self.random_len not in self.len_to_idx:
                self.random_len = next(iter(self.len_to_idx.keys()))
            indices = self.len_to_idx[self.random_len]
            idx = next(iter(indices))
            indices.remove(idx)
            if len(indices) == 0:
                del self.len_to_idx[self.random_len]
            yield idx
