import transformer
from dataset import BrainDataset, BigBrainDataset, UltraBigBrainDataset, DataMode, collate_fn, UltraBigBrainBatchSampler
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm

def get_gpt2_model():
    return transformer.TransformerLetsGoooooooooooo()

data_path = 'wikitext-103-raw-v1/test-00000-of-00001.txt'

def run_epoch(data_mode: DataMode) -> None:
    model = get_gpt2_model()
    dataloader = None
    batch_size = 32
    shuffle = True
    print(type(data_mode))
    if data_mode == DataMode.BRAIN:
        print('BRAIN BRO')
        dataset = BrainDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif data_mode == DataMode.BIG_BRAIN:
        print('BIG BRAIN BRUUH')
        dataset = BigBrainDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        print('ULTRA BIG BRAIN BRUUH')
        dataset = BigBrainDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=UltraBigBrainBatchSampler(dataset, batch_size))
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        print('ULTRA DUPER BIG BRAIN BRUUH')
    else:
        print(type(data_mode), type(DataMode.BRAIN))
        exit(0)

    for tgt, src in tqdm(dataloader):
        output = model(tgt, src)
