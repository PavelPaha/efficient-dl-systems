import transformer
from dataset import BrainDataset, DataMode
from torch.utils.data import DataLoader
import torch


def get_gpt2_model() -> torch.nn.Module:
    ntokens = 50257
    d_model = 768
    nhead = 12
    d_hid = 3072
    nlayers = 12
    dropout = 0.1
    model = transformer.TransformerModel(ntokens, d_model, nhead, d_hid, nlayers, dropout)
    return model

data_path = 'wikitext-103-raw-v1/test-00000-of-00001.txt'


def run_epoch(data_mode: DataMode) -> None:
    model = get_gpt2_model()
    dataloader = None
    dataset = BrainDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in dataset:
        input = data[0]
        print('SHAPE=', data[1][..., None].shape)
        # att_mask = data[1][..., None].repeat(2, input.shape[-1])
        output = model(input, data[1])
