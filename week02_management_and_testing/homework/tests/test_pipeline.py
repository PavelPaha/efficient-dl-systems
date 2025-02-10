import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step
from modeling.unet import UnetModel
from modeling.training import train_epoch, generate_samples
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb
import hydra
from hydra.utils import instantiate


@pytest.fixture
def cfg():
    with hydra.initialize(config_path="../conf", version_base=None):
        cfg = hydra.compose(config_name="config")

    # print(OmegaConf.to_yaml(cfg))
    return cfg


@pytest.fixture
def train_dataset(cfg):
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        cfg.dataset.output_folder,
        train=True,
        download=False,
        transform=instantiate(cfg.train.transform)
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cuda:5"]])
def test_train_on_one_batch(device, train_dataset, cfg):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = instantiate(cfg.train.model)
    ddpm.to(device)

    optim = torch.optim.AdamW(ddpm.parameters(), lr=2e-4)
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(["device"], [["cuda:5"]])
def test_training(device, train_dataset):
    # note: implement and test a complete training procedure (including sampling)
    with hydra.initialize(config_path="../conf", version_base=None):
        cfg = hydra.compose(config_name="config")

    print(OmegaConf.to_yaml(cfg))

    wandb.init(project=cfg.wandb.project_name)
    ddpm = instantiate(cfg.train.model)
    ddpm.to(device)

    optimizer_cfg = cfg.train.optimizer
    optimizer_params = OmegaConf.to_container(optimizer_cfg, resolve=True)
    optimizer_params["params"] = ddpm.parameters()
    optim = instantiate(optimizer_params)

    dataloader = DataLoader(train_dataset, batch_size=cfg.train.train_params.batch_size, num_workers=4, shuffle=True)

    epochs =  cfg.train.train_params.epochs

    ddpm.train()
    for epoch in range(epochs):
        loss = train_epoch(ddpm, dataloader, optim, device)
        wandb.log(
            {
                'loss_on_epoch': loss,
                'epoch_number': epoch
            }
        )

        grid_name = f"samples/{epoch:02d}.png"
        grid = generate_samples(ddpm, device)
        wandb.log(
            {
                'grid_on_epoch': wandb.Image(grid)
            }
        )
    wandb.finish()
