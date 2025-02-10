import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import wandb

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str):
    model.train()
    pbar = tqdm(dataloader)

    total_loss = 0.
    for i, (x, _) in enumerate(pbar):
        train_loss = train_step(model, x, optimizer, device)
        total_loss += train_loss

    total_loss /= len(dataloader)
    return total_loss

def generate_samples(model: DiffusionModel, device):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (3, 32, 32), device=device)
        grid = make_grid(samples, nrow=4)
        # save_image(grid, path)
    return grid
