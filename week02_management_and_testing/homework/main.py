import torch
import hydra
import io
import wandb
import tempfile
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel



def log_artifact(cfg: DictConfig):
    config_yaml = OmegaConf.to_yaml(cfg)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as temp_file:
        temp_file.write(config_yaml)
        temp_file_path = temp_file.name
    artifact = wandb.Artifact(name="hydra_config", type="config")
    artifact.add_file(temp_file_path, name="hydra_config.yaml")
    wandb.log_artifact(artifact)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    wandb.init(project=cfg.wandb.project_name)
    log_artifact(cfg)

    ddpm = instantiate(cfg.train.model)
    device = cfg.train.train_params.device
    ddpm.to(device)
    print(cfg.dataset.output_folder)
    dataset = CIFAR10(
        cfg.dataset.output_folder,
        train=True,
        download=False,
        transform=instantiate(cfg.train.transform)
    )

    dataloader = DataLoader(dataset, batch_size=cfg.train.train_params.batch_size, num_workers=4, shuffle=True)

    optimizer_cfg = cfg.train.optimizer
    optimizer_params = OmegaConf.to_container(optimizer_cfg, resolve=True)
    optimizer_params["params"] = ddpm.parameters()
    optim = instantiate(optimizer_params)
    epochs = cfg.train.train_params.epochs
    for i in range(epochs):
        train_epoch(ddpm, dataloader, optim, device)
        grid = generate_samples(ddpm, device)
        wandb.log(
            {
                'grid_on_epoch': wandb.Image(grid)
            }
        )

    wandb.finish()

if __name__ == "__main__":
    main()
