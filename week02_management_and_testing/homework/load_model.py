import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision.datasets import CIFAR10


def load_model(cfg: DictConfig):
    dataset = CIFAR10(
        'asdfa',
        train=True,
        download=True,
        transform=train_transforms,
    )
