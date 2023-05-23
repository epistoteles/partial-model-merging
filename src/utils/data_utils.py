import git
import os

import torch
import torchvision
import torchvision.transforms as T
from safetensors.torch import save_file, load_file

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze


##############################
# directory helper functions #
##############################


def _get_root_dir() -> str:
    """
    Returns the root directory of the git repository
    :return: the root directory path
    """
    return git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")


def _get_data_dir() -> str:
    """
    Returns the data directory of the git repository (and creates it if it doesn't exist)
    :return: the data directory path
    """
    root_dir = _get_root_dir()
    data_dir = os.path.join(root_dir, "data/")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _get_checkpoints_dir() -> str:
    """
    Returns the checkpoints directory of the git repository (and creates it if it doesn't exist)
    :return: the checkpoints directory path
    """
    root_dir = _get_root_dir()
    checkpoints_dir = os.path.join(root_dir, "checkpoints/")
    os.makedirs(checkpoints_dir, exist_ok=True)
    return checkpoints_dir


############################
# model saving and loading #
############################


def save_model(model, filename: str):
    """
    Saves a PyTorch model state dict as .pt file
    :param model: the model whose state dict we want to save
    :param filename: the name of the output .safetensors file (optionally including path)
    :return: None
    """
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"
    checkpoints_dir = _get_checkpoints_dir()
    if checkpoints_dir not in filename:
        filename = os.path.join(checkpoints_dir, filename)
    save_file(model.state_dict(), filename)


def load_model(model: torch.nn.Module, filename: str) -> torch.nn.Module:
    """
    Loads a PyTorch model state dict from a .safetensors file
    :param model: the model to apply the state dict to
    :param filename: the name of the state dict .safetensors file (optionally including path)
    :return: None
    """
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"
    checkpoints_dir = _get_checkpoints_dir()
    if checkpoints_dir not in filename:
        filename = os.path.join(checkpoints_dir, filename)
    state_dict = load_file(filename)
    model.load_state_dict(state_dict)
    return model


############################
# datasets and dataloaders #
############################


def _convert_dataset_to_beton(dataset: torch.utils.data.Dataset, filename: str):
    """
    Converts a torchvision dataset into a ffcv-compatible .beton dataset
    :param dataset: a RGB torchvision dataset
    :param filename: the name of the output .beton file (optionally including path)
    :return: None
    """
    if not filename.endswith(".beton"):
        filename += ".beton"
    writer = DatasetWriter(filename, {"image": RGBImageField(), "label": IntField()})
    writer.from_indexed_dataset(dataset)


def _download_CIFAR10() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Returns (and downloads, if necessary) the CIFAR10 train and test dataset
    :return:
    """
    data_dir = _get_data_dir()
    train_dset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    test_dset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    return train_dset, test_dset


def _get_CIFAR10_beton() -> tuple[str, str]:
    """
    Returns the .beton filepaths, downloads and converts the dataset first if they don't already exist.
    :return: (train .beton filepath, test .beton filepath)
    """
    data_dir = _get_data_dir()
    train_beton_path = os.path.join(data_dir, "cifar_train.beton")
    test_beton_path = os.path.join(data_dir, "cifar_test.beton")
    if not (os.path.exists(train_beton_path) and os.path.exists(test_beton_path)):
        print("CIFAR10 dataset not present - downloading and/or converting ...")
        train_dset, test_dset = _download_CIFAR10()
        _convert_dataset_to_beton(train_dset, train_beton_path)
        _convert_dataset_to_beton(test_dset, test_beton_path)
    return train_beton_path, test_beton_path


def get_loaders_CIFAR10(loader: str = None) -> tuple[Loader, Loader, Loader]:
    """
    Creates and returns three FFCV CIFAR10 loaders. Downloads and converts CIFAR10 if necessary.
    :param loader: only returns the specified loader if set (options: 'train_aug', 'train_noaug', 'test')  TODO
    :return: (train_aug_loader, train_noaug_loader, test_loader)
    """
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    device = torch.device("cuda:0")
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
    pre_p = [SimpleRGBImageDecoder()]
    post_p = [
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
    aug_p = [
        RandomHorizontalFlip(),
        RandomTranslate(padding=4),
    ]

    train_beton_path, test_beton_path = _get_CIFAR10_beton()

    train_aug_loader = Loader(
        train_beton_path,
        batch_size=500,
        num_workers=min(8, os.cpu_count()),
        order=OrderOption.RANDOM,
        drop_last=True,
        pipelines={"image": pre_p + aug_p + post_p, "label": label_pipeline},
    )
    train_noaug_loader = Loader(
        train_beton_path,
        batch_size=1000,
        num_workers=min(8, os.cpu_count()),
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines={"image": pre_p + post_p, "label": label_pipeline},
    )
    test_loader = Loader(
        test_beton_path,
        batch_size=1000,
        num_workers=min(8, os.cpu_count()),
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines={"image": pre_p + post_p, "label": label_pipeline},
    )

    return train_aug_loader, train_noaug_loader, test_loader
