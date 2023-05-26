import git
import os

import torch
import torchvision
import torchvision.transforms as T
from safetensors.torch import save_file, load_file

import numpy as np
import scipy
import tqdm

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

from src.models.VGG import VGG


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


###########################
# basic tensor operations #
###########################


def ensure_numpy(x):
    """
    Ensures that x is a numpy array
    :param x: numpy array or Torch tensor
    :return: numpy array
    """
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def normalize(x):
    """
    Normalizes a tensor or numpy array between 0 and 1
    :param x: the tensor or numpy array
    :return: the normalized tensor or numpy array
    """
    x -= x.min()
    x /= x.max()
    return x


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


#######################
# model manipulations #
#######################


def expand_model(model: torch.nn.Module, expansion_factor: float):
    """
    Returns a functionally equivalent but wider model. The (right-appended) weights and biases are all zero.
    :param model: the original model
    :param expansion_factor: the factor by which to expand/widen the model (must be >1)
    :return: the expanded model
    """
    assert expansion_factor > 1, "Expansion factor must be greater than 1.0"
    model_expanded = VGG(model.size, width_multiplier=model.width_multiplier * expansion_factor)
    sd_expanded = model_expanded.state_dict()
    sd = model.state_dict()
    for key in sd.keys():
        sd_expanded[key] = torch.zeros_like(sd_expanded[key])
        slice_indices = tuple(slice(0, sd[key].size(i)) for i in range(sd[key].dim()))
        sd_expanded[key][slice_indices] = sd[key]
    model_expanded.load_state_dict(sd_expanded)
    return model_expanded


def subnet(model: torch.nn.Module, n_layers: int):
    """
    Returns a subnet from layer 1 to layer n_layers (in the feature extractor)
    :param model: the original model
    :param n_layers: the first n_layers will be sliced
    :return: torch.nn.Module
    """
    return model.features[:n_layers]


################################
# correlation matrix functions #
################################


def get_corr_matrix(
    subnet_a: torch.nn.Module, subnet_b: torch.nn.Module, loader: Loader, epochs: int = 1, normalize: bool = True
):
    """
    Given two networks subnet_a, subnet_b which each output a feature map of shape NxCxWxH this will reshape
    both outputs to (N*W*H)xC and then compute a CxC correlation matrix between the outputs of the two networks
    N = dataset size, C = # of individual feature maps, H, W = height and width of one feature map
    :param subnet_a:
    :param subnet_b:
    :param loader: a data loader that resembles the input distribution (typically the train_aug_loader)
    :param epochs: for how many epochs to collect feature maps - values >1 only make a difference if the loader
                   uses augmentations
    :param normalize:
    """
    n = epochs * len(loader)
    mean_a = mean_b = std_a = std_b = None
    with torch.no_grad():
        subnet_a.eval()
        subnet_b.eval()
        for _ in range(epochs):
            for i, (images, _) in enumerate(tqdm(loader)):
                img_t = images.float().cuda()
                out_a = subnet_a(img_t)
                out_a = out_a.reshape(out_a.shape[0], out_a.shape[1], -1).permute(0, 2, 1)
                out_a = out_a.reshape(-1, out_a.shape[2]).double()

                out_b = subnet_b(img_t)
                out_b = out_b.reshape(out_b.shape[0], out_b.shape[1], -1).permute(0, 2, 1)
                out_b = out_b.reshape(-1, out_b.shape[2]).double()

                mean_a_batch = out_a.mean(dim=0)
                mean_b_batch = out_b.mean(dim=0)
                std_a_batch = out_a.std(dim=0)
                std_b_batch = out_b.std(dim=0)
                outer_batch = (out_a.T @ out_b) / out_a.shape[0]

                if i == 0:
                    mean_a = torch.zeros_like(mean_a_batch)
                    mean_b = torch.zeros_like(mean_b_batch)
                    std_a = torch.zeros_like(std_a_batch)
                    std_b = torch.zeros_like(std_b_batch)
                    outer = torch.zeros_like(outer_batch)
                mean_a += mean_a_batch / n
                mean_b += mean_b_batch / n
                std_a += std_a_batch / n
                std_b += std_b_batch / n
                outer += outer_batch / n

    cov = outer - torch.outer(mean_a, mean_b)
    if normalize:
        corr = cov / (torch.outer(std_a, std_b) + 1e-4)
        return corr
    else:
        return cov


def get_layer_perm_from_corr(corr_mtx):
    corr_mtx = ensure_numpy(corr_mtx)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx, maximize=True)
    assert (row_ind == np.arange(len(corr_mtx))).all()
    perm_map = torch.tensor(col_ind).long()
    return perm_map


def get_layer_perm(subnet_a, subnet_b, loader):
    """
    Returns the channel permutation map to make the activations of layer 1..n in subnet_a most closely
    match those in subnet_b.  TODO(Check if this actually right - only last layer?)
    :param subnet_a: The reference subnet that stays the same
    :param subnet_b: The subnet for which we want the permutation map
    :return: the permutation map
    """
    corr_mtx = get_corr_matrix(subnet_a, subnet_b, loader)
    return get_layer_perm_from_corr(corr_mtx)


# modifies the weight matrices of a convolution and batchnorm
# layer given a permutation of the output channels
def permute_output(perm_map, conv, bn):
    pre_weights = [conv.weight]
    if conv.bias is not None:
        pre_weights.append(conv.bias)
    if bn is not None:
        pre_weights.extend(
            [
                bn.weight,
                bn.bias,
                bn.running_mean,
                bn.running_var,
            ]
        )
    for w in pre_weights:
        w.data = w[perm_map]


# modifies the weight matrix of a layer for a given permutation of the input channels
# works for both conv2d and linear
def permute_input(perm_map, layer):
    w = layer.weight
    w.data = w[:, perm_map]


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
