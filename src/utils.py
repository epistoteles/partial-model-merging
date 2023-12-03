from copy import deepcopy, copy

import git
import os
from pathlib import Path
from tqdm import tqdm
import re
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
from PIL import Image
from image import DrawImage

import numpy as np
import scipy

from rich.table import Column, Table
from rich.text import Text
from rich.console import Console
import rich

import torch
import torchvision
import torchvision.transforms as T
from torch.cuda.amp import autocast
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader

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
from src.models.ResNet import ResNet18, ResNet20, BasicBlock, forward_just_residual, forward_just_downsample
from src.models.MLP import MLP


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


def get_evaluations_dir(subdir: str = None) -> str:
    """
    Returns the evaluations directory of the git repository (and creates it if it doesn't exist)
    :param subdir: if set, creates a subdir in the evaluations dir (if it doesn't exist already) and returns its path
    :return: the evaluations directory path
    """
    root_dir = _get_root_dir()
    evaluations_dir = os.path.join(root_dir, "evaluations/")
    os.makedirs(evaluations_dir, exist_ok=True)
    if subdir is not None:
        evaluations_dir = os.path.join(evaluations_dir, subdir)
        os.makedirs(evaluations_dir, exist_ok=True)
    return evaluations_dir


def get_plots_dir(subdir: str = None) -> str:
    """
    Returns the plots directory of the git repository (and creates it if it doesn't exist)
    :param subdir: if set, creates a subdir in the plots dir (if it doesn't exist already) and returns its path
    :return: the plots directory path
    """
    root_dir = _get_root_dir()
    plots_dir = os.path.join(root_dir, "plots/")
    os.makedirs(plots_dir, exist_ok=True)
    if subdir is not None:
        plots_dir = os.path.join(plots_dir, subdir)
        os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def get_all_model_names() -> list[str]:
    """
    Returns the names of all saved models
    :return: a sorted list of the checkpoint names of all saved models
    """
    checkpoints_dir = _get_checkpoints_dir()
    return sorted([Path(x).stem for x in os.listdir(checkpoints_dir) if x.endswith(".safetensors")])


def get_paired_model_names() -> list[tuple[str]]:
    """
    Returns a list of model tuples (variant 1, variant 2) of matching models [e.g. (a,b), (a, c), (b,c)].
    :return: a sorted list of tuples of the checkpoint names
    """
    all_models = get_all_model_names()
    prefix_dict = defaultdict(list)
    matching_pairs = []

    for model in all_models:
        prefix, suffix = model.rsplit("-", 1)
        prefix_dict[prefix].append(suffix)

    for prefix, suffix_list in prefix_dict.items():
        combinations = itertools.combinations(suffix_list, 2)
        pairs = [(f"{prefix}-{first}", f"{prefix}-{second}") for first, second in combinations]
        matching_pairs += pairs

    return matching_pairs


def parse_model_name(model_name, as_dict=False):
    """
    Extracts hyperparameters from the model name (or full path)
    :param model_name: the model name, e.g. "CIFAR10-VGG11-2x-a.safetensors" or "MNIST-MLP3-bn-1.5x-a"
    :param as_dict: return the values as dict if true
    :return: a hyperparameter list
    """
    if model_name.endswith(".safetensors"):
        model_name = model_name[:-12]
    exp = r"([A-Za-z0-9]+)-([A-Za-z]+)([0-9]+)-([A-Za-z]*)-?([0-9]+\.?[0-9]*)x-([a-z]+)(?:.[A-Za-z]+)?"
    dataset, model_type, size, batch_norm, width, variant = re.match(exp, model_name).groups()
    size, width = int(size), float(width)
    batch_norm = "bn" in batch_norm
    if as_dict:
        return {
            "dataset": dataset,
            "model_type": model_type,
            "size": size,
            "batch_norm": batch_norm,
            "width": width,
            "variant": variant,
        }
    else:
        return dataset, model_type, size, batch_norm, width, variant


###########################
#   overview functions    #
###########################


def model_table(dataset: str, architecture: str, bn: bool):
    """
    Prints an overview table of trained and evaluated models
    :param dataset: the dataset (that was trained on)
    :param architecture: the architecture (MLP, VGG, ResNet)
    :param bn: whether bn was used or not
    :return:
    """
    model_names = get_all_model_names()
    eval_dir = get_evaluations_dir("two_models")
    matches = lambda x: x[0] == dataset and x[1] == architecture and x[3] == bn
    models = [x for x in [parse_model_name(x) for x in model_names] if matches(x)]
    sizes = sorted(list({x[2] for x in models}))
    widths = sorted(list({x[4] for x in models}))
    table = Table(
        "width", *[f"{architecture}{size}" for size in sizes], title=f"{architecture}s trained on {dataset}, {bn=}"
    )
    for width in widths:
        variants = [
            Text(",".join([model[5] for model in models if model[4] == width and model[2] == size]), style="violet")
            for size in sizes
        ]
        table.add_row(str(width), *variants)
        eval_names = [
            os.path.join(
                eval_dir,
                f"{dataset}-{architecture}{size}-{'bn-' if bn else ''}{int(width) if width % 1 == 0 else width}x-ab.safetensors",
            )
            for size in sizes
        ]
        eval_exists = [os.path.exists(eval_name) for eval_name in eval_names]
        eval_steps = [
            get_evaluated_overlaps(name, as_string=True) if exists else ""
            for (name, exists) in zip(eval_names, eval_exists)
        ]
        table.add_row("", *[Text(x if x else "no eval", style="green" if x else "red") for x in eval_steps])
        accs = [get_metrics(name) if exists else "" for (name, exists) in zip(eval_names, eval_exists)]
        table.add_row(
            "",
            *[
                Text(
                    f"endpoints: {x['acc_endpoint_avg']:.4f} / {x['loss_endpoint_avg']:.4f}" if x else "", style="white"
                )
                for x in accs
            ],
        )
        table.add_row(
            "",
            *[
                Text(f"merging: {x['acc_merging']:.4f} / {x['loss_merging']:.4f}" if x else "", style="white")
                for x in accs
            ],
        )
        table.add_row(
            "",
            *[
                Text(
                    f"merging + REP: {x['acc_merging_REPAIR']:.4f} / {x['loss_merging_REPAIR']:.4f}" if x else "",
                    style="white",
                )
                for x in accs
            ],
        )
        table.add_section()
    console = Console()
    console.print(table)


def get_evaluated_overlaps(evaluation_filename: str, as_string: bool = False):
    """
    Given an eval .safetensors path, returns the already evaluated overlaps
    :param evaluation_filename: the safetensors path
    :return: the list of evaluated overlaps
    """
    metrics = load_file(evaluation_filename)
    keys = metrics.keys()
    result = []
    if "merging_test_accs" in keys:
        result += [1.0]
    for k in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
        if f"partial_merging_{k}_test_accs" in keys:
            result += [k]
    if as_string:
        return ", ".join([str(x) for x in result])
    else:
        return result


def get_metrics(evaluation_filename: str):
    """
    Given an eval .safetensors path, returns the already evaluated overlaps
    :param evaluation_filename: the safetensors path
    :return: the list of metrics
    """
    metrics = load_file(evaluation_filename)
    keys = metrics.keys()
    result = {}
    if "merging_test_accs" in keys:
        result["acc_endpoint_a"] = metrics["merging_test_accs"][0].item()
        result["acc_endpoint_b"] = metrics["merging_test_accs"][-1].item()
        result["acc_endpoint_avg"] = (result["acc_endpoint_a"] + result["acc_endpoint_b"]) / 2
        result["acc_merging"] = metrics["merging_test_accs"][10].item()
        result["loss_endpoint_a"] = metrics["merging_test_losses"][0].item()
        result["loss_endpoint_b"] = metrics["merging_test_losses"][-1].item()
        result["loss_endpoint_avg"] = (result["loss_endpoint_a"] + result["loss_endpoint_b"]) / 2
        result["loss_merging"] = metrics["merging_test_losses"][10].item()
    if "merging_REPAIR_test_accs" in keys:
        result["acc_merging_REPAIR"] = metrics["merging_REPAIR_test_accs"][10].item()
        result["loss_merging_REPAIR"] = metrics["merging_REPAIR_test_losses"][10].item()
    return result


###########################
# basic tensor operations #
###########################


def ensure_numpy(x):
    """
    Ensures that x is a numpy array (and converts it if necessary)
    :param x: numpy array or Torch tensor
    :return: numpy array
    """
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def normalize(x):
    """
    Returns a normalized (between 0 and 1) copy of a tensor or numpy array
    :param x: the tensor or numpy array
    :return: the normalized tensor or numpy array
    """
    x = deepcopy(x)
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


def load_model(filename: str, model: torch.nn.Module = None) -> torch.nn.Module:
    """
    Loads a PyTorch model state dict from a .safetensors file
    :param filename: the name of the state dict .safetensors file (optionally including path)
    :param model: the model to apply the state dict to; it will get created if not supplied
    :return: torch.nn.Module
    """
    if model is None:
        model = model_like(filename)
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"
    checkpoints_dir = _get_checkpoints_dir()
    if checkpoints_dir not in filename:
        filename = os.path.join(checkpoints_dir, filename)
    state_dict = load_file(filename)
    model.load_state_dict(state_dict)
    return model


def load_models_ab(filename_stump: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Loads the 'a' and 'b' variant of a PyTorch model from the .safetensors files
    :param filename_stump: the name of the state dict .safetensors file (excluding the variant, e.g. '-a')
    :return: a tuple of the models (model_a, model_b)
    """
    model_a = load_model(f"{filename_stump}-a").cuda()
    model_b = load_model(f"{filename_stump}-b").cuda()
    return model_a, model_b


#######################
# model manipulations #
#######################


def model_like(model: str | torch.nn.Module) -> torch.nn.Module:
    """
    Creates a new model with the same hyperparameters as the reference model (but newly initialized parameters)
    :param model: the filename of the reference model or the reference model itself
    :return: a freshly initialized model of the same type
    """
    if isinstance(model, str):
        dataset, model_type, size, batch_norm, width, _ = parse_model_name(model)
        num_classes = get_num_classes(dataset)
    elif isinstance(model, torch.nn.Module):
        model_type = "VGG" if isinstance(model, VGG) else "MLP" if isinstance(model, MLP) else "ResNet"
        size = model.size
        width = model.width
        batch_norm = model.bn
        num_classes = model.num_classes
    else:
        raise ValueError("Model has to be string (filename) or model instance")
    if model_type == "VGG":
        new_model = VGG(size=size, width=width, bn=batch_norm, num_classes=num_classes)
    elif model_type == "ResNet":
        if size == 18:
            ResNet = ResNet18
        elif size == 20:
            ResNet = ResNet20
        else:
            raise ValueError(f"Unknown ResNet size {size}")
        new_model = ResNet(width=width, num_classes=num_classes)
    elif model_type == "MLP":
        new_model = MLP(size=size, width=width, bn=batch_norm, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type {model_type} in {model}")
    return new_model


def expand_model(
    model: torch.nn.Module, expansion_factor: int | float | list[float] | torch.FloatTensor, append: str = "right"
):
    """
    Returns a functionally equivalent but wider model. The appended weights and biases are all zero.
    :param model: the original model
    :param expansion_factor: the factor by which to expand/widen the model (must be >1);
                             alternatively you can provide a list or FloatTensor of length model.num_layers, which
                             expands each layer of the model by a different factor (at least one must be >1)
    :param append: whether to append the new zero-weights/-biases to the right or the left of the tensor
    :return: the expanded model
    """
    if type(expansion_factor) is int:
        expansion_factor = float(expansion_factor)
    if type(expansion_factor) is float:
        assert expansion_factor > 1, "Expansion factor must be greater than 1.0"
        expansion_factor = [expansion_factor] * model.num_layers
    assert (
        len(expansion_factor) == model.num_layers
    ), f"Expansion factor list has wrong length; is: {len(expansion_factor)}, needed: {model.num_layers}"
    expansion_factor = torch.FloatTensor(expansion_factor)
    assert expansion_factor.min() >= 1.0, "Expansion factors <1 are not allowed"
    assert expansion_factor.max() > 1.0, "At least one expansion factor must be >1"
    assert append in ["right", "left"], "Append parameter must be 'right' or 'left'"

    if isinstance(model, VGG):
        model_expanded = VGG(
            size=model.size, width=model.width * expansion_factor, bn=model.bn, num_classes=model.num_classes
        )
    elif isinstance(model, ResNet18):
        assert (
            isinstance(expansion_factor, int)
            or isinstance(expansion_factor, float)
            or (
                len(expansion_factor) == 17
                and expansion_factor[0] == expansion_factor[2] == expansion_factor[4]
                and expansion_factor[6] == expansion_factor[8]
                and expansion_factor[10] == expansion_factor[12]
                and expansion_factor[14] == expansion_factor[16]
            )
        ), "width of residual activations must match after expansion"
        model_expanded = ResNet18(width=model.width * expansion_factor, num_classes=model.num_classes)
    elif isinstance(model, ResNet20):
        model_expanded = ResNet20(width=model.width * expansion_factor, num_classes=model.num_classes)
    elif isinstance(model, MLP):
        model_expanded = MLP(
            size=model.size, width=model.width * expansion_factor, bn=model.bn, num_classes=model.num_classes
        )
    else:
        raise ValueError("Unknown model type")

    sd_expanded = model_expanded.state_dict()
    sd = model.state_dict()
    for key in sd.keys():
        if "is_buffer" in key:  # e.g. features.0.is_buffer
            sd_expanded[key] = torch.ones_like(sd_expanded[key]).bool()  # init is_buffer flags as True
        else:  # e.g. features.0.weight, features.0.bias, classifier.weight, classifier.bias, ...
            sd_expanded[key] = torch.zeros_like(sd_expanded[key])  # init weights/biases as 0.0
        if append == "right":
            slice_indices = tuple(slice(0, sd[key].size(i)) for i in range(sd[key].dim()))
        else:  # append == "left"
            slice_indices = tuple(
                slice(sd_expanded[key].size(i) - sd[key].size(i), sd_expanded[key].size(i))
                for i in range(sd[key].dim())
            )
        sd_expanded[key][slice_indices] = sd[key]
    model_expanded.load_state_dict(sd_expanded)

    return model_expanded


def subnet(
    model: torch.nn.Module, num_layers: int, only_return: str = None, with_relu: bool = True
) -> torch.nn.Sequential:
    """
    Returns a subnet from layer 1 to layer n_layers (only counting conv layers in the feature extractor; no classifier).
    The returned subnet will include following bn and relu layers before the next conv, but no pooling layers.
    adapted from https://github.com/KellerJordan/REPAIR
    :param model: the original model
    :param num_layers: the first n_layers will be sliced
    :param only_return: for ResNets: after a block, only return the "residual" or "downsample" output (i.e. don't add)
                        do *not* use this for permuting, as it returns a copy of the model
    :param with_relu: add ReLU layer at end if True, otherwise the subnet ends with linear/conv layer  TODO: implement for ResNet
    :return: torch.nn.Module
    """
    assert isinstance(num_layers, int) and 0 < num_layers <= model.num_layers

    if isinstance(model, MLP):
        result = torch.nn.Sequential()
        for layer in model.classifier:
            result.append(layer)
            if num_layers == 0:
                if isinstance(layer, torch.nn.ReLU):
                    break
            if isinstance(layer, torch.nn.Linear):
                num_layers -= 1
        if not with_relu:
            if isinstance(result[-1], torch.nn.ReLU):
                return result[:-1]
        return result

    elif isinstance(model, VGG):
        result = torch.nn.Sequential()
        for layer in model.features:
            result.append(layer)
            if num_layers == 0:
                if isinstance(layer, torch.nn.ReLU):
                    break
            if isinstance(layer, torch.nn.Conv2d):
                num_layers -= 1
        if not with_relu:
            if isinstance(result[-1], torch.nn.ReLU):
                return result[:-1]
        return result

    elif isinstance(model, ResNet18 | ResNet20):
        blocks = get_blocks(model)
        if num_layers % 2 == 1:
            index = (num_layers + 1) // 2
            result = blocks[:index]
            if only_return is not None:
                result = deepcopy(result)
                match only_return:
                    case "residual":
                        new_forward_func = forward_just_residual
                    case "downsample":
                        new_forward_func = forward_just_downsample
                    case _:
                        raise ValueError("Invalid only_return argument")
                funcType = type(result[-1].forward)
                result[-1].forward = funcType(new_forward_func, result[-1])
            return result
        else:
            index = num_layers // 2
            result = blocks[:index]
            result.append(torch.nn.Sequential(blocks[index].conv1, blocks[index].bn1, torch.nn.ReLU()))
            return result

    else:
        raise NotImplementedError(f"Cannot create subnet of type {type(model)}")


def get_blocks(resnet: ResNet18 | ResNet20) -> torch.nn.Sequential:
    """
    Returns the individual blocks of the ResNet as an iterable
    :param resnet: the ResNet model
    :return: the iterable
    """
    first_block = torch.nn.Sequential(resnet.conv1, resnet.bn1)
    if hasattr(resnet, "relu"):  # torchvision.models.resnet has named layer
        first_block.append(resnet.relu)
    else:
        first_block.append(torch.nn.ReLU())
    if hasattr(resnet, "maxpool"):  # torchvision.models.resnet uses maxpool in layer 1
        first_block.append(resnet.maxpool)
    blocks = torch.nn.Sequential(
        first_block,
        *resnet.layer1,
        *resnet.layer2,
        *resnet.layer3,
    )
    if hasattr(resnet, "layer4"):  # ResNet18
        blocks.extend(resnet.layer4)
    return blocks


def add_junctures(resnet: ResNet18 | ResNet20):
    """
    Adds artificial downsampling point-wise convolutions to BasicBlocks to keep track of  TODO: not needed, delete
    applied permutations while aligning two models.
    :param resnet: the ResNet
    :return: a new identical ResNet with junctures
    """
    new_resnet = model_like(resnet)
    new_resnet.load_state_dict(resnet.state_dict())
    blocks = get_blocks(new_resnet)[1:]
    for block in blocks:
        if block.downsample is not None and len(block.downsample) > 0:
            continue
        planes = len(block.bn2.weight)
        shortcut = torch.nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        shortcut.weight.data[:, :, 0, 0] = torch.eye(planes)
        block.downsample = shortcut
    return new_resnet.cuda().eval()


def get_num_layers(model):
    """
    Returns the number of layers ({conv + bn + relu} and classifier) of a model
    :param model: the model
    :return: the number of layers
    """
    if isinstance(model, VGG):
        conv_num = len([layer for layer in model.features if isinstance(layer, torch.nn.Conv2d)])
        classifier = 1
        return conv_num + classifier
    elif isinstance(model, ResNet18):
        raise NotImplementedError()  # TODO
    elif isinstance(model, ResNet20):
        raise NotImplementedError()  # TODO


def get_num_params(model):
    """
    Returns the total number of parameters in a model
    :param model: the model
    :return: the total number of parameters
    """
    return sum([v.numel() for k, v in model.state_dict().items() if "is_buffer" not in k])


def remove_buffer_flags(model):
    """
    Sets all .is_buffer parameters in the model to None (only necessary for sinkhorn-rebasin)
    :param model: the model
    :return: None, modifies the model in-place
    """
    for module in model.modules():
        if "is_buffer" in dict(module.named_parameters()).keys():
            module.is_buffer = None


#####################
# merging functions #
#####################

# flake8: noqa: C901
def permute_model(reference_model: torch.nn.Module, model: torch.nn.Module, loader, save_corr_path: str = None):
    """
    Merges the two models using traditional activation matching
    adapted from https://github.com/KellerJordan/REPAIR
    :param reference_model: the reference model (not affected)
    :param model: the model to be permuted
    :param loader: the data loader to use for calculating the activations; usually train_aug_loader
    :param save_corrs: save the candidate and LAP-selected correlations in each layer under this path
    :return: the permuted model
    """
    sd = model.state_dict()
    model = model_like(model)
    model.load_state_dict(sd)
    model.cuda().eval()
    reference_model.cuda().eval()

    if isinstance(model, MLP | VGG):
        for layer in range(1, model.num_layers + 1):
            subnet_ref = subnet(reference_model, layer)
            subnet_model = subnet(model, layer)
            if layer >= 2:
                permute_input(perm_map, get_last_layer_from_subnet(subnet_model, "conv/linear"))
            perm_map = get_layer_perm(subnet_ref, subnet_model, loader, save_corr_path, layer=layer)
            permute_output(
                perm_map,
                get_last_layer_from_subnet(subnet_model, "conv/linear"),
                get_last_layer_from_subnet(subnet_model, "bn"),
            )
            if layer == model.num_layers:
                if isinstance(model, MLP):
                    permute_input(perm_map, model.classifier[-2])
                else:  # VGG
                    permute_input(perm_map, model.classifier)

    elif isinstance(model, ResNet18):
        # we just record the correlations here but don't use them
        if save_corr_path is not None:
            for layer in [1, 3, 7, 11, 15]:
                subnet_ref = subnet(reference_model, layer)
                subnet_model = subnet(model, layer)
                _ = get_layer_perm(subnet_ref, subnet_model, loader, save_corr_path, layer=layer)
            for layer in [7, 11, 15]:
                subnet_ref = subnet(reference_model, layer, only_return="downsample")
                subnet_model = subnet(model, layer, only_return="downsample")
                _ = get_layer_perm(subnet_ref, subnet_model, loader, save_corr_path, layer=f"{layer}.downsample")
                subnet_ref = subnet(reference_model, layer, only_return="residual")
                subnet_model = subnet(model, layer, only_return="residual")
                _ = get_layer_perm(subnet_ref, subnet_model, loader, save_corr_path, layer=f"{layer}.residual")
        # intra-block permutation
        for layer in [2, 4, 6, 8, 10, 12, 14, 16]:
            subnet_ref = subnet(reference_model, layer)
            subnet_model = subnet(model, layer)
            perm_map = get_layer_perm(subnet_ref, subnet_model, loader, save_corr_path, layer=layer)
            subnet_model = subnet(model, layer + 1)
            permute_output(perm_map, subnet_model[-1].conv1, subnet_model[-1].bn1)
            permute_input(perm_map, subnet_model[-1].conv2)
        # inter-block permutation
        for layer in [5, 9, 13, 17]:
            subnet_ref = subnet(reference_model, layer)
            subnet_model = subnet(model, layer)
            if layer >= 9:
                permute_input(perm_map, [subnet_model[-2].conv1, subnet_model[-2].downsample[0]])
            perm_map = get_layer_perm(subnet_ref, subnet_model, loader, save_corr_path, layer=layer)
            if layer == 5:  # special case for first conv
                permute_output(perm_map, model.conv1, model.bn1)
                permute_input(perm_map, [subnet_model[-1].conv1, subnet_model[-2].conv1])
            else:
                permute_output(perm_map, subnet_model[-2].downsample[0], subnet_model[-2].downsample[1])
                permute_input(perm_map, [subnet_model[-1].conv1])
            if layer == 17:  # special case for linear classifier
                permute_input(perm_map, model.linear)
            permute_output(perm_map, subnet_model[-1].conv2, subnet_model[-1].bn2)
            permute_output(perm_map, subnet_model[-2].conv2, subnet_model[-2].bn2)

    elif isinstance(model, ResNet20):
        # we just record the correlations here but don't use them
        if save_corr_path is not None:
            for layer in [1, 3, 5, 9, 11, 15, 17]:
                subnet_ref = subnet(reference_model, layer)
                subnet_model = subnet(model, layer)
                _ = get_layer_perm(subnet_ref, subnet_model, loader, save_corr_path, layer=layer)
        # intra-block permutation
        for layer in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
            subnet_ref = subnet(reference_model, layer)
            subnet_model = subnet(model, layer)
            perm_map = get_layer_perm(subnet_ref, subnet_model, loader, save_corr_path, layer=layer)
            subnet_model = subnet(model, layer + 1)
            permute_output(perm_map, subnet_model[-1].conv1, subnet_model[-1].bn1)
            permute_input(perm_map, subnet_model[-1].conv2)
        # inter-block permutation
        for layer in [7, 13, 19]:
            subnet_ref = subnet(reference_model, layer)
            subnet_model = subnet(model, layer)
            if layer >= 13:
                permute_input(perm_map, [subnet_model[-3].conv1, subnet_model[-3].downsample[0]])
            perm_map = get_layer_perm(subnet_ref, subnet_model, loader, save_corr_path, layer=layer)
            if layer == 7:  # special case for first conv
                permute_output(perm_map, model.conv1, model.bn1)
                permute_input(perm_map, [subnet_model[-1].conv1, subnet_model[-2].conv1, subnet_model[-3].conv1])
            else:
                permute_output(perm_map, subnet_model[-3].downsample[0], subnet_model[-3].downsample[1])
                permute_input(perm_map, [subnet_model[-1].conv1, subnet_model[-2].conv1])
            if layer == 19:  # special case for linear classifier
                permute_input(perm_map, model.linear)
            permute_output(perm_map, subnet_model[-1].conv2, subnet_model[-1].bn2)
            permute_output(perm_map, subnet_model[-2].conv2, subnet_model[-2].bn2)
            permute_output(perm_map, subnet_model[-3].conv2, subnet_model[-3].bn2)

    else:
        raise ValueError(f"Unknown model type {type(model)}")

    return model


# modifies the weight matrices of a convolution and batchnorm
# layer given a permutation of the output channels
def permute_output(perm_map, layer, bn=None):
    """
    TODO: write docs
    adapted from https://github.com/KellerJordan/REPAIR
    :param perm_map:
    :param layer:
    :param bn:
    :return:
    """
    assert isinstance(layer, torch.nn.Conv2d | torch.nn.Linear), "layer is not Conv2d or Linear"
    assert bn is None or isinstance(
        bn, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d
    ), "bn layer is not BatchNorm1d or BatchNorm2d"

    pre_weights = [layer.weight]
    if layer.bias is not None:
        pre_weights.append(layer.bias)
    if layer.is_buffer is not None:
        pre_weights.append(layer.is_buffer)
    if bn is not None:
        pre_weights.extend(
            [
                bn.weight,
                bn.bias,
                bn.running_mean,
                bn.running_var,
            ]
        )
        if bn.is_buffer is not None:
            pre_weights.append(bn.is_buffer)
    for w in pre_weights:
        w.data = w[perm_map]


# modifies the weight matrix of a layer for a given permutation of the input channels
# works for both conv2d and linear
def permute_input(perm_map, after_layers):
    """
    TODO: write docs
    :param perm_map:
    :param after_layers:
    :return:
    """
    if not isinstance(after_layers, list):
        after_layers = [after_layers]
    post_weights = [c.weight for c in after_layers]
    for w in post_weights:
        w.data = w[:, perm_map]


def interpolate_models(model_a: torch.nn.Module, model_b: torch.nn.Module, alpha: float = 0.5):
    """
    Interpolates the parameters between two models a and b. Does *not* permute/align the models for you.
    :param model_a: the first model
    :param model_b: the second model
    :param alpha: the interpolation percentage for model_b; 1-alpha for model a
    :return: the interpolated child model
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    sd_interpolated = {key: (1 - alpha) * sd_a[key].cuda() + alpha * sd_b[key].cuda() for key in sd_a.keys()}
    model_merged = model_like(model_a)
    model_merged.load_state_dict(sd_interpolated)
    return model_merged


def add_models(model_a: torch.nn.Module, model_b: torch.nn.Module):
    """
    Adds up the parameters of two models a and b. Does *not* permute/align the models for you.
    :param model_a: the first model
    :param model_b: the second model
    :return: the added up child model
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    sd_interpolated = {key: sd_a[key].cuda() + sd_b[key].cuda() for key in sd_a.keys()}
    model_merged = model_like(model_a)
    model_merged.load_state_dict(sd_interpolated)
    return model_merged


def smart_interpolate_models(model_a: torch.nn.Module, model_b: torch.nn.Module, alpha: float = 0.5):
    """
    Interpolates the parameters between two models a and b. Does *not* permute/align the models for you.
    When at least one of the two parameters originates from a buffer neuron, the parameters get added instead
    of interpolated.
    TODO implement for ResNet
    :param model_a: the first model
    :param model_b: the second model
    :param alpha: the interpolation percentage for model_b; 1-alpha for model a
    :return: the interpolated child model
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    sd_interpolated = {}
    for key in sd_a.keys():
        matching_buffer = key.split(".")
        matching_buffer[-1] = "is_buffer"
        matching_buffer = ".".join(matching_buffer)
        if key.endswith("is_buffer"):
            sd_interpolated[key] = (sd_a[key] | sd_b[key]).cuda()  # TODO replace with bit-wise and (&)?
        elif key.endswith("num_batches_tracked"):
            sd_interpolated[key] = ((sd_a[key] + sd_b[key]) / 2).long().cuda()
        elif matching_buffer in sd_a.keys():
            mask = sd_a[matching_buffer] | sd_b[matching_buffer]
            sd_interpolated[key] = torch.where(
                mask.view(-1, *((1,) * (sd_a[key].dim() - 1))).expand_as(sd_a[key]),
                sd_a[key].cuda() + sd_b[key].cuda(),
                (1 - alpha) * sd_a[key].cuda() + alpha * sd_b[key].cuda(),
            )
        else:
            sd_interpolated[key] = (1 - alpha) * sd_a[key].cuda() + alpha * sd_b[key].cuda()
    model_merged = model_like(model_a)
    model_merged.load_state_dict(sd_interpolated)
    return model_merged


def interpolate_models_keep_overlap(model_a: torch.nn.Module, model_b: torch.nn.Module, alpha: float = 0.5):
    """
    Interpolates the parameters between two models a and b. Does *not* permute/align the models for you.
    When interpolating expanded models, only the overlapping non-buffer zones are being kept.
    TODO implement for ResNet
    :param model_a: the first model
    :param model_b: the second model
    :param alpha: the interpolation percentage for model_b; 1-alpha for model a
    :return: the interpolated child model
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    sd_interpolated = {}
    for key in sd_a.keys():
        matching_buffer = key.split(".")
        matching_buffer[-1] = "is_buffer"
        matching_buffer = ".".join(matching_buffer)
        if key.endswith("is_buffer"):
            sd_interpolated[key] = torch.zeros_like(sd_a[key]).bool().cuda()
        elif matching_buffer in sd_a.keys():
            mask = sd_a[matching_buffer] | sd_b[matching_buffer]
            sd_interpolated[key] = torch.where(
                mask.view(-1, *((1,) * (sd_a[key].dim() - 1))).expand_as(sd_a[key]),
                torch.zeros_like(sd_a[key]).cuda(),  # = setting the non-overlapping parts to 0
                (1 - alpha) * sd_a[key].cuda() + alpha * sd_b[key].cuda(),
            )
        else:
            sd_interpolated[key] = (1 - alpha) * sd_a[key].cuda() + alpha * sd_b[key].cuda()
    model_merged = model_like(model_a)
    model_merged.load_state_dict(sd_interpolated)
    return model_merged


def interpolate_models_discard_overlap(model_a: torch.nn.Module, model_b: torch.nn.Module, alpha: float = 0.5):
    """
    Interpolates the parameters between two models a and b. Does *not* permute/align the models for you.
    When interpolating expanded models, only the non-overlapping buffer zones are being kept (= partial ensembling).
    TODO implement for ResNet
    :param model_a: the first model
    :param model_b: the second model
    :param alpha: the interpolation percentage for model_b; 1-alpha for model a
    :return: the interpolated child model
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    sd_interpolated = {}
    for key in sd_a.keys():
        matching_buffer = key.split(".")
        matching_buffer[-1] = "is_buffer"
        matching_buffer = ".".join(matching_buffer)
        if key.endswith("is_buffer"):
            sd_interpolated[key] = torch.zeros_like(sd_a[key]).bool().cuda()
        elif matching_buffer in sd_a.keys():
            mask = sd_a[matching_buffer] | sd_b[matching_buffer]
            sd_interpolated[key] = torch.where(
                mask.view(-1, *((1,) * (sd_a[key].dim() - 1))).expand_as(sd_a[key]),
                sd_a[key].cuda() + sd_b[key].cuda(),
                torch.zeros_like(sd_a[key]).cuda(),  # = setting the overlapping parts to 0
            )
        else:
            sd_interpolated[key] = (1 - alpha) * sd_a[key].cuda() + alpha * sd_b[key].cuda()
    model_merged = model_like(model_a)
    model_merged.load_state_dict(sd_interpolated)
    return model_merged


################################
# correlation matrix functions #
################################


def get_corr_matrix(
    subnet_a: torch.nn.Module, subnet_b: torch.nn.Module, loader: Loader, epochs: int = 1, strategy: int = 1
):
    """
    Given two networks subnet_a, subnet_b which each output a feature map of shape NxCxWxH this will reshape
    both outputs to (N*W*H)xC and then compute a CxC correlation matrix between the outputs of the two networks
    N = dataset size, C = # of individual feature maps, H, W = height and width of one feature map
    adapted from https://github.com/KellerJordan/REPAIR
    :param subnet_a:
    :param subnet_b:
    :param loader: a data loader that resembles the input distribution (typically the train_aug_loader)
    :param epochs: for how many epochs to collect feature maps (values >1 only make a difference if the loader
                   uses augmentations)
    :param strategy: which strategy to use:
                      1. use correlation matrix for matching
                      2. use covariance matrix for matching
                      3. use correlation matrix + normalized mean activation of reference model for matching
                      4. use correlation matrix + normalized mean activation of model for matching
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
    if strategy == 2:
        return cov
    corr = cov / (torch.outer(std_a, std_b) + 1e-4)
    # corr = manipulate_corr_matrix(corr)  # TODO: fix first before re-using! detects buffer neurons incorrectly
    if strategy == 1:
        return corr
    elif strategy == 3:
        return corr + normalize(mean_a).unsqueeze(1)
    elif strategy == 4:
        return corr + normalize(mean_b)


def smart_get_corr_matrix(
    subnet_a: torch.nn.Sequential, subnet_b: torch.nn.Sequential, loader: Loader, epochs: int = 1, strategy: int = 1
):
    """
    The same as get_corr_matrix, but takes is_buffer flags into account and modifies the correlation
    matrix accordingly (by setting buffer parts to 1.1 or -1.1).
    """
    reference_is_buffer = get_is_buffer_from_subnet(subnet_a)
    model_is_buffer = get_is_buffer_from_subnet(subnet_b)
    corr = get_corr_matrix(subnet_a, subnet_b, loader, epochs, strategy)
    corr[reference_is_buffer, :] = 1.1
    corr[:, model_is_buffer] = 1.1
    corr[reference_is_buffer.unsqueeze(1) & model_is_buffer.unsqueeze(0)] = -1.1
    return corr


def get_is_buffer_from_subnet(net: torch.nn.Sequential) -> torch.BoolTensor:
    """
    Given any subnet made with subnet(), returns the last relevant is_buffer parameter.
    :param net: the subnet
    :return: the is_buffer torch.BoolTensor
    """
    if isinstance(net[-1], BasicBlock):  # it's a ResNet block
        is_buffer = net[-1].bn2.is_buffer
    elif isinstance(net[-1], torch.nn.Sequential):  # it's half a ResNet block
        is_buffer = net[-1][1].is_buffer
    elif isinstance(
        net[-1], torch.nn.Linear | torch.nn.Conv2d | torch.nn.BatchNorm1d | torch.nn.BatchNorm2d
    ):  # it's an MLP/VGG without a final ReLU
        is_buffer = net[-1].is_buffer
    elif isinstance(
        net[-2], torch.nn.Linear | torch.nn.Conv2d | torch.nn.BatchNorm1d | torch.nn.BatchNorm2d
    ):  # it's an MLP/VGG with final ReLU
        is_buffer = net[-2].is_buffer
    else:
        raise ValueError(f"Unknown subnet makeup, last module has type {type(net[-1])}")
    assert isinstance(is_buffer, torch.BoolTensor | torch.cuda.BoolTensor)
    return is_buffer


def get_last_layer_from_subnet(net: torch.nn.Sequential, layer: str):
    """
    Given any subnet made with subnet(), returns the last relevant layer of kind layer (or None if it doesn't exist).
    :param net: the subnet
    :param layer: the kind of layer you're looking for; one of "conv/linear", "bn"
    :return: the layer
    """
    assert layer in ["conv/linear", "bn"], f"Layer parameter must be either 'conv/linear' or 'bn'; actual: {layer}"

    if isinstance(net[-1], BasicBlock):  # it's a ResNet block
        raise NotImplementedError
    elif isinstance(net[-1], torch.nn.Sequential):  # it's half a ResNet block
        raise NotImplementedError
    elif isinstance(net[-1], torch.nn.Linear | torch.nn.Conv2d):  # it's an MLP/VGG without a final ReLU
        if layer == "conv/linear":
            return net[-1]
        if layer == "bn":
            return None
    elif isinstance(net[-1], torch.nn.BatchNorm1d | torch.nn.BatchNorm2d):  # it's an MLP without a final ReLU
        if layer == "conv/linear":
            return net[-2]
        if layer == "bn":
            return net[-1]
    elif isinstance(net[-1], torch.nn.ReLU) and isinstance(
        net[-2], torch.nn.Linear | torch.nn.Conv2d
    ):  # it's an MLP/VGG with a final ReLU
        if layer == "conv/linear":
            return net[-2]
        if layer == "bn":
            return None
    elif isinstance(net[-1], torch.nn.ReLU) and isinstance(
        net[-2], torch.nn.BatchNorm1d | torch.nn.BatchNorm2d
    ):  # it's an MLP with a final ReLU
        if layer == "conv/linear":
            return net[-3]
        if layer == "bn":
            return net[-2]
    else:
        raise ValueError(f"Unknown subnet makeup, last module has type {type(net[-1])}")


def print_corr_matrix(corr_mtx):
    """
    Prints the correlation matrix as color image in the terminal where red = -1, white = 0, blue = 1.
    :param corr_mtx: the correlation matrix
    :return: None
    """
    corr_mtx = np.repeat(corr_mtx, 2).reshape(corr_mtx.shape[0], corr_mtx.shape[1] * 2)
    corr_mtx = (corr_mtx + 1) * 0.5
    cmap = plt.colormaps["RdBu"]
    img = Image.fromarray(np.uint8(cmap(corr_mtx) * 255))
    DrawImage(img, size=(corr_mtx.shape[1], corr_mtx.shape[0])).draw_image()


def manipulate_corr_matrix(corr_mtx):  # TODO: check if still used, then delete!
    """
    Auto-detects the buffer areas in the correlation matrix (all zeros) and manipulates them like this:

    [[0, 0, 0, 0, 0]          [[-1, 1, 1,-1, 1]
     [0, x, x, 0, x]           [ 1, x, x, 1, x]
     [0, x, x, 0, x]    ->     [ 1, x, x, 1, x]
     [0, 0, 0, 0, 0]           [-1, 1, 1,-1, 1]
     [0, x, x, 0, x]]          [ 1, x, x, 1, x]]

    If no buffer zone is detected, the correlation matrix is returned unmodified.

    # TODO: sometimes non-buffer rows/cols have a corr of only zeros, fix that! They should not be set to 1/-1!

    :param corr_mtx: the correlation matrix
    :return: the manipulated correlation matrix
    """
    assert corr_mtx.dim() == 2
    assert corr_mtx.shape[0] == corr_mtx.shape[1]  # not strictly necessary

    all_zero_rows = torch.atleast_1d(torch.nonzero(torch.all(corr_mtx == 0, dim=1)).squeeze())
    all_zero_cols = torch.atleast_1d(torch.nonzero(torch.all(corr_mtx == 0, dim=0)).squeeze())

    print(all_zero_rows)
    print(all_zero_cols)

    corr_mtx[all_zero_rows] = 1
    corr_mtx[:, all_zero_cols] = 1
    overlapping = torch.cartesian_prod(all_zero_rows, all_zero_cols)
    print(overlapping)
    corr_mtx[overlapping[:, 0], overlapping[:, 1]] = -1

    return corr_mtx


def get_layer_perm_from_corr(corr_mtx, save_corr_path: str = None, layer: int | str = None):
    """
    Given a correlation matrix, returns the optimal permutation map that the LAP solver returns.
    :param corr_mtx: a correlation matrix
    :param save_corr_path: the path under which to save the candidate and LAP-selected correlations
    :param layer: which layer we are currently evaluating (relevant for the save_corr_path dict)
    :return: the permutation map
    """
    corr_mtx = ensure_numpy(corr_mtx)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx, maximize=True)
    assert (row_ind == np.arange(len(corr_mtx))).all()
    perm_map = torch.LongTensor(col_ind)
    if save_corr_path is not None:
        assert save_corr_path.endswith(".safetensors")
        if os.path.isfile(save_corr_path) and layer != 1:
            corrs = load_file(save_corr_path)
        else:
            corrs = {}
        corrs[f"layer{layer}.correlations"] = torch.FloatTensor(corr_mtx)
        corrs[f"layer{layer}.perm_map"] = perm_map
        corrs = dict(sorted(corrs.items(), key=lambda x: int(x[0].split(".")[0].replace("layer", ""))))  # sort dict
        print(f"Saving corrs of layer {layer} to {save_corr_path}")
        print("Candidates:", corr_mtx.mean())
        print("Selected:", corr_mtx[row_ind, col_ind].mean())
        save_file(corrs, filename=save_corr_path)
    return perm_map


def get_layer_perm(subnet_a, subnet_b, loader, save_corr_path: str = None, layer: int | str = None):
    """
    Returns the channel permutation map to make the activations of the last layer in subnet_a
    most closely match those in the last layer of subnet_b.
    :param subnet_a: The reference subnet that stays the same
    :param subnet_b: The subnet for which we want the permutation map
    :param loader: The data loader used to collect the activations
    :param save_corr_path: The path under which to save the candidate and LAP-selected correlations
    :param layer: which layer we are currently evaluating (relevant for the save_corr_path dict)
    :return: the permutation map
    """
    corr_mtx = smart_get_corr_matrix(subnet_a, subnet_b, loader)
    return get_layer_perm_from_corr(corr_mtx, save_corr_path, layer)


####################
# REPAIR functions #
####################


class REPAIRTracker(torch.nn.Module):
    """
    A wrapper class for tracking (and later resetting) linear or conv activations with REPAIR
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        if isinstance(layer, torch.nn.Conv2d):
            self.h = h = layer.out_channels
            self.bn = torch.nn.BatchNorm2d(h)
        elif isinstance(layer, torch.nn.Linear):
            self.h = h = layer.out_features
            self.bn = torch.nn.BatchNorm1d(h)
        else:
            raise ValueError("Unknown layer type")
        self.rescale = False

    def set_stats(self, goal_mean, goal_var):
        self.bn.bias.data = goal_mean
        goal_std = (goal_var + 1e-5).sqrt()
        self.bn.weight.data = goal_std

    def forward(self, x):
        x = self.layer(x)
        if self.rescale:
            x = self.bn(x)
        else:
            self.bn(x)
        return x


def make_tracked_model(model):
    """
    Converts a regular model (e.g. VGG) into a tracked model (for rescaling the activations).
    :param model: the input model
    :return: the (functionally equivalent) tracked model
    """
    tracked_model = model_like(model).cuda()
    tracked_model.load_state_dict(model.state_dict())
    if isinstance(model, VGG | ResNet18 | ResNet20):
        feats = tracked_model.features
    elif isinstance(model, MLP):
        feats = tracked_model.classifier[:-2]
    else:
        raise ValueError("Unknown model type")
    for i, layer in enumerate(feats):
        if isinstance(layer, torch.nn.Conv2d | torch.nn.Linear):
            feats[i] = REPAIRTracker(layer)
    if isinstance(model, MLP):
        tracked_model.classifier = feats.extend(tracked_model.classifier[-2:])
    return tracked_model.cuda().eval()


def fuse_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    """
    Fuses the convolutional and tracking batch norm layer into a single convolutional layer with appropriate
    parameters that exhibits the same activation pattern.
    :param conv: the Conv2d layer
    :param bn: the batch norm layer
    :return: the new Conv2d layer
    """
    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True,
    )
    fused.is_buffer = conv.is_buffer

    # setting weights
    w_conv = conv.weight.clone()
    bn_std = (bn.eps + bn.running_var).sqrt()
    gamma = bn.weight / bn_std
    fused.weight.data = w_conv * gamma.reshape(-1, 1, 1, 1)

    # setting bias
    b_conv = conv.bias if conv.bias is not None else torch.zeros_like(bn.bias)
    beta = bn.bias + gamma * (-bn.running_mean + b_conv)
    fused.bias.data = beta
    return fused


def fuse_linear_bn(linear: torch.nn.Linear, bn: torch.nn.BatchNorm1d):
    """
    Fuses the linear and tracking batch norm layer into a single linear layer with appropriate
    parameters that exhibits the same activation pattern.
    :param linear: the linear layer
    :param bn: the batch norm layer
    :return: the new linear layer
    """
    fused = torch.nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=True,
    )
    fused.is_buffer = linear.is_buffer

    # setting weights
    w_linear = linear.weight.clone()
    bn_std = (bn.eps + bn.running_var).sqrt()
    gamma = bn.weight / bn_std
    fused.weight.data = w_linear * gamma.reshape(-1, 1)

    # setting bias
    b_linear = linear.bias if linear.bias is not None else torch.zeros_like(bn.bias)
    beta = bn.bias + gamma * (-bn.running_mean + b_linear)
    fused.bias.data = beta
    return fused


def fuse_tracked_model(model):
    """
    Fuses all REPAIRTracker layers in a tracked model.
    :param model: the tracked model
    :return: the fused (regular) model
    """
    model_fused = model_like(model).cuda()
    if isinstance(model, VGG | ResNet18 | ResNet20):
        feats = model_fused.features
        for i, layer in enumerate(model.features):
            if isinstance(layer, REPAIRTracker):
                conv = fuse_conv_bn(layer.layer, layer.bn)
                feats[i].load_state_dict(conv.state_dict())
        model_fused.classifier.load_state_dict(model.classifier.state_dict())
    elif isinstance(model, MLP):
        feats = model_fused.classifier[:-2]
        for i, layer in enumerate(model.classifier[:-2]):
            if isinstance(layer, REPAIRTracker):
                linear = fuse_linear_bn(layer.layer, layer.bn)
                feats[i].load_state_dict(linear.state_dict())
        model_fused.classifier[-2:].load_state_dict(model.classifier[-2:].state_dict())
    return model_fused


def reset_bn_stats(model: torch.nn.Module, loader, epochs: int = 1) -> None:
    """
    Recalculates and overwrites the batch norm statistics in all BatchNorm2d layers of the model.
    adapted from https://github.com/KellerJordan/REPAIR
    :param model: the model which to modify
    :param loader: the loader for calculating the batch norm stats; typically train_aug_loader
    :param epochs: for how many epochs to collect the batch norm stats; more is better (only if augmentations are used)
    :return: None
    """
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d):
            m.momentum = None  # use simple average
            m.reset_running_stats()

    # run a single no_grad train epoch to recalc bn stats (this happens automatically)
    model.train()
    for _ in range(epochs):
        with torch.no_grad(), autocast():
            for images, _ in loader:
                _ = model(images.cuda())


def repair(model, parent_model_a, parent_model_b, loader, alpha: float = 0.5):
    """
    REPAIRs a (merged) model
    adapted from https://github.com/KellerJordan/REPAIR
    :param model: the merged model before REPAIR
    :param parent_model_a: one of the parent models
    :param parent_model_b: the other parent model (order doesn't matter)
    :param loader: a data loader for recalculating the barch norm statistics; typically train_aug_loader
    :param alpha: the alpha value used to create the model from the parent models
    :return: the merged model after REPAIR
    """
    # calculate the statistics of every hidden unit in the endpoint networks
    # this is done practically using PyTorch BatchNorm2d layers
    tracked_model_a = make_tracked_model(parent_model_a)
    tracked_model_b = make_tracked_model(parent_model_b)
    tracked_model = make_tracked_model(model)

    reset_bn_stats(tracked_model_a, loader)
    reset_bn_stats(tracked_model_b, loader)

    # set the goal mean/std in added bn layers of interpolated network, and turn batch renormalization on
    for m_a, m, m_b in zip(tracked_model_a.modules(), tracked_model.modules(), tracked_model_b.modules()):
        if not isinstance(m_a, REPAIRTracker):
            continue
        # get goal statistics -- interpolate the mean and std of parent networks
        mu_a = m_a.bn.running_mean
        mu_b = m_b.bn.running_mean
        goal_mean = (1 - alpha) * mu_a + alpha * mu_b
        var_a = m_a.bn.running_var
        var_b = m_b.bn.running_var
        goal_var = ((1 - alpha) * var_a.sqrt() + alpha * var_b.sqrt()).square()
        # set these in the interpolated bn controller
        m.set_stats(goal_mean, goal_var)
        # turn rescaling on
        m.rescale = True

    # reset the tracked mean/var and fuse rescalings back into conv layers
    reset_bn_stats(tracked_model, loader)

    # fuse the rescaling+shift coefficients back into conv layers
    tracked_model = fuse_tracked_model(tracked_model)

    return tracked_model


def partial_repair(model, parent_model_a, parent_model_b, loader, alpha: float = 0.5):
    """
    REPAIRs a (merged) model only in the overlapping part
    adapted from https://github.com/KellerJordan/REPAIR
    :param model: the merged model before REPAIR
    :param parent_model_a: one of the parent models
    :param parent_model_b: the other parent model (order doesn't matter)
    :param loader: a data loader for recalculating the barch norm statistics; typically train_aug_loader
    :param alpha: the alpha value used to create the model from the parent models
    :return: the merged model after REPAIR
    """
    # calculate the statistics of every hidden unit in the endpoint networks
    # this is done practically using PyTorch BatchNorm2d layers
    tracked_model_a = make_tracked_model(parent_model_a)
    tracked_model_b = make_tracked_model(parent_model_b)
    tracked_model = make_tracked_model(model)

    reset_bn_stats(tracked_model_a, loader)
    reset_bn_stats(tracked_model_b, loader)

    # set the goal mean/std in added bn layers of interpolated network, and turn batch renormalization on
    for m_a, m, m_b in zip(tracked_model_a.modules(), tracked_model.modules(), tracked_model_b.modules()):
        if not isinstance(m_a, REPAIRTracker):
            continue
        # get buffer flags
        buffer_a = m_a.layer.is_buffer
        buffer_b = m_b.layer.is_buffer
        mask = buffer_a | buffer_b
        # get goal statistics -- interpolate the mean and std of parent networks
        mu_a = m_a.bn.running_mean
        mu_b = m_b.bn.running_mean
        goal_mean = torch.where(
            mask,
            mu_a + mu_b,
            (1 - alpha) * mu_a + alpha * mu_b,
        )
        var_a = m_a.bn.running_var
        var_b = m_b.bn.running_var
        goal_var = torch.where(
            mask,
            var_a + var_b,
            ((1 - alpha) * var_a.sqrt() + alpha * var_b.sqrt()).square(),
        )
        # set these in the interpolated bn controller
        m.set_stats(goal_mean, goal_var)
        # turn rescaling on
        m.rescale = True

    # reset the tracked mean/var and fuse rescalings back into conv layers
    reset_bn_stats(tracked_model, loader)

    # fuse the rescaling+shift coefficients back into conv layers
    tracked_model = fuse_tracked_model(tracked_model)

    return tracked_model


def get_variance_per_layer(model, loader):
    tracked_model = make_tracked_model(model)
    reset_bn_stats(tracked_model, loader)
    mu_per_layer = []
    var_per_layer = []
    mu_per_layer_buffer = []
    mu_per_layer_nobuffer = []
    var_per_layer_buffer = []
    var_per_layer_nobuffer = []
    for m in tracked_model.modules():
        if not isinstance(m, REPAIRTracker):
            continue
        buffer = m.conv.is_buffer
        mu = m.bn.running_mean
        var = m.bn.running_var
        mu_per_layer.append(mu.mean().item())
        var_per_layer.append(var.mean().item())
        mu_per_layer_buffer.append(mu[buffer].mean().item())
        var_per_layer_buffer.append(var[buffer].mean().item())
        mu_per_layer_nobuffer.append(mu[~buffer].mean().item())
        var_per_layer_nobuffer.append(var[~buffer].mean().item())
    return torch.Tensor(
        [
            mu_per_layer,
            mu_per_layer_buffer,
            mu_per_layer_nobuffer,
            var_per_layer,
            var_per_layer_buffer,
            var_per_layer_nobuffer,
        ]
    )


############################
# datasets and dataloaders #
############################


def get_num_classes(dataset: str):
    """Returns the number of classes in a dataset"""
    cfg = {"CIFAR10": 10, "CIFAR100": 100, "SVHN": 10, "MNIST": 10}
    return cfg[dataset]


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


def _download_dataset(dataset) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Returns (and downloads, if necessary) the train and test dataset
    :param dataset: one of 'CIFAR10', 'CIFAR100', 'SVHN'  TODO: add ImageNet
    :return: (train dataset, test dataset)
    """
    data_dir = _get_data_dir()
    if dataset == "CIFAR10":
        train_dset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
        test_dset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    elif dataset == "CIFAR100":
        train_dset = torchvision.datasets.CIFAR100(data_dir, train=True, download=True)
        test_dset = torchvision.datasets.CIFAR100(data_dir, train=False, download=True)
    elif dataset == "SVHN":
        train_dset = torchvision.datasets.SVHN(data_dir, split="train", download=True)
        test_dset = torchvision.datasets.SVHN(data_dir, split="test", download=True)
    elif dataset == "MNIST":
        train_dset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        test_dset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return train_dset, test_dset


def _get_beton_path(dataset) -> tuple[str, str]:
    """
    Returns the .beton filepaths, downloads and converts the dataset first if they don't already exist.
    :param dataset: one of 'CIFAR10', 'CIFAR100'  TODO: add more
    :return: (train .beton filepath, test .beton filepath)
    """
    data_dir = _get_data_dir()
    train_beton_path = os.path.join(data_dir, f"{dataset}_train.beton")
    test_beton_path = os.path.join(data_dir, f"{dataset}_test.beton")
    if not (os.path.exists(train_beton_path) and os.path.exists(test_beton_path)):
        print(f"{dataset} dataset not present - downloading and/or converting ...")
        train_dset, test_dset = _download_dataset(dataset)
        _convert_dataset_to_beton(train_dset, train_beton_path)
        _convert_dataset_to_beton(test_dset, test_beton_path)
    return train_beton_path, test_beton_path


def get_loaders(dataset: str) -> tuple[Loader, Loader, Loader] | tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns three FFCV (or PyTorch for b/w images) loaders.
    Downloads and converts the underlying dataset if necessary.
    adapted from https://github.com/KellerJordan/REPAIR
    :param dataset: one of 'CIFAR10', 'CIFAR100', 'SVHN', 'MNIST'  TODO: add more?
    :return: (train_aug_loader, train_noaug_loader, test_loader)
    """
    dataset = dataset.upper()
    if dataset == "CIFAR10":
        MEAN = [125.307, 122.961, 113.8575]  # correct (these values are from the FFCV CIFAR example)
        STD = [51.5865, 50.847, 51.255]  # too low, but kept as is for reproducibility
        # MEAN = [125.30691805, 122.95039414, 113.86538318]  # correct values
        # STD = [62.99321928, 62.08870764, 66.70489964]      # correct values
    elif dataset == "CIFAR100":
        MEAN = [129.30416561, 124.0699627, 112.43405006]
        STD = [68.1702429, 65.39180804, 70.41837019]
    elif dataset == "SVHN":
        MEAN = [111.60893668, 113.16127466, 120.56512767]
        STD = [50.49768174, 51.2589843, 50.24421614]
    elif dataset == "MNIST":
        return _get_loaders_no_FFCV(dataset)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    device = torch.device("cuda:0")
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
    pre_p = [SimpleRGBImageDecoder()]
    post_p = [
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        T.Normalize(MEAN, STD),
    ]
    aug_p = (
        [
            RandomHorizontalFlip(),
            RandomTranslate(padding=4),
        ]
        if dataset not in ["SVHN", "MNIST"]
        else [RandomTranslate(padding=4)]
    )

    train_beton_path, test_beton_path = _get_beton_path(dataset)

    train_aug_loader = Loader(
        train_beton_path,
        batch_size=500,
        num_workers=min(8, len(os.sched_getaffinity(0))),
        order=OrderOption.RANDOM,
        drop_last=True,
        pipelines={"image": pre_p + aug_p + post_p, "label": label_pipeline},
    )
    train_noaug_loader = Loader(
        train_beton_path,
        batch_size=1000,
        num_workers=min(8, len(os.sched_getaffinity(0))),
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines={"image": pre_p + post_p, "label": label_pipeline},
    )
    test_loader = Loader(
        test_beton_path,
        batch_size=1000,
        num_workers=min(8, len(os.sched_getaffinity(0))),
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines={"image": pre_p + post_p, "label": label_pipeline},
    )

    return train_aug_loader, train_noaug_loader, test_loader


def _get_loaders_no_FFCV(dataset: str) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = dataset.upper()

    aug_transform = T.Compose(
        [
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            T.ToTensor(),
            # T.Normalize(MEAN, STD),
            T.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    noaug_transform = T.Compose(
        [
            T.ToTensor(),
            # T.Normalize(MEAN, STD),
            T.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    data_dir = _get_data_dir()
    if dataset == "MNIST":
        train_aug_data = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=aug_transform)
        train_noaug_data = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=noaug_transform)
        test_data = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=noaug_transform)

    train_aug_loader = DataLoader(
        train_aug_data, batch_size=1000, num_workers=min(8, len(os.sched_getaffinity(0))), shuffle=True
    )

    train_noaug_loader = DataLoader(
        train_noaug_data, batch_size=1000, num_workers=min(8, len(os.sched_getaffinity(0))), shuffle=True
    )

    test_loader = DataLoader(test_data, batch_size=1000, num_workers=min(8, len(os.sched_getaffinity(0))), shuffle=True)

    return train_aug_loader, train_noaug_loader, test_loader
