import torch
from tqdm import tqdm
from ffcv.loader import Loader
import scipy
import numpy as np


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


def subnet(model: torch.nn.Module, n_layers: int):
    """
    Returns a subnet from layer 1 to layer n_layers (in the feature extractor)
    :param model: the original model
    :param n_layers: the first n_layers will be sliced
    :return: torch.nn.Module
    """
    return model.features[:n_layers]


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
    pre_weights = [
        conv.weight,
    ]
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
