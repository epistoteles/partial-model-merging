import torch
from tqdm import tqdm
from ffcv.loader import Loader
import scipy
import numpy as np


def run_corr_matrix(
    subnet_a: torch.nn.Module, subnet_b: torch.nn.Module, loader: Loader, epochs: int = 1, norm: bool = True
):
    """
    Given two networks subnet_a, subnet_b which each output a feature map of shape NxCxWxH this will reshape
    both outputs to (N*W*H)xC and then compute a CxC correlation matrix between the outputs of the two networks
    N = dataset size, C = # of individual feature maps, H, W = height and width of one feature map
    """
    n = epochs * len(loader)
    mean0 = mean1 = std0 = std1 = None
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

                mean0_b = out_a.mean(dim=0)
                mean1_b = out_b.mean(dim=0)
                std0_b = out_a.std(dim=0)
                std1_b = out_b.std(dim=0)
                outer_b = (out_a.T @ out_b) / out_a.shape[0]

                if i == 0:
                    mean0 = torch.zeros_like(mean0_b)
                    mean1 = torch.zeros_like(mean1_b)
                    std0 = torch.zeros_like(std0_b)
                    std1 = torch.zeros_like(std1_b)
                    outer = torch.zeros_like(outer_b)
                mean0 += mean0_b / n
                mean1 += mean1_b / n
                std0 += std0_b / n
                std1 += std1_b / n
                outer += outer_b / n

    cov = outer - torch.outer(mean0, mean1)
    if norm:
        corr = cov / (torch.outer(std0, std1) + 1e-4)
        return corr
    else:
        return cov


def subnet(model, n_layers):
    return model.features[:n_layers]


def get_layer_perm_from_corr(corr_mtx):
    corr_mtx = corr_mtx.cpu().numpy()
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
    corr_mtx = run_corr_matrix(subnet_a, subnet_b, loader)
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
