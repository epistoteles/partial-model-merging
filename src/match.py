from torch import nn
from models.VGG import VGG
from utils.data_utils import get_loaders_CIFAR10, load_model
from utils.matching_utils import subnet, run_corr_matrix
import scipy


model_a = VGG(11)
model_b = VGG(11)
model_a = load_model(model_a, "VGG11-1x-a.pt").cuda()
model_b = load_model(model_b, "VGG11-1x-a.pt").cuda()

train_aug_loader, _, _ = get_loaders_CIFAR10()

corrs = []
lap_solutions = []
best_corrs = []
feats_b = model_b.features
n = len(feats_b)
for i in range(n):
    layer = feats_b[i]
    if isinstance(layer, nn.Conv2d):
        # get permutation and permute output of conv and maybe bn
        if isinstance(feats_b[i + 1], nn.BatchNorm2d):
            assert isinstance(feats_b[i + 2], nn.ReLU)
            corr = run_corr_matrix(subnet(model_a, i + 3), subnet(model_b, i + 3), train_aug_loader).cpu().numpy()
            lap_solution = scipy.optimize.linear_sum_assignment(corr, maximize=True)
            best_corr = corr[lap_solution]
            corrs += [corr]
            lap_solutions += [lap_solution]
            best_corrs += [best_corr]
        else:
            assert isinstance(feats_b[i + 1], nn.ReLU)
            corr = run_corr_matrix(subnet(model_a, i + 2), subnet(model_b, i + 2), train_aug_loader).cpu().numpy()
            lap_solution = scipy.optimize.linear_sum_assignment(corr, maximize=True)
            best_corr = corr[lap_solution]
            corrs += [corr]
            lap_solutions += [lap_solution]
            best_corrs += [best_corr]
        # look for succeeding layer to permute input
        next_layer = None
        for j in range(i + 1, n):
            if isinstance(feats_b[j], nn.Conv2d):
                next_layer = feats_b[j]
                break
        if next_layer is None:
            next_layer = model_b.classifier
