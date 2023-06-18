from torch import nn
from src.utils import (
    get_loaders,
    load_model,
    subnet,
    get_corr_matrix,
    permute_input,
    permute_output,
    get_layer_perm_from_corr,
)
import scipy
import plotext
import numpy as np


model_a = load_model("VGG11-1x-a.pt").cuda()
model_b = load_model("VGG11-1x-b.pt").cuda()

train_aug_loader, train_noaug_loader, _ = get_loaders("CIFAR10")

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
            corr = get_corr_matrix(subnet(model_a, i + 3), subnet(model_b, i + 3), train_aug_loader).cpu().numpy()
            lap_solution = scipy.optimize.linear_sum_assignment(corr, maximize=True)
            perm_map = get_layer_perm_from_corr(corr)
            permute_output(perm_map, feats_b[i], feats_b[i + 1])
            best_corr = corr[lap_solution]
            corrs += [corr]
            lap_solutions += [lap_solution]
            best_corrs += [best_corr]
        else:
            assert isinstance(feats_b[i + 1], nn.ReLU)
            corr = get_corr_matrix(subnet(model_a, i + 2), subnet(model_b, i + 2), train_aug_loader).cpu().numpy()
            lap_solution = scipy.optimize.linear_sum_assignment(corr, maximize=True)
            perm_map = get_layer_perm_from_corr(corr)
            permute_output(perm_map, feats_b[i], None)
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
        permute_input(perm_map, next_layer)


# plot the histograms in terminal
plotext.subplots(1, len(best_corrs))
plotext.title("Histogram of correlation coefficients of feature maps")
thresholds = np.linspace(-1, 1, 41)
labels = [f">{x:.2f}" for x in thresholds[:-1]]
for i, (best_corr, corr) in enumerate(zip(best_corrs, corrs)):
    plotext.subplot(1, i + 1).plotsize(plotext.tw() // len(best_corrs), None)
    plotext.subplot(1, i + 1).title(f"Conv2d #{i+1}")

    histogram_counts = np.array(
        [
            sum(1 for x in corr.flatten() if thresholds[idx] <= x < thresholds[idx + 1])
            for idx in range(len(thresholds) - 1)
        ]
    )
    histogram_counts = histogram_counts / histogram_counts.sum()
    plotext.bar(labels, histogram_counts, orientation="h", label="all", color="blue")

    histogram_counts = np.array(
        [sum(1 for x in best_corr if thresholds[idx] <= x < thresholds[idx + 1]) for idx in range(len(thresholds) - 1)]
    )
    histogram_counts = histogram_counts / histogram_counts.sum()
    plotext.bar(labels, histogram_counts, orientation="h", label="selected by LAP", color="blue+")
plotext.show()

for corr in corrs:
    print(f"Max: {corr.max()}")
    print(f"Avg: {corr.mean()}")
    print(f"Min: {corr.min()}")
    print("")
