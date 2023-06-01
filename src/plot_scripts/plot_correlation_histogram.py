import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import scipy
import os
from pathlib import Path

from src.utils.utils import (
    load_model,
    get_corr_matrix,
    subnet,
    get_layer_perm_from_corr,
    permute_output,
    permute_input,
    get_plots_dir,
    parse_model_name,
    get_loaders,
    get_all_model_names,
)


def plot_model_filters(model_name_a, model_name_b):
    dataset_a, model_type_a, size_a, width_b, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, width_a, variant_b = parse_model_name(model_name_b)

    model_a = load_model(model_name_a).cuda()
    model_b = load_model(model_name_b).cuda()

    train_aug_loader, train_noaug_loader, _ = get_loaders(dataset_a)

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
                corr = get_corr_matrix(subnet(model_a, i + 3), subnet(model_b, i + 3), train_noaug_loader).cpu().numpy()
                lap_solution = scipy.optimize.linear_sum_assignment(corr, maximize=True)
                perm_map = get_layer_perm_from_corr(corr)
                permute_output(perm_map, feats_b[i], feats_b[i + 1])
                best_corr = corr[lap_solution]
                corrs += [corr]
                lap_solutions += [lap_solution]
                best_corrs += [best_corr]
            else:
                assert isinstance(feats_b[i + 1], nn.ReLU)
                corr = get_corr_matrix(subnet(model_a, i + 2), subnet(model_b, i + 2), train_noaug_loader).cpu().numpy()
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

    fig, axes = plt.subplots(1, len(best_corrs), sharex=True, sharey=True, figsize=(1.2 * len(best_corrs), 6))
    fig.suptitle(
        f"Histogram of correlations selected by LAP solver, per conv. layer,\n"
        f"{dataset_a}, {model_type_a}{size_a}, {width_a}×width, model {variant_a} + {variant_b}",
    )

    for i in range(len(best_corrs)):
        axes[i].axhline(y=0, color="black", linewidth=1)
        sns.histplot(y=best_corrs[i], ax=axes[i], binrange=(-1, 1), bins=40)
        axes[i].set_ylim(-1.02, 1.02)
        axes[i].set_title(f"Conv. {i+1}", size=12)
        if i == 0:
            axes[i].set_yticks([-1, -0.5, 0, 0.5, 1])
            axes[i].set_ylabel("Pearson’s r")
        else:
            axes[i].set_yticks([])

    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name_a[:-2]}.png"), dpi=600)
    print(f"📊 Plot saved for {model_name_a}, {model_name_b}")


for model_name_a, model_name_b in [("CIFAR10-VGG11-1x-a", "CIFAR10-VGG11-1x-b")]:
    plot_model_filters(model_name_a, model_name_b)
