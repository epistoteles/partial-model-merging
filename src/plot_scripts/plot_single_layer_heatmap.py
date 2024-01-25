import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from safetensors.torch import load_file
import matplotlib.transforms as transforms

from src.utils import get_evaluations_dir, get_plots_dir
import os


dataset = "CIFAR10E"
architecture = "ResNet"
size = 11 if architecture == "VGG" else 18
bn = True
width = 1
variants = 'ef'
absolute = True

for repair in [True, False]:
    eval_dir = get_evaluations_dir()
    metrics = load_file(
        eval_dir + f"experiment_b/experiment-b-{dataset}-{architecture}{size}-bn-{width}x-{variants}.safetensors"
    )
    metrics_default = load_file(
        eval_dir + f"two_models/{dataset}-{architecture}{size}-bn-{width}x-{variants}.safetensors"
    )

    accs = torch.flip(metrics[f"only_expand_layer_i{'_REPAIR' if repair else ''}_test_accs"], dims=[0])[:, :12]
    losses = torch.flip(metrics[f"only_expand_layer_i{'_REPAIR' if repair else ''}_test_losses"], dims=[0])[:, :12]
    params = (torch.flip(metrics["only_expand_layer_i_num_params"][:, :12], dims=[0]) / metrics["default_num_params"])[:, :12] - 1

    acc_endpoints = metrics_default["ensembling_test_accs"][[0, -1]].mean()
    acc_merging = metrics_default[f"merging{'_REPAIR' if repair else ''}_test_accs"][10]
    acc_barrier_reduction = (accs - acc_merging) / (acc_endpoints - acc_merging)

    loss_endpoints = metrics_default["ensembling_test_losses"][[0, -1]].mean()
    loss_merging = metrics_default[f"merging{'_REPAIR' if repair else ''}_test_losses"][10]
    loss_barrier_reduction = (losses - loss_merging) / (loss_endpoints - loss_merging)

    if absolute:
        acc_barrier_reduction = accs
        loss_barrier_reduction = losses / 100

    # Creating subplots
    if architecture == "ResNet":
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.3))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    plt.suptitle(
        f"Expanding individual layers\n{architecture}{size}, {dataset}, {width}×width, {'after' if repair else 'before'} REPAIR"
    )

    if absolute:
        axes[0].set_title("Test accuracy barrier reduction (%)")
        axes[1].set_title("Test loss barrier reduction (%)")
    else:
        axes[0].set_title("Test accuracy")
        axes[1].set_title("Test loss")
    axes[2].set_title("Parameter increase (%)")

    # Plotting the heatmaps
    sns.heatmap(acc_barrier_reduction * 100, ax=axes[0], cbar=False, annot=True, fmt=".1f" if acc_barrier_reduction.max() < 10 else ".0f")
    sns.heatmap(loss_barrier_reduction * 100, ax=axes[1], cbar=False, annot=True, fmt=".2f" if loss_barrier_reduction.max() < 10 else ".0f")
    sns.heatmap(params * 100, ax=axes[2], cbar=False, annot=True, fmt=".1f")

    # Setting up the axes
    # Show y-axis only on the left subplot
    axes[0].set_ylabel("gamma")
    axes[0].set_yticklabels(reversed([f"{x:.1f}" for x in torch.linspace(0.1, 1, 10)]))
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Show x-axis on all subplots
    for ax in axes:
        ax.set_xlabel("layer")
        if architecture == "VGG":
            ax.set_xticklabels(range(1, len(accs[0]) + 1))
            plt.subplots_adjust(wspace=0.1)
        else:
            ax.set_xticklabels(list(range(2, 17, 2)) + ["{1,3,5}", "{7,9}", "{11,13}", "{15,17}"], rotation=-45, ha="left")
            plt.subplots_adjust(wspace=-1)

    plt.tight_layout()
    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(
        os.path.join(
            plots_dir,
            f"per_layer_{dataset}_{architecture}{'_REPAIR' if repair else ''}_{variants}.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()
    print(f"📊 per layer {dataset} {architecture} {repair=} plot saved")
