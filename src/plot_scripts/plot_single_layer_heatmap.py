import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

# from src.utils import get_plots_dir
import os


dataset = "SVHN"
architecture = "VGG"
bn = True

metrics = ["acc", "loss"]

for metric in metrics:

    tensor1 = torch.randn(10, 8) * 10
    tensor2 = torch.randn(10, 8) * 50
    tensor3 = torch.randn(10, 8) * 100

    # Creating subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Plotting the heatmaps
    sns.heatmap(tensor1, ax=axes[0], cbar=False)
    sns.heatmap(tensor2, ax=axes[1], cbar=False)
    sns.heatmap(tensor3, ax=axes[2], cbar=False)

    # Setting up the axes
    # Show y-axis only on the left subplot
    axes[0].set_ylabel("gamma")
    for ax in axes[1:]:
        ax.set_ylabel("")

    # Show x-axis on all subplots
    for ax in axes:
        ax.set_xlabel("layer")

    plt.tight_layout()
    plots_dir = "/home/korbinian/Documents/Masterarbeit/partial-model-merging/plots/plot_single_layer_heatmap/"
    # plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(
        os.path.join(
            plots_dir,
            f"per_layer_{metric}_{dataset}_{architecture}.png",
        ),
        dpi=600,
    )
    plt.close()
    print(f"📊 per layer {metric} {dataset} {architecture} plot saved")
