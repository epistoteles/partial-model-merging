import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from safetensors.torch import load_file
from src.utils import get_plots_dir, get_evaluations_dir


dataset = "SVHN"
architecture = "VGG"
bn = True

metrics = ["acc", "loss"]
sizes = [11] if architecture == "VGG" else [18]
widths = [1] if architecture == "VGG" else [1]
expansions = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

endpoint = torch.zeros(len(metrics), len(widths), len(sizes))
ensembling = torch.zeros(len(metrics), len(widths), len(sizes))
partial_merging = torch.zeros(len(metrics), len(widths), len(sizes), len(expansions))
partial_merging_REPAIR = torch.zeros(len(metrics), len(widths), len(sizes), len(expansions))
random_partial_merging = torch.zeros(len(metrics), len(widths), len(sizes), len(expansions))
random_partial_merging_REPAIR = torch.zeros(len(metrics), len(widths), len(sizes), len(expansions))
smallest_partial_merging = torch.zeros(len(metrics), len(widths), len(sizes), len(expansions))
smallest_partial_merging_REPAIR = torch.zeros(len(metrics), len(widths), len(sizes), len(expansions))

for i, size in enumerate(sizes):
    for j, width in enumerate(widths):
        dir = get_evaluations_dir(subdir="two_models")
        filename = f"{dir}/{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x-ab.safetensors"
        if os.path.exists(filename):
            metrics = load_file(filename)
            acc_a = metrics["ensembling_test_accs"][0].item()
            acc_b = metrics["ensembling_test_accs"][-1].item()
            acc_avg = (acc_a + acc_b) / 2
            acc_ensembling = metrics["ensembling_test_accs"][10].item()
            loss_a = metrics["ensembling_test_losses"][0].item()
            loss_b = metrics["ensembling_test_losses"][-1].item()
            loss_avg = (loss_a + loss_b) / 2
            loss_ensembling = metrics["ensembling_test_losses"][10].item()
            endpoint[0, j, i] = acc_avg
            ensembling[0, j, i] = acc_ensembling
            endpoint[1, j, i] = loss_avg
            ensembling[1, j, i] = loss_ensembling
            partial_merging[0, j, i, 0] = metrics["merging_test_accs"][10]
            partial_merging_REPAIR[0, j, i, 0] = metrics["merging_REPAIR_test_accs"][10]
            partial_merging[1, j, i, 0] = metrics["merging_test_losses"][10]
            partial_merging_REPAIR[1, j, i, 0] = metrics["merging_REPAIR_test_losses"][10]
            for idx, k in enumerate(expansions[1:]):
                partial_merging[0, j, i, idx + 1] = metrics[f"partial_merging_{k}_test_accs"][10]
                partial_merging_REPAIR[0, j, i, idx + 1] = metrics[f"partial_merging_REPAIR_{k}_test_accs"][10]
                partial_merging[1, j, i, idx + 1] = metrics[f"partial_merging_{k}_test_losses"][10]
                partial_merging_REPAIR[1, j, i, idx + 1] = metrics[f"partial_merging_REPAIR_{k}_test_losses"][10]
            dir = get_evaluations_dir(subdir="random-or-smallest")
            filename = (
                f"{dir}/random-or-smallest-{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x-ab.safetensors"
            )
            random_metrics = load_file(filename)
            for idx, k in enumerate(expansions[1:-1]):
                random_partial_merging[0, j, i, idx + 1] = random_metrics[
                    f"pull_apart_randomly_merging_{k:g}_test_accs"
                ][10]
                random_partial_merging_REPAIR[0, j, i, idx + 1] = random_metrics[
                    f"pull_apart_randomly_merging_REPAIR_{k:g}_test_accs"
                ][10]
                random_partial_merging[1, j, i, idx + 1] = random_metrics[
                    f"pull_apart_randomly_merging_{k:g}_test_losses"
                ][10]
                random_partial_merging_REPAIR[1, j, i, idx + 1] = random_metrics[
                    f"pull_apart_randomly_merging_REPAIR_{k:g}_test_losses"
                ][10]
            for idx, k in enumerate(expansions[1:-1]):
                smallest_partial_merging[0, j, i, idx + 1] = random_metrics[
                    f"pull_apart_smallest_merging_{k:g}_test_accs"
                ][10]
                smallest_partial_merging_REPAIR[0, j, i, idx + 1] = random_metrics[
                    f"pull_apart_smallest_merging_REPAIR_{k:g}_test_accs"
                ][10]
                smallest_partial_merging[1, j, i, idx + 1] = random_metrics[
                    f"pull_apart_smallest_merging_{k:g}_test_losses"
                ][10]
                smallest_partial_merging_REPAIR[1, j, i, idx + 1] = random_metrics[
                    f"pull_apart_smallest_merging_REPAIR_{k:g}_test_losses"
                ][10]
            random_partial_merging[:, j, i, 0] = partial_merging[:, j, i, 0]
            random_partial_merging[:, j, i, -1] = partial_merging[:, j, i, -1]
            random_partial_merging_REPAIR[:, j, i, 0] = partial_merging_REPAIR[:, j, i, 0]
            random_partial_merging_REPAIR[:, j, i, -1] = partial_merging_REPAIR[:, j, i, -1]
            smallest_partial_merging[:, j, i, 0] = partial_merging[:, j, i, 0]
            smallest_partial_merging[:, j, i, -1] = partial_merging[:, j, i, -1]
            smallest_partial_merging_REPAIR[:, j, i, 0] = partial_merging_REPAIR[:, j, i, 0]
            smallest_partial_merging_REPAIR[:, j, i, -1] = partial_merging_REPAIR[:, j, i, -1]

full_barrier_absolute = endpoint.unsqueeze(-1) - partial_merging[:, :, :, 0:1]

barrier_reduction_absolute = partial_merging - partial_merging[:, :, :, 0:1]
barrier_reduction_relative = barrier_reduction_absolute / full_barrier_absolute
barrier_reduction_absolute_REPAIR = partial_merging_REPAIR - partial_merging[:, :, :, 0:1]
barrier_reduction_relative_REPAIR = barrier_reduction_absolute_REPAIR / full_barrier_absolute

random_barrier_reduction_absolute = random_partial_merging - random_partial_merging[:, :, :, 0:1]
random_barrier_reduction_relative = random_barrier_reduction_absolute / full_barrier_absolute
random_barrier_reduction_absolute_REPAIR = random_partial_merging_REPAIR - random_partial_merging[:, :, :, 0:1]
random_barrier_reduction_relative_REPAIR = random_barrier_reduction_absolute_REPAIR / full_barrier_absolute

smallest_barrier_reduction_absolute = smallest_partial_merging - smallest_partial_merging[:, :, :, 0:1]
smallest_barrier_reduction_relative = smallest_barrier_reduction_absolute / full_barrier_absolute
smallest_barrier_reduction_absolute_REPAIR = smallest_partial_merging_REPAIR - smallest_partial_merging[:, :, :, 0:1]
smallest_barrier_reduction_relative_REPAIR = smallest_barrier_reduction_absolute_REPAIR / full_barrier_absolute

for m, metric in enumerate(["accuracy", "loss"]):
    plt.figure(figsize=(6, 6))
    plt.xlabel("added width (%)")
    plt.ylabel(f"{metric} barrier reduction (%)")
    plt.xticks(torch.linspace(0, 100, 11))
    plt.title(
        f"{metric.title()} barrier reduction w.r.t. added width\n{dataset}, {architecture+str(sizes[0])}, 1×width"
    )

    # AUC diagonal
    sns.lineplot(x=torch.linspace(0, 100, 11), y=torch.linspace(0, 100, 11), color="grey")

    # 100% horizontal line
    sns.lineplot(x=torch.linspace(0, 100, 11), y=[100] * 11, color="grey")

    # baseline forced assignment
    sns.lineplot(
        x=torch.linspace(0, 100, 11),
        y=barrier_reduction_relative[m][0][0] * 100,
        label="relaxed LAP (forced)",
        color="red",
        marker="o",
        markersize=4,
    )
    sns.lineplot(
        x=torch.linspace(0, 100, 11),
        y=barrier_reduction_relative_REPAIR[m][0][0] * 100,
        dashes=(2, 2),
        color="red",
        marker="o",
        markersize=4,
    )

    # random neurons
    sns.lineplot(
        x=torch.linspace(0, 100, 11),
        y=random_barrier_reduction_relative[m][0][0] * 100,
        label="randomly selected units (forced)",
        color="orange",
        marker="o",
        markersize=4,
    )
    sns.lineplot(
        x=torch.linspace(0, 100, 11),
        y=random_barrier_reduction_relative_REPAIR[m][0][0] * 100,
        dashes=(2, 2),
        color="orange",
        marker="o",
        markersize=4,
    )

    # # smallest corrs
    # sns.lineplot(
    #     x=torch.linspace(0, 100, 11),
    #     y=smallest_barrier_reduction_relative[m][0][0] * 100,
    #     label="smallest corr units (forced)",
    #     color="green",
    #     marker="o",
    #     markersize=4,
    # )
    # sns.lineplot(
    #     x=torch.linspace(0, 100, 11),
    #     y=smallest_barrier_reduction_relative_REPAIR[m][0][0] * 100,
    #     dashes=(2, 2),
    #     color="green",
    #     marker="o",
    #     markersize=4,
    # )

    # just for the REPAIR label
    sns.lineplot(x=[0, 0], y=[0, 0], dashes=(2, 2), label="with REPAIR", color="grey")

    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(
        os.path.join(
            plots_dir,
            f"random_vs_lap_{metric}_AUC_{dataset}_{architecture}.png",
        ),
        dpi=600,
    )
    plt.close()
    print(f"📊 {metric} AUC {dataset} {architecture} (w.r.t. width) plot saved")
