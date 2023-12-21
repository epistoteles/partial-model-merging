import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from safetensors.torch import load_file
from src.utils import get_plots_dir, get_evaluations_dir

dataset = "CIFAR10"
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

num_layers = 8 if architecture == "VGG" else 17

smart_order_before_merging = torch.zeros(len(metrics), len(widths), len(sizes), num_layers + 1)
smart_order_before_merging_REPAIR = torch.zeros(len(metrics), len(widths), len(sizes), num_layers + 1)
smart_order_before_param_increase = torch.ones(len(widths), len(sizes), num_layers + 1)

smart_order_after_merging = torch.zeros(len(metrics), len(widths), len(sizes), num_layers + 1)
smart_order_after_merging_REPAIR = torch.zeros(len(metrics), len(widths), len(sizes), num_layers + 1)
smart_order_after_param_increase = torch.ones(len(widths), len(sizes), num_layers + 1)

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

            smart_order_before_merging[0, j, i, 0] = metrics["merging_test_accs"][10]
            smart_order_before_merging_REPAIR[0, j, i, 0] = metrics["merging_REPAIR_test_accs"][10]
            smart_order_before_merging[1, j, i, 0] = metrics["merging_test_losses"][10]
            smart_order_before_merging_REPAIR[1, j, i, 0] = metrics["merging_REPAIR_test_losses"][10]

            dir = get_evaluations_dir(subdir="smart_order")
            filename = (
                f"{dir}/smart_order_before-{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x-ab.safetensors"
            )
            smart_order_before_metrics = load_file(filename)
            for k in range(num_layers):
                smart_order_before_merging[0, j, i, k + 1] = smart_order_before_metrics[
                    f"smart_order_merging_{k:g}_test_accs"
                ][10]
                smart_order_before_merging_REPAIR[0, j, i, k + 1] = smart_order_before_metrics[
                    f"smart_order_merging_REPAIR_{k:g}_test_accs"
                ][10]
                smart_order_before_merging[1, j, i, k + 1] = smart_order_before_metrics[
                    f"smart_order_merging_{k:g}_test_losses"
                ][10]
                smart_order_before_merging_REPAIR[1, j, i, k + 1] = smart_order_before_metrics[
                    f"smart_order_merging_REPAIR_{k:g}_test_losses"
                ][10]
                smart_order_before_param_increase[j, i, k + 1] = smart_order_before_metrics[
                    f"smart_order_merging_{k:g}_param_increase"
                ]

            smart_order_after_merging[0, j, i, 0] = metrics["merging_test_accs"][10]
            smart_order_after_merging_REPAIR[0, j, i, 0] = metrics["merging_REPAIR_test_accs"][10]
            smart_order_after_merging[1, j, i, 0] = metrics["merging_test_losses"][10]
            smart_order_after_merging_REPAIR[1, j, i, 0] = metrics["merging_REPAIR_test_losses"][10]

            filename = (
                f"{dir}/smart_order_after-{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x-ab.safetensors"
            )
            smart_order_after_metrics = load_file(filename)
            for k in range(num_layers):
                smart_order_after_merging[0, j, i, k + 1] = smart_order_after_metrics[
                    f"smart_order_merging_{k:g}_test_accs"
                ][10]
                smart_order_after_merging_REPAIR[0, j, i, k + 1] = smart_order_after_metrics[
                    f"smart_order_merging_REPAIR_{k:g}_test_accs"
                ][10]
                smart_order_after_merging[1, j, i, k + 1] = smart_order_after_metrics[
                    f"smart_order_merging_{k:g}_test_losses"
                ][10]
                smart_order_after_merging_REPAIR[1, j, i, k + 1] = smart_order_after_metrics[
                    f"smart_order_merging_REPAIR_{k:g}_test_losses"
                ][10]
                smart_order_after_param_increase[j, i, k + 1] = smart_order_after_metrics[
                    f"smart_order_merging_{k:g}_param_increase"
                ]

full_barrier_absolute = endpoint.unsqueeze(-1) - partial_merging[:, :, :, 0:1]

barrier_reduction_absolute = partial_merging - partial_merging[:, :, :, 0:1]
barrier_reduction_relative = barrier_reduction_absolute / full_barrier_absolute
barrier_reduction_absolute_REPAIR = partial_merging_REPAIR - partial_merging[:, :, :, 0:1]
barrier_reduction_relative_REPAIR = barrier_reduction_absolute_REPAIR / full_barrier_absolute

smart_order_before_barrier_reduction_absolute = smart_order_before_merging - smart_order_before_merging[:, :, :, 0:1]
smart_order_before_barrier_reduction_relative = smart_order_before_barrier_reduction_absolute / full_barrier_absolute
smart_order_before_barrier_reduction_absolute_REPAIR = (
    smart_order_before_merging_REPAIR - smart_order_before_merging[:, :, :, 0:1]
)
smart_order_before_barrier_reduction_relative_REPAIR = (
    smart_order_before_barrier_reduction_absolute_REPAIR / full_barrier_absolute
)

smart_order_after_barrier_reduction_absolute = smart_order_after_merging - smart_order_after_merging[:, :, :, 0:1]
smart_order_after_barrier_reduction_relative = smart_order_after_barrier_reduction_absolute / full_barrier_absolute
smart_order_after_barrier_reduction_absolute_REPAIR = (
    smart_order_after_merging_REPAIR - smart_order_after_merging[:, :, :, 0:1]
)
smart_order_after_barrier_reduction_relative_REPAIR = (
    smart_order_after_barrier_reduction_absolute_REPAIR / full_barrier_absolute
)

for m, metric in enumerate(["accuracy", "loss"]):
    plt.figure(figsize=(6, 6))
    plt.xlabel("added parameters (%)")
    plt.ylabel(f"{metric} barrier reduction (%)")
    plt.xticks(torch.linspace(0, 100, 11))
    plt.title(
        f"{metric.title()} barrier reduction w.r.t. added parameters\n{dataset}, {architecture + str(sizes[0])}, 1×width"
    )

    # AUC diagonal
    sns.lineplot(x=torch.linspace(0, 100, 11), y=torch.linspace(0, 100, 11), color="grey")

    # 100% horizontal line
    sns.lineplot(x=torch.linspace(0, 100, 11), y=[100] * 11, color="grey")

    param_increase = (
        (torch.Tensor([1, 1.1897, 1.3587, 1.5107, 1.6398, 1.74, 1.8396, 1.9093, 1.960, 1.9899, 1.9999]) - 1)
        if architecture == "VGG"
        else (torch.Tensor([1, 1.1886, 1.3616, 1.5053, 1.6381, 1.7477, 1.8375, 1.9092, 1.9577, 1.9892, 2.0000]) - 1)
    )
    param_increase *= 100

    # baseline forced assignment
    sns.lineplot(
        x=param_increase,
        y=barrier_reduction_relative[m][0][0] * 100,
        label="forced buffer assignment",
        color="red",
        marker="o",
        markersize=4,
    )
    sns.lineplot(
        x=param_increase,
        y=barrier_reduction_relative_REPAIR[m][0][0] * 100,
        dashes=(2, 2),
        color="red",
        marker="o",
        markersize=4,
    )

    # smart order before REPAIR
    sns.lineplot(
        x=(smart_order_before_param_increase[0][0] - 1) * 100,
        y=smart_order_before_barrier_reduction_relative[m][0][0] * 100,
        label="smart order before",
        color="orange",
        marker="o",
        markersize=4,
    )
    sns.lineplot(
        x=(smart_order_before_param_increase[0][0] - 1) * 100,
        y=smart_order_before_barrier_reduction_relative_REPAIR[m][0][0] * 100,
        dashes=(2, 2),
        color="orange",
        marker="o",
        markersize=4,
    )

    # smart order after REPAIR
    sns.lineplot(
        x=(smart_order_after_param_increase[0][0] - 1) * 100,
        y=smart_order_after_barrier_reduction_relative[m][0][0] * 100,
        label="smart order after",
        color="blue",
        marker="o",
        markersize=4,
    )
    sns.lineplot(
        x=(smart_order_after_param_increase[0][0] - 1) * 100,
        y=smart_order_after_barrier_reduction_relative_REPAIR[m][0][0] * 100,
        dashes=(2, 2),
        color="blue",
        marker="o",
        markersize=4,
    )

    # just for the REPAIR label
    sns.lineplot(x=[0, 0], y=[0, 0], dashes=(2, 2), label="with REPAIR", color="grey")

    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(
        os.path.join(
            plots_dir,
            f"smart_order_{metric}_AUC_{dataset}_{architecture}.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()
    print(f"📊 {metric} AUC {dataset} {architecture} (w.r.t. params) plot saved")
