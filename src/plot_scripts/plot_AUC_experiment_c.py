import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from safetensors.torch import load_file
from src.utils import get_plots_dir, get_evaluations_dir, get_num_params, model_like


def get_metrics(dataset, architecture, size, bn, width):
    dir = get_evaluations_dir(subdir="two_models")
    filename = f"{dir}/{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x-ab.safetensors"
    if os.path.exists(filename):
        metrics = load_file(filename)
        d = {}
        d["acc_avg"] = metrics["ensembling_test_accs"][[0, -1]].mean().item()
        d["acc_ensembling"] = metrics["ensembling_test_accs"][10].item()
        d["acc_merging"] = metrics["merging_test_accs"][10].item()
        d["acc_merging_REPAIR"] = metrics["merging_REPAIR_test_accs"][10].item()
        d["loss_avg"] = metrics["ensembling_test_losses"][[0, -1]].mean().item()
        d["loss_ensembling"] = metrics["ensembling_test_losses"][10].item()
        d["loss_merging"] = metrics["merging_test_losses"][10].item()
        d["loss_merging_REPAIR"] = metrics["merging_REPAIR_test_losses"][10].item()
        d["acc_expansions"] = torch.ones(11) * d["acc_merging"]
        d["loss_expansions"] = torch.ones(11) * d["loss_merging"]
        d["acc_expansions_REPAIR"] = torch.ones(11) * d["acc_merging_REPAIR"]
        d["loss_expansions_REPAIR"] = torch.ones(11) * d["loss_merging_REPAIR"]
        d["param_increase_expansions"] = (
            torch.Tensor([1, 1.1897, 1.3587, 1.5107, 1.6398, 1.74, 1.8396, 1.9093, 1.960, 1.9899, 1.9999]) - 1
        )
        for idx, k in enumerate([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]):
            d["acc_expansions"][idx + 1] = metrics[f"partial_merging_{k}_test_accs"][10]
            d["loss_expansions"][idx + 1] = metrics[f"partial_merging_{k}_test_losses"][10]
            d["acc_expansions_REPAIR"][idx + 1] = metrics[f"partial_merging_REPAIR_{k}_test_accs"][10]
            d["loss_expansions_REPAIR"][idx + 1] = metrics[f"partial_merging_REPAIR_{k}_test_losses"][10]
    else:
        raise ValueError(f"File does not exist: {filename}")
    return d


metrics = ["acc", "loss"]
datasets = ["CIFAR10", "SVHN"]
architecture = "ResNet"
size = 18
bn = True
width = 4

for m, metric in enumerate(metrics):
    for dataset in datasets:
        dir = get_evaluations_dir(subdir="experiment_c")
        filename = f"{dir}/experiment-c-{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x-ab.safetensors"
        metrics = load_file(filename)
        metrics_dict = get_metrics(dataset, architecture, size, bn, width)

        test = metrics[f"only_expand_layer_i_test_{metric}{'e' if metric == 'loss' else ''}s"][:, :8]
        test_REPAIR = metrics[f"only_expand_layer_i_REPAIR_test_{metric}{'e' if metric == 'loss' else ''}s"][:, :8]

        full_barrier_absolute = metrics_dict[f"{metric}_avg"] - metrics_dict[f"{metric}_merging"]
        barrier_reduction_absolute = test - metrics_dict[f"{metric}_merging"]
        barrier_reduction_relative = barrier_reduction_absolute / full_barrier_absolute

        barrier_reduction_absolute_REPAIR = test_REPAIR - metrics_dict[f"{metric}_merging"]
        barrier_reduction_relative_REPAIR = barrier_reduction_absolute_REPAIR / full_barrier_absolute
        first_value_REPAIR = torch.Tensor(
            [(metrics_dict[f"{metric}_merging_REPAIR"] - metrics_dict[f"{metric}_merging"]) / full_barrier_absolute]
        )

        param_increase = (
            metrics["only_expand_layer_i_num_params"][:, :8]
            / get_num_params(model_like(f"{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x-a"))
        ) - 1

        plt.figure(figsize=(6, 6))
        plt.xlabel("added parameters (%)")
        plt.ylabel(f"{metric} barrier reduction (%)")
        plt.xticks(torch.linspace(0, 100, 11))
        plt.title(
            f"{metric.title()} barrier reduction w.r.t. added parameters\n{dataset}, {architecture}{size}, {width}×width, bn={bn}"
        )

        # AUC diagonal
        sns.lineplot(x=torch.linspace(0, 100, 11), y=torch.linspace(0, 100, 11), color="grey")

        # 100% horizontal line
        sns.lineplot(x=torch.linspace(0, 100, 11), y=[100] * 11, color="grey")

        sns.lineplot(
            x=torch.cat((torch.Tensor([0]), param_increase.T.flatten() * 100)),
            y=torch.cat((torch.Tensor([0]), barrier_reduction_relative.T.flatten() * 100)),
            label="left to right",
            color="red",
            marker="o",
            markersize=4,
        )
        sns.lineplot(
            x=torch.cat((torch.Tensor([0]), param_increase.T.flatten() * 100)),
            y=torch.cat((first_value_REPAIR * 100, barrier_reduction_relative_REPAIR.T.flatten() * 100)),
            dashes=(2, 2),
            color="red",
            marker="o",
            markersize=4,
        )
        sns.lineplot(
            x=metrics_dict["param_increase_expansions"] * 100,
            y=100
            - (metrics_dict[f"{metric}_avg"] - metrics_dict[f"{metric}_expansions"]) / full_barrier_absolute * 100,
            label="all layers",
            color="orange",
            marker="o",
            markersize=4,
        )
        sns.lineplot(
            x=metrics_dict["param_increase_expansions"] * 100,
            y=100
            - (metrics_dict[f"{metric}_avg"] - metrics_dict[f"{metric}_expansions_REPAIR"])
            / full_barrier_absolute
            * 100,
            dashes=(2, 2),
            color="orange",
            marker="o",
            markersize=4,
        )

        # just for the REPAIR label
        sns.lineplot(x=[0, 0], y=[0, 0], dashes=(2, 2), label="with REPAIR", color="grey")

        plots_dir = get_plots_dir(subdir=Path(__file__).stem)
        plt.savefig(
            os.path.join(
                plots_dir,
                f"{metric}_AUC_{dataset}_{architecture}.png",
            ),
            dpi=600,
        )
        plt.close()
        print(f"📊 {metric} AUC {dataset} {architecture} plot saved")
