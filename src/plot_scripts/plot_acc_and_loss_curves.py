import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from src.utils import load_model, normalize, get_plots_dir, parse_model_name, get_all_model_names
from src.evaluate import evaluate_two_models


def plot_acc_and_loss_curves(model_name_a: str, model_name_b: str = None):
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    metrics = evaluate_two_models(model_name_a, model_name_b)

    for metric, split in product(["accs", "losses"], ["train", "test"]):
        plt.figure(figsize=(12, 8))
        plt.xlabel("alpha")
        plt.ylabel(f"{split} {metric}")
        plt.title(f"{dataset_a}, {model_type_a}{size_a}, {width_a}×width, model {variant_a} vs. {variant_b}")

        if metric == "losses":
            sns.lineplot(
                x=torch.Tensor([0, 1]),
                y=torch.Tensor(
                    [metrics[f"ensembling_{split}_{metric}"][0], metrics[f"ensembling_{split}_{metric}"][-1]]
                ),
                label="zero loss barrier",
                color="grey",
                dashes=(2, 2),
            )

        sns.lineplot(x=metrics["alphas"], y=metrics[f"ensembling_{split}_{metric}"], label="ensembling", color="black")
        # sns.lineplot(x=metrics["alphas"], y=metrics[f"naive_{split}_{metric}"], label="naive merging", color='')
        sns.lineplot(x=metrics["alphas"], y=metrics[f"merging_{split}_{metric}"], label="merging", color="orange")

        plots_dir = get_plots_dir(subdir=Path(__file__).stem)
        plt.savefig(
            os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name_a}{variant_b}_{split}_{metric}_1.png"), dpi=600
        )

        if f"partial_merging_1.1_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"partial_merging_1.1_{split}_{metric}"],
                label="partial merging (+10% buffer)",
                color="red",
            )
        if f"partial_merging_1.5_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"partial_merging_1.5_{split}_{metric}"],
                label="partial merging (+50% buffer)",
                color="blue",
            )
        if f"partial_merging_1.8_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"partial_merging_1.8_{split}_{metric}"],
                label="partial merging (+80% buffer)",
                color="green",
            )
        if f"partial_merging_2.0_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"partial_merging_2.0_{split}_{metric}"],
                label="partial merging (+100% buffer)",
                color="purple",
            )

        plt.savefig(
            os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name_a}{variant_b}_{split}_{metric}_2.png"), dpi=600
        )

        if f"merging_REPAIR_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"merging_REPAIR_{split}_{metric}"],
                label="merging + REPAIR ",
                color="orange",
                dashes=(2, 2),
            )
        if f"partial_merging_REPAIR_1.1_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"partial_merging_REPAIR_1.1_{split}_{metric}"],
                label="partial merging + REPAIR (+10% buffer)",
                color="red",
                dashes=(2, 2),
            )
        if f"partial_merging_REPAIR_1.5_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"partial_merging_REPAIR_1.5_{split}_{metric}"],
                label="partial merging + REPAIR (+50% buffer)",
                color="blue",
                dashes=(2, 2),
            )
        if f"partial_merging_REPAIR_1.8_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"partial_merging_REPAIR_1.8_{split}_{metric}"],
                label="partial merging + REPAIR (+80% buffer)",
                color="green",
                dashes=(2, 2),
            )
        if f"partial_merging_REPAIR_2.0_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"partial_merging_REPAIR_2.0_{split}_{metric}"],
                label="partial merging + REPAIR (+100% buffer)",
                color="purple",
                dashes=(2, 2),
            )

        plt.savefig(
            os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name_a}{variant_b}_{split}_{metric}_3.png"), dpi=600
        )
        plt.close()
        print(f"📊 {split} {metric} plot saved for {model_name_a}, {model_name_b}")
