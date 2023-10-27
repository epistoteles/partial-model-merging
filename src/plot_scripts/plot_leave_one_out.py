import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from src.utils import load_model, normalize, get_plots_dir, parse_model_name, get_all_model_names
from src.evaluate import experiment_a, evaluate_single_model


def plot_leave_one_out(model_name_a: str, model_name_b: str):
    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    metrics_a = evaluate_single_model(model_name_a)
    metrics_b = evaluate_single_model(model_name_b)
    metrics = experiment_a(model_name_a, model_name_b)

    for i, (split, metric) in enumerate(product(["train", "test"], ["accs", "losses"])):
        for repair in ["", "_REPAIR"]:
            plt.figure(figsize=(12, 8))
            plt.xlabel("layer")
            plt.ylabel(f"{split} {metric}")
            plt.title(
                f"{dataset_a}, {model_type_a}{size_a}, bn={batch_norm_a}, {width_a}×width, leave-one-out experiment"
            )

            sns.lineplot(
                x=metrics["layers"],
                y=metrics[f"full_ensembling_{split}_{metric}"],
                label="full ensembling",
                color="blue",
            )
            sns.lineplot(
                x=metrics["layers"],
                y=metrics[f"full_merging{repair}_{split}_{metric}"],
                label="full merging",
                color="orange",
            )

            sns.lineplot(x=metrics["layers"], y=metrics_a[i], label="baseline model a", color="grey")
            sns.lineplot(x=metrics["layers"], y=metrics_b[i], label="baseline model b", color="grey")

            sns.lineplot(
                x=metrics["layers"],
                y=metrics[f"only_ensemble_i{repair}_{split}_{metric}"],
                label="only ensemble layer i",
                color="red",
            )
            sns.lineplot(
                x=metrics["layers"],
                y=metrics[f"only_merge_i{repair}_{split}_{metric}"],
                label="only merge layer i",
                color="green",
            )

            plots_dir = get_plots_dir(subdir=Path(__file__).stem)
            plt.savefig(
                os.path.join(
                    plots_dir, f"{Path(__file__).stem}_{model_name_a}{variant_b}_{split}_{metric}{repair}.png"
                ),
                dpi=600,
            )
            plt.close()
            print(f"📊 {split} {metric} leave-one-out plot saved for {model_name_a}, {model_name_b}")
