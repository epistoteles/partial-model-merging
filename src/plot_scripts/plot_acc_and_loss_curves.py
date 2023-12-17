import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch
from safetensors.torch import load_file

from src.utils import load_model, normalize, get_plots_dir, parse_model_name, get_all_model_names, get_evaluations_dir
from src.evaluate import evaluate_two_models


def plot_acc_and_loss_curves(model_name_a: str, model_name_b: str = None):
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    evaluations_dir = get_evaluations_dir(subdir="two_models")
    filepath = os.path.join(evaluations_dir, f"{model_name_a}{variant_b}.safetensors")
    if os.path.exists(filepath):
        metrics = load_file(filepath)
    else:
        metrics = evaluate_two_models(model_name_a, model_name_b)

    for metric, split in product(["accs", "losses"], ["train", "test"]):
        plt.figure(figsize=(12, 8))
        plt.xlabel("alpha")
        plt.ylabel(f"{split} {metric}")
        plt.title(
            f"{dataset_a}, {model_type_a}{size_a},{' bn,' if batch_norm_a else ''} {width_a}×width, model {variant_a} vs. {variant_b}"
        )

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
        sns.lineplot(
            x=metrics["alphas"], y=metrics[f"merging_{split}_{metric}"], label="full merging", color=plt.cm.rainbow(0)
        )

        ax = plt.gca()
        ax.set_facecolor("#ffebeb" if metric == "acc" else "#f5ffeb")
        m_1 = metrics[f"ensembling_{split}_{metric}"][0]
        m_2 = metrics[f"ensembling_{split}_{metric}"][-1]
        m_diff = m_1 - m_2
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        vertices = [
            (xlim[0], m_1 - m_diff * xlim[0]),
            (xlim[1], m_1 - m_diff * xlim[0]),
            (xlim[1], ylim[1]),
            (xlim[0], ylim[1]),
        ]
        polygon = patches.Polygon(
            vertices, closed=True, facecolor="#f5ffeb" if metric == "acc" else "#ffebeb", edgecolor="none", zorder=-1
        )
        ax.add_patch(polygon)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plots_dir = get_plots_dir(subdir=Path(__file__).stem)
        os.makedirs(os.path.join(plots_dir, f"{model_name_a}{variant_b}/"), exist_ok=True)
        plt.savefig(
            os.path.join(
                plots_dir,
                f"{model_name_a}{variant_b}/{Path(__file__).stem}_{model_name_a}{variant_b}_{split}_{metric}_1.png",
            ),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0,
        )

        expansions = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

        for k in expansions:
            if f"partial_merging_{k}_{split}_{metric}" in metrics.keys():
                sns.lineplot(
                    x=metrics["alphas"],
                    y=metrics[f"partial_merging_{k}_{split}_{metric}"],
                    label=f"partial merging (+{int(round((k-1)*100))}% buffer)",
                    color=plt.cm.rainbow(k - 1),
                )

        plt.savefig(
            os.path.join(
                plots_dir,
                f"{model_name_a}{variant_b}/{Path(__file__).stem}_{model_name_a}{variant_b}_{split}_{metric}_2.png",
            ),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0,
        )

        if f"merging_REPAIR_{split}_{metric}" in metrics.keys():
            sns.lineplot(
                x=metrics["alphas"],
                y=metrics[f"merging_REPAIR_{split}_{metric}"],
                label="merging + REPAIR ",
                color=plt.cm.rainbow(0),
                dashes=(2, 2),
            )

        for k in expansions:
            if f"partial_merging_REPAIR_{k}_{split}_{metric}" in metrics.keys():
                sns.lineplot(
                    x=metrics["alphas"],
                    y=metrics[f"partial_merging_REPAIR_{k}_{split}_{metric}"],
                    # label=f"partial merging + REPAIR (+{int(round((k-1)*100))}% buffer)",
                    color=plt.cm.rainbow(k - 1),
                    dashes=(2, 2),
                )

        plt.savefig(
            os.path.join(
                plots_dir,
                f"{model_name_a}{variant_b}/{Path(__file__).stem}_{model_name_a}{variant_b}_{split}_{metric}_3.png",
            ),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        print(f"📊 {split} {metric} plot saved for {model_name_a}, {model_name_b}")
