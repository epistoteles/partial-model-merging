import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import torch

from src.utils import get_evaluations_dir, get_plots_dir, parse_model_name, load_file


def _sd_item_to_key(item):
    _key = item.replace("layer", "").split(".")
    _key = [_convert(x) for x in _key]
    return _key


def _convert(x):
    try:
        return int(x)
    except ValueError:
        return x


def plot_correlation_histogram(model_name_a: str, model_name_b: str = None):
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, bn_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, bn_b, width_b, variant_b = parse_model_name(model_name_b)

    save_corr_path = os.path.join(get_evaluations_dir(subdir="correlations"), f"{model_name_a}{variant_b}.safetensors")

    metrics = load_file(save_corr_path)
    stems = list(dict.fromkeys([key.rsplit(".", 1)[0] for key in metrics.keys()]))
    stems = sorted(stems, key=_sd_item_to_key)
    stems = [x for x in stems if x not in ["layer7", "layer11", "layer15"]]

    corrs = [metrics[stem + ".correlations"] for stem in stems]
    perm_maps = [metrics[stem + ".perm_map"] for stem in stems]
    chosen_corrs = [c[torch.arange(0, len(p)).long(), p] for c, p in zip(corrs, perm_maps)]

    fig, axes = plt.subplots(
        1,
        len(chosen_corrs),
        figsize=(1.1 * len(chosen_corrs), 5) if model_type_a == "ResNet" else (1.5 * len(chosen_corrs), 4.5),
    )
    fig.suptitle(
        f"Histogram of correlations selected by LAP solver, per  layer,\n"
        f"{dataset_a}, {model_type_a}{size_a}, {width_a}×width",
        y=1.05,
    )

    for i in range(len(chosen_corrs)):
        axes[i].axhline(y=0, color="black", linewidth=1)
        sns.histplot(y=chosen_corrs[i], ax=axes[i], binrange=(-1, 1), bins=50, color="#d9ead3ff", stat="count")
        axes[i].set_ylim(-1.02, 1.02)
        axes[i].set_title(stems[i].replace(".", "\n").replace("layer", "conv"), size=10)
        axes[i].set_xlabel("count")
        last_xtick = axes[i].get_xticks()[-2]
        xlim = axes[i].get_xlim()[1]
        print(last_xtick)
        print(xlim)
        print(last_xtick / xlim)
        if last_xtick / xlim > 0.9:
            print(True)
            xlim = last_xtick * 1.25
            print(xlim)
        axes[i].set_xlim(0, xlim)
        axes[i].set_xticks([0, last_xtick])

        if i == 0:
            axes[i].set_yticks([-1, -0.5, 0, 0.5, 1])
            axes[i].set_ylabel("Pearson’s r")
        else:
            axes[i].set_yticks([])

    plt.subplots_adjust(wspace=0.1)
    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(
        os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name_a[:-2]}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    print(f"📊 Correlations plot saved for {model_name_a}, {model_name_b}")
