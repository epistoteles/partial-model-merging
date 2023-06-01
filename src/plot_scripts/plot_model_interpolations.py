import matplotlib.pyplot as plt
import os
from pathlib import Path

from src.utils import get_plots_dir, parse_model_name


def plot_model_interpolation(model_name_a, model_name_b):
    dataset, model_type, size, batch_norm, width, variant = parse_model_name(model_name_a)
    # model_a = load_model(model_name_a)
    # model_b = load_model(model_name_b)

    # TODO: load values instead of running model, then plot

    plt.figure(figsize=(6, 6))

    plt.xlabel("filter index / # filter (%)")
    plt.ylabel("normalized abs. sum of filter weights")
    plt.title(f"{dataset}, {model_type}{size}{'-bn' if batch_norm else ''}, {width}×width, model {variant}")

    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(os.path.join(plots_dir, f"plot_ranked_filters_{model_name_a.replace('-a', '')}.png"), dpi=600)
    print(f"📊 Plot saved for {model_name_a} + {model_name_b}")
