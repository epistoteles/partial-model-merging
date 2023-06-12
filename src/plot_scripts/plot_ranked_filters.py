import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os
from pathlib import Path

from src.utils import load_model, normalize, get_plots_dir, parse_model_name, get_all_model_names


def plot_model_filters(model_name: str):
    dataset, model_type, size, batch_norm, width, variant = parse_model_name(model_name)
    model = load_model(model_name)
    # model = expand_model(model, 1.2)
    sd = model.state_dict()

    sums = []
    for i, key in enumerate(sd.keys()):
        if len(sd[key].shape) == 4:  # this means the parameters are convolutional kernels (and not biases, bn, ...)
            abs_sums = torch.abs(sd[key]).sum(dim=(1, 2, 3))
            normed_sums = normalize(abs_sums)
            normed_sums = torch.sort(normed_sums, descending=True)[0]
            sums += [normed_sums]

    plt.figure(figsize=(6, 6))

    for i, s in enumerate(sums):
        sns.lineplot(x=np.linspace(0, 1, len(s)), y=s, label=f"Conv. {i + 1}")

    plt.xlabel("filter index / # filter (%)")
    plt.ylabel("normalized abs. sum of filter weights")
    plt.title(f"{dataset}, {model_type}{size}, {width}×width, model {variant}")

    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name}.png"), dpi=600)
    print(f"📊 Plot saved for {model_name}")


for model_name in get_all_model_names():
    plot_model_filters(model_name)
