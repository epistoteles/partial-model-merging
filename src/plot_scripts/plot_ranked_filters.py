import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

from src.models.VGG import VGG
from src.utils.utils import load_model, normalize, get_plots_dir, parse_model_name


def plot_model_filters(model_name):
    model_type, size, width, variant = parse_model_name(model_name)
    model = VGG(size)
    model = load_model(model, model_name)
    sd = model.state_dict()

    sums = []
    for i, key in enumerate(sd.keys()):
        if "features" in key and "weight" in key:
            abs_sums = torch.abs(sd[key]).sum(dim=(1, 2, 3))
            normed_sums = normalize(abs_sums)
            normed_sums = torch.sort(normed_sums, descending=True)[0]
            sums += [normed_sums]

    plt.figure(figsize=(6, 6))

    for i, s in enumerate(sums):
        sns.lineplot(x=np.linspace(0, 1, len(s)), y=s, label=f"Conv. {i + 1}")

    plt.xlabel("filter index / # filter (%)")
    plt.ylabel("normalized abs. sum of filter weights")
    plt.title(f"CIFAR10, {model_type}{size}, {width}×width, model {model_name[-1]}")

    plt.savefig(os.path.join(get_plots_dir(), f"plot_ranked_filters_{model_name}.png"), dpi=600)


for model_name in ["VGG11-1x-a", "VGG11-1x-b"]:
    plot_model_filters(model_name)
