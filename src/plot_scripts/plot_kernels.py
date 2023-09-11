import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path
from math import ceil, sqrt
from src.utils import load_model, get_plots_dir


def plot_kernels(model_name, model=None):
    if model is None:
        model = load_model(model_name)
    sd = model.state_dict()

    weights = sd["features.0.weight"]
    is_buffer = sd["features.0.is_buffer"]

    num_kernels_nobuffer = weights[~is_buffer].shape[0]
    num_kernels = weights.shape[0]

    rows = sqrt(num_kernels_nobuffer)
    cols = ceil(num_kernels / rows)
    rows = ceil(rows)

    # for CIFAR10 (see utils)
    MEAN = torch.Tensor([125.307, 122.961, 113.8575])
    STD = torch.Tensor([51.5865, 50.847, 51.255])

    weights_rgb = ((weights * STD * 2) + MEAN).to(torch.uint8).numpy()

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols, rows))

    for i in range(cols):
        for j in range(rows):
            kernel = weights_rgb[i * rows + j]
            axes[j, i].imshow(kernel)
            axes[j, i].axis("off")

    plt.tight_layout()

    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name}_kernels.png"), dpi=600)
    plt.close()
    print(f"📊 Kernel plot saved for {model_name}")
