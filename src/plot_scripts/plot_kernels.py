import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path

from src.utils import load_model, get_plots_dir


def plot_kernels(model_name, model=None):
    if model is None:
        model = load_model(model_name)
    sd = model.state_dict()

    weights = sd["features.0.weight"]

    # for CIFAR10 (see utils)
    MEAN = torch.Tensor([125.307, 122.961, 113.8575])
    STD = torch.Tensor([51.5865, 50.847, 51.255])

    weights_rgb = ((weights * STD * 2) + MEAN).to(torch.uint8).numpy()

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))

    # Loop through the kernels and plot them in the grid
    for i in range(8):
        for j in range(8):
            # Get the kernel weights for this position in the grid
            kernel = weights_rgb[i * 8 + j]

            # Display the kernel as an image in the corresponding subplot
            axes[i, j].imshow(kernel)
            axes[i, j].axis("off")  # Turn off axis labels and ticks

    plt.tight_layout()

    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name}_kernels.png"), dpi=600)
    plt.close()
    print(f"📊 Kernel plot saved for {model_name}")
