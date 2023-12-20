import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from safetensors.torch import load_file

from src.utils import get_plots_dir, get_evaluations_dir
import os


dataset = "CIFAR10"
architecture = "ResNet"
size = 18
width = 1

metrics = load_file(
    os.path.join(
        get_evaluations_dir(subdir="adaptive"), f"adaptive-{dataset}-{architecture}{size}-bn-{width}x-ab.safetensors"
    )
)


thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8]

plt.figure(figsize=(12, 8))
plt.xlabel("layer")
plt.ylabel("width increase (%)")
plt.title(f"expansions resulting from adaptive buffer assignment\n{dataset}, {architecture}{size}, {width}×width")

sns.lineplot(
    x=range(1, len(metrics["adaptive_merging_0.1_used_neurons_relative"]) + 1),
    y=[100] * len(metrics["adaptive_merging_0.1_used_neurons_relative"]),
    label="epsilon=1.0",
    color=plt.cm.rainbow(0.999),
    marker="o",
    markersize=4,
)

for threshold in reversed(thresholds):
    if f"adaptive_merging_{threshold:g}_used_neurons_relative" in metrics.keys():
        used_neurons_relative = metrics[f"adaptive_merging_{threshold:g}_used_neurons_relative"]
        sns.lineplot(
            x=range(1, len(used_neurons_relative) + 1),
            y=used_neurons_relative * 100,
            label=f"epsilon={threshold:g}",
            color=plt.cm.rainbow(threshold),
            marker="o",
            markersize=4,
        )

sns.lineplot(
    x=range(1, len(metrics["adaptive_merging_0.1_used_neurons_relative"]) + 1),
    y=[0] * len(metrics["adaptive_merging_0.1_used_neurons_relative"]),
    label="epsilon=0.0",
    color=plt.cm.rainbow(0),
    marker="o",
    markersize=4,
)

plt.xticks(range(1, len(metrics["adaptive_merging_0.1_used_neurons_relative"]) + 1))

plt.tight_layout()
plots_dir = get_plots_dir(subdir=Path(__file__).stem)
plt.savefig(
    os.path.join(
        plots_dir,
        f"adaptive_expansion_{dataset}_{architecture}.png",
    ),
    dpi=600,
    bbox_inches="tight",
)
plt.close()
print(f"📊 adaptive expansion{dataset} {architecture}{size} plot saved")
