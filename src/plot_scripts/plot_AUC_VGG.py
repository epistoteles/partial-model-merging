import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from safetensors.torch import load_file
from src.utils import get_plots_dir, get_evaluations_dir


dataset = "CIFAR10"
architecture = "VGG"
bn = True

sizes = [11]
widths = [0.25, 0.5, 1, 2]
expansions = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
accs_endpoint = torch.zeros(len(widths), len(sizes))
accs_ensembling = torch.zeros(len(widths), len(sizes))
accs_partial_merging = torch.zeros(len(widths), len(sizes), len(expansions))
accs_partial_merging_REPAIR = torch.zeros(len(widths), len(sizes), len(expansions))

for i, size in enumerate(sizes):
    for j, width in enumerate(widths):
        dir = get_evaluations_dir(subdir="two_models")
        metrics = load_file(f"{dir}/{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x-ab.safetensors")
        acc_a = metrics["ensembling_test_accs"][0].item()
        acc_b = metrics["ensembling_test_accs"][-1].item()
        acc_avg = (acc_a + acc_b) / 2
        acc_ensembling = metrics["ensembling_test_accs"][10].item()

        accs_endpoint[j, i] = acc_avg
        accs_ensembling[j, i] = acc_ensembling

        accs_partial_merging[j, i, 0] = metrics["merging_test_accs"][10]
        accs_partial_merging_REPAIR[j, i, 0] = metrics["merging_REPAIR_test_accs"][10]

        for idx, k in enumerate(expansions[1:]):
            accs_partial_merging[j, i, idx + 1] = metrics[f"partial_merging_{k}_test_accs"][10]
            accs_partial_merging_REPAIR[j, i, idx + 1] = metrics[f"partial_merging_REPAIR_{k}_test_accs"][10]

full_barrier_absolute = accs_endpoint - accs_partial_merging[:, 0, 0:1]
barrier_reduction_absolute = accs_partial_merging - accs_partial_merging[:, :, 0:1]
barrier_reduction_relative = barrier_reduction_absolute / full_barrier_absolute.unsqueeze(-1)

full_barrier_absolute_REPAIR = accs_endpoint - accs_partial_merging_REPAIR[:, 0, 0:1]
barrier_reduction_absolute_REPAIR = accs_partial_merging_REPAIR - accs_partial_merging_REPAIR[:, :, 0:1]
barrier_reduction_relative_REPAIR = barrier_reduction_absolute_REPAIR / full_barrier_absolute_REPAIR.unsqueeze(-1)


plt.figure(figsize=(7, 7))
plt.xlabel("added buffer (%)")
plt.ylabel("accuracy barrier reduction (%)")
plt.xticks(torch.linspace(0, 100, 11))

# AUC diagonal
sns.lineplot(x=torch.linspace(0, 100, 11), y=torch.linspace(0, 100, 11), dashes=(2, 2), color="grey")

# 100% horizontal line
sns.lineplot(x=torch.linspace(0, 100, 11), y=[100] * 11, dashes=(2, 2), color="grey")

for idx, (width, color) in enumerate(zip(widths, ["red", "orange", "yellow", "green"])):
    sns.lineplot(
        x=torch.linspace(0, 100, 11), y=barrier_reduction_relative_REPAIR[idx][0] * 100, label=width, color=color
    )

plots_dir = get_plots_dir(subdir=Path(__file__).stem)
plt.savefig(
    os.path.join(
        plots_dir,
        "AUC_VGG11.png",
    ),
    dpi=600,
)
plt.close()
print("📊 AUC VGG11 plot saved")
