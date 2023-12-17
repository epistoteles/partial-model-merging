import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from safetensors.torch import load_file
from src.utils import get_plots_dir, get_evaluations_dir


plt.figure(figsize=(4, 4))
plt.xlabel("added buffers in both endpoint (%)")
plt.ylabel("added non-zero parameter count (%)")
plt.xticks(torch.linspace(0, 100, 11))
plt.yticks(torch.linspace(0, 100, 6))
plt.title("Width increase vs. parameter increase")

# # AUC diagonal
# sns.lineplot(x=torch.linspace(0, 100, 11), y=torch.linspace(0, 100, 11), color="grey")

expansions = torch.linspace(0, 100, 11)

params_vgg11 = torch.Tensor(
    [9225610, 10976040, 12535147, 13937515, 15128738, 16142410, 16972104, 17615083, 18083179, 18358274, 18451210]
)
params_resnet18 = torch.Tensor(
    [703886, 836648, 958400, 1059566, 1153024, 1230150, 1293392, 1343832, 1377990, 1400136, 1407742]
)

increase_vgg11 = ((params_vgg11 / params_vgg11[0]) - 1) * 100
increase_resnet18 = ((params_resnet18 / params_resnet18[0]) - 1) * 100

sns.lineplot(
    x=expansions,
    y=increase_vgg11,
    label="VGG11",
    color="grey",
    marker="o",
    markersize=4,
)

sns.lineplot(
    x=expansions,
    y=increase_resnet18,
    label="ResNet18",
    color="black",
    dashes=(2, 2),
    marker="o",
    markersize=4,
)

plots_dir = get_plots_dir(subdir=Path(__file__).stem)
plt.savefig(
    os.path.join(
        plots_dir,
        "param_count.png",
    ),
    dpi=600,
)
plt.close()
print("📊 param count plot saved")
