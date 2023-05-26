import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

from src.models.VGG import VGG
from src.utils.utils import load_model, normalize, get_plots_dir


TITLE = "CIFAR10, VGG11, 1×width"
MODEL_TO_PLOT = "VGG11-1x-b"
model = VGG(11)


model = load_model(model, MODEL_TO_PLOT)
sd = model.state_dict()

sums = []
for i, key in enumerate(sd.keys()):
    if "features" in key and "weight" in key:
        abs_sums = torch.abs(sd[key]).sum(dim=(1, 2, 3))
        normed_sums = normalize(abs_sums)
        normed_sums = torch.sort(normed_sums, descending=True)[0]
        sums += [normed_sums]

num_convs = len(sums)

plt.figure(figsize=(6, 6))

for i, s in enumerate(sums):
    sns.lineplot(x=np.linspace(0, 1, len(s)), y=s, label=f"Conv. {i+1}")

plt.xlabel("filter index / # filter (%)")
plt.ylabel("normalized abs. sum of filter weights")
plt.title(TITLE)

plt.savefig(os.path.join(get_plots_dir(), "plot_ranked_filters.png"), dpi=600)
