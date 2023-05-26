import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

from src.models.VGG import VGG
from src.utils.utils import load_model, normalize


TITLE = "CIFAR10, VGG11"
MODEL_TO_PLOT = "VGG11-1x-a"
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

for s in sums:
    sns.lineplot(x=np.linspace(0, 1, len(s)), y=s)

# plt.show()
