import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from src.utils import load_model, normalize, get_plots_dir, parse_model_name, get_all_model_names
from src.evaluate import evaluate_two_models


def plot_reduction(dataset: str, architecture: str, bn: bool):
    sizes = torch.arange(3, 11).tolist()
    widths = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    endpoint_accuracies = torch.zeros(len(widths), len(sizes))
    for i, size in enumerate(sizes):
        for j, width in enumerate(widths):
            metrics = evaluate_two_models(f"{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x")
            acc_a = metrics["merging_test_accs"][0].item()
            acc_b = metrics["merging_test_accs"][-1].item()
            acc_avg = (acc_a + acc_b) / 2
            endpoint_accuracies[j, i] = acc_avg
    print(endpoint_accuracies)
