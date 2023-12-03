import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from safetensors.torch import load_file
from src.utils import load_model, normalize, get_plots_dir, parse_model_name, get_all_model_names, get_evaluations_dir
from src.evaluate import evaluate_two_models


def plot_reduction(dataset: str, architecture: str, bn: bool):
    sizes = torch.arange(3, 11).tolist()
    widths = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    accs_endpoint = torch.zeros(len(widths), len(sizes))
    accs_ensembling = torch.zeros(len(widths), len(sizes))
    accs_merging = torch.zeros(len(widths), len(sizes))
    accs_1_1 = torch.zeros(len(widths), len(sizes))
    accs_1_5 = torch.zeros(len(widths), len(sizes))
    for i, size in enumerate(sizes):
        for j, width in enumerate(widths):
            dir = get_evaluations_dir(subdir="two_models")
            metrics = load_file(f"{dir}/{dataset}-{architecture}{size}-{'bn-' if bn else ''}{width}x-ab.safetensors")
            acc_a = metrics["merging_test_accs"][0].item()
            acc_b = metrics["merging_test_accs"][-1].item()
            acc_avg = (acc_a + acc_b) / 2
            acc_ensembling = metrics["ensembling_test_accs"][10].item()
            acc_merging = metrics["merging_test_accs"][10].item()
            acc_merging = metrics["merging_test_accs"][10].item()
            acc_1_1 = metrics["partial_merging_1.1_test_accs"][10].item()
            acc_1_5 = metrics["partial_merging_1.5_test_accs"][10].item()
            accs_endpoint[j, i] = acc_avg
            accs_ensembling[j, i] = acc_ensembling
            accs_merging[j, i] = acc_merging
            accs_1_1[j, i] = acc_1_1
            accs_1_5[j, i] = acc_1_5
    return None
