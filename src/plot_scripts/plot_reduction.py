import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import product

import torch

from src.utils import load_model, normalize, get_plots_dir, parse_model_name, get_all_model_names
from src.evaluate import evaluate_two_models


def plot_reduction(dataset: str, architecture: str, bn: bool):
    pass
