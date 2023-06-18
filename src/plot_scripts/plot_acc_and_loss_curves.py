import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from src.utils import load_model, normalize, get_plots_dir, parse_model_name, get_all_model_names
from src.evaluate import evaluate_two_models


def plot_acc_and_loss_curves(model_name_a: str, model_name_b: str):
    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    metrics = evaluate_two_models("CIFAR10-VGG11-1x-a", "CIFAR10-VGG11-1x-b")

    plt.figure(figsize=(6, 4))

    sns.lineplot(x=metrics["alphas"], y=metrics["ensembling_test_accs"], label="ensembling")
    sns.lineplot(x=metrics["alphas"], y=metrics["naive_test_accs"], label="naive merging")
    sns.lineplot(x=metrics["alphas"], y=metrics["merging_test_accs"], label="merging")

    plt.xlabel("alpha")
    plt.ylabel("test accuracy")
    plt.title(f"{dataset_a}, {model_type_a}{size_a}, {width_a}×width, model {variant_a} vs. {variant_b}")

    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    plt.savefig(os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name_a}{variant_b}.png"), dpi=600)
    plt.close()
    print(f"📊 Plot saved for {model_name_a}, {model_name_b}")
