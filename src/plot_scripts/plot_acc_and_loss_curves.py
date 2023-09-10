import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from src.utils import load_model, normalize, get_plots_dir, parse_model_name, get_all_model_names
from src.evaluate import evaluate_two_models


def plot_acc_and_loss_curves(model_name_a: str, model_name_b: str):
    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    metrics = evaluate_two_models(model_name_a, model_name_b)

    for metric, split in zip(["accs", "losses"], ["train", "test"]):
        plt.figure(figsize=(12, 8))

        sns.lineplot(x=metrics["alphas"], y=metrics[f"ensembling_{split}_{metric}"], label="ensembling", color="black")
        # sns.lineplot(x=metrics["alphas"], y=metrics[f"naive_{split}_{metric}"], label="naive merging", color='')
        sns.lineplot(x=metrics["alphas"], y=metrics[f"merging_{split}_{metric}"], label="merging", color="orange")
        sns.lineplot(
            x=metrics["alphas"],
            y=metrics[f"partial_merging_1.1_{split}_{metric}"],
            label="partial merging (110% width)",
            color="red",
        )
        sns.lineplot(
            x=metrics["alphas"],
            y=metrics[f"partial_merging_1.5_{split}_{metric}"],
            label="partial merging (150% width)",
            color="blue",
        )
        sns.lineplot(
            x=metrics["alphas"],
            y=metrics[f"partial_merging_1.8_{split}_{metric}"],
            label="partial merging (180% width)",
            color="green",
        )
        sns.lineplot(
            x=metrics["alphas"],
            y=metrics[f"partial_merging_2.0_{split}_{metric}"],
            label="partial merging (200% width)",
            color="purple",
        )

        sns.lineplot(
            x=metrics["alphas"],
            y=metrics[f"merging_REPAIR_{split}_{metric}"],
            label="merging + REPAIR ",
            color="orange",
            dashes=(2, 2),
        )
        sns.lineplot(
            x=metrics["alphas"],
            y=metrics[f"partial_merging_REPAIR_1.1_{split}_{metric}"],
            label="partial merging + REPAIR (110% width)",
            color="red",
            dashes=(2, 2),
        )
        sns.lineplot(
            x=metrics["alphas"],
            y=metrics[f"partial_merging_REPAIR_1.5_{split}_{metric}"],
            label="partial merging + REPAIR (150% width)",
            color="blue",
            dashes=(2, 2),
        )
        sns.lineplot(
            x=metrics["alphas"],
            y=metrics[f"partial_merging_REPAIR_1.8_{split}_{metric}"],
            label="partial merging + REPAIR (180% width)",
            color="green",
            dashes=(2, 2),
        )
        sns.lineplot(
            x=metrics["alphas"],
            y=metrics[f"partial_merging_REPAIR_2.0_{split}_{metric}"],
            label="partial merging + REPAIR (200% width)",
            color="purple",
            dashes=(2, 2),
        )

        plt.xlabel("alpha")
        plt.ylabel(f"{split} {metric}")
        plt.title(f"{dataset_a}, {model_type_a}{size_a}, {width_a}×width, model {variant_a} vs. {variant_b}")

        plots_dir = get_plots_dir(subdir=Path(__file__).stem)
        plt.savefig(os.path.join(plots_dir, f"{Path(__file__).stem}_{model_name_a}{variant_b}_acc.png"), dpi=600)
        plt.close()
        print(f"📊 {metric} plot saved for {model_name_a}, {model_name_b}")
