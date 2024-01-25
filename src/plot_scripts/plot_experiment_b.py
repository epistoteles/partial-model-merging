from src.utils import get_evaluations_dir, parse_model_name, get_plots_dir
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def plot_experiment_b(model_name_a: str, model_name_b: str = None):
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    evaluations_dir = get_evaluations_dir(subdir="experiment_b")
    filepath = os.path.join(
        evaluations_dir,
        f"experiment-b-{dataset_a}-{model_type_a}{size_a}{'-bn' if batch_norm_a else ''}-{width_a}x-{model_name_a[-1]}{model_name_b[-1]}.safetensors",
    )
    if os.path.exists(filepath):
        metrics = load_file(filepath)
    else:
        raise ValueError(f"Models not evaluated in experiment b: {filepath} does not exist")

    plt.figure(figsize=(12, 8))
    plt.xlabel("layer")
    plt.ylabel("accuracy")
    plt.title(
        f"Experiment B\n{dataset_a}, {model_type_a}{size_a},{' bn,' if batch_norm_a else ''} {width_a}×width, model {variant_a} vs. {variant_b}"
    )

    accs = metrics["only_expand_layer_i_REPAIR_test_accs"]

    expansions = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    for idx, k in enumerate(expansions):
        sns.lineplot(
            x=range(1, len(accs[0]) + 1),
            y=accs[idx],
            label=f"+{int(round((k - 1) * 100))}% buffer",
            color=plt.cm.rainbow(k - 1),
        )

    plots_dir = get_plots_dir(subdir=Path(__file__).stem)
    os.makedirs(os.path.join(plots_dir, f"{model_name_a}{variant_b}/"), exist_ok=True)
    plt.savefig(
        os.path.join(
            plots_dir,
            f"{model_name_a}{variant_b}/{Path(__file__).stem}_{model_name_a}{variant_b}_test_accs.png",
        ),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
    print(f"📊 test accs plot saved for {model_name_a}, {model_name_b}")
