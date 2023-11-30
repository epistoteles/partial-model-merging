from src.utils import get_all_model_names
from src.evaluate import evaluate_two_models
from src.plot_scripts.plot_acc_and_loss_curves import plot_acc_and_loss_curves
from src.plot_scripts.plot_correlation_histogram import plot_correlation_histogram

model_stems = {x[:-2] for x in get_all_model_names() if "MLP" in x and "bn" in x}

for model_name in model_stems:
    evaluate_two_models(model_name, expansions=[1.1, 1.5])
    plot_acc_and_loss_curves(model_name)
    plot_correlation_histogram(model_name)
