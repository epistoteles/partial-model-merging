from src.utils import get_all_model_names
from src.evaluate import evaluate_two_models
from src.plot_scripts.plot_acc_and_loss_curves import plot_acc_and_loss_curves
from src.plot_scripts.plot_correlation_histogram import plot_correlation_histogram

for w in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
    model_name = f"MNIST-MLP5-{w}x"
    evaluate_two_models(model_name, expansions=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    plot_acc_and_loss_curves(model_name)
    plot_correlation_histogram(model_name)
