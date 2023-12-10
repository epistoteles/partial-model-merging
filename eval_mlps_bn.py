from src.utils import get_all_model_names
from src.evaluate import evaluate_two_models
from src.plot_scripts.plot_acc_and_loss_curves import plot_acc_and_loss_curves
from src.plot_scripts.plot_correlation_histogram import plot_correlation_histogram

for width in [0.125, 0.25, 0.5, 1, 2, 4]:
    for size in [3, 4, 5, 6, 7, 8, 9, 10]:
        model_name = f"MNIST-MLP{size}-bn-{width}x"
        evaluate_two_models(model_name, expansions=[1.2])
        plot_acc_and_loss_curves(model_name)
        plot_correlation_histogram(model_name)
