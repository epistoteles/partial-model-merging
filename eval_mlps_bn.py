from src.evaluate import evaluate_two_models
from src.plot_scripts.plot_acc_and_loss_curves import plot_acc_and_loss_curves
from src.plot_scripts.plot_correlation_histogram import plot_correlation_histogram

for w in [0.125, 0.25, 0.5, 1.0, 2.0]:  # 4, 8
    for s in range(3, 11):
        model_name = f"MNIST-MLP{s}-bn-{w}x"
        evaluate_two_models(model_name, expansions=[1.1, 1.2, 1.5])
        plot_acc_and_loss_curves(model_name)
        plot_correlation_histogram(model_name)
