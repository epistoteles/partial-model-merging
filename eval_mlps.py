from src.evaluate import evaluate_two_models
from src.plot_scripts.plot_acc_and_loss_curves import plot_acc_and_loss_curves

for w in [1, 2, 4, 8]:
    for s in range(3, 11):
        model_name = f"MNIST-MLP{s}-{w}.0x"
        evaluate_two_models(model_name, expansions=[1.1, 1.2, 1.5])
        plot_acc_and_loss_curves(model_name)

for w in [1, 2, 4, 8]:
    for s in range(3, 11):
        model_name = f"MNIST-MLP{s}-bn-{w}.0x"
        evaluate_two_models(model_name, expansions=[1.1, 1.2, 1.5])
        plot_acc_and_loss_curves(model_name)
