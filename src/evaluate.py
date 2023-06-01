import torch
import numpy as np
from torch.cuda.amp import autocast
import torch.nn.functional as F
import os
from codecarbon import track_emissions
from src.utils import load_model, get_loaders, parse_model_name, get_evaluations_dir


def _evaluate_single_model(model: torch.nn.Module, loader):
    """
    Evaluates a single model in terms of accuracy and loss
    :param model: the model
    :param loader: a matching FFCV data loader
    :return: (accuracy, loss)
    """
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.cuda())
            pred = outputs.argmax(dim=1)
            correct += (labels.cuda() == pred).sum().item()
            total += len(labels)
            loss = F.cross_entropy(outputs, labels.cuda())
            losses.append(loss.item())
    return correct / total, np.array(losses).mean()


@track_emissions(log_level="error")
def evaluate_single_model(model_name: str):
    """
    Evaluates a single model in terms of accuracy and loss (and saves the result)
    :param model_name: the name of the model checkpoint
    :return: (train accuracy, train loss, test accuracy, test loss)
    """
    dataset, model_type, size, width, variant = parse_model_name(model_name)
    evaluations_dir = get_evaluations_dir(subdir="single_model")
    filepath = os.path.join(evaluations_dir, f"{model_name}.csv")
    columns = ("train_acc", "train_loss", "test_acc", "test_loss")

    if os.path.exists(filepath):
        train_acc, train_loss, test_acc, test_loss = np.genfromtxt("test.csv", delimiter=",", skip_header=1)
        values = (train_acc, train_loss, test_acc, test_loss)
        print(f"📤 Loaded saved accuracies and losses for {model_name}")
    else:
        model = load_model(model_name)
        _, train_noaug_loader, test_loader = get_loaders(dataset)

        train_acc, train_loss = _evaluate_single_model(model, train_noaug_loader)
        test_acc, test_loss = _evaluate_single_model(model, test_loader)

        values = (train_acc, train_loss, test_acc, test_loss)
        np.savetxt(filepath, [columns, values], delimiter=",", fmt="%s")

        print(f"📥 Accuracies and losses saved for {model_name}")

    for c, v in zip(columns, values):
        print(f"{c:<12} {v}")

    return train_acc, train_loss, test_acc, test_loss
