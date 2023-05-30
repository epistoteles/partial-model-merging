import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from models.VGG import VGG
from utils.utils import get_loaders_CIFAR10, save_model

from rich import print
from rich import pretty

pretty.install()


# evaluates accuracy
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.cuda())
            pred = outputs.argmax(dim=1)
            correct += (labels.cuda() == pred).sum().item()
    return correct


def main():
    train_aug_loader, _, test_loader = get_loaders_CIFAR10()

    model = VGG(size=args.size, width=args.width).cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # this lr schedule will start and end with a lr of 0, which should have no effect on the weights,
    # but recalibrates the batch norm layers (if they exist)
    n_iters = len(train_aug_loader)
    lr_schedule = np.interp(np.arange(1 + args.epochs * n_iters), [0, 5 * n_iters, args.epochs * n_iters], [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    losses = []
    for _ in tqdm(range(args.epochs)):
        model.train()
        for i, (inputs, labels) in enumerate(train_aug_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs.cuda())
                loss = loss_fn(outputs, labels.cuda())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

    save_model(model, f"{args.dataset}-VGG{args.size}-{args.width}x-{args.name}")


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=11)
parser.add_argument("--width", type=int, default=1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.08)
parser.add_argument("--name", type=str, default="a")
parser.add_argument("--dataset", type=str, default="CIFAR10")
if __name__ == "__main__":
    args = parser.parse_args()
    main()
