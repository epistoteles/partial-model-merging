import argparse
from tqdm import tqdm
import numpy as np
from codecarbon import track_emissions
import wandb

from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from models.VGG import VGG
from src.utils import get_loaders, save_model
from src.evaluate import get_acc_and_loss

from rich import pretty, print

pretty.install()


@track_emissions(log_level="critical")
def main():
    if args.wandb:
        wandb.init(project="partial-model-merging", config=args)

    train_aug_loader, _, test_loader = get_loaders(args.dataset)
    model = VGG(size=args.size, width=args.width, bn=args.batch_norm).cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # this lr schedule will start and end with a lr of 0, which should have no effect on the weights,
    # but should recalibrate the batch norm layers (if they exist)
    n_iters = len(train_aug_loader)
    lr_schedule = np.interp(np.arange(1 + args.epochs * n_iters), [0, 5 * n_iters, args.epochs * n_iters], [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    for epoch in tqdm(range(args.epochs)):
        model.train()
        total = 0
        train_loss = 0.0
        train_correct = 0
        for i, (inputs, labels) in enumerate(train_aug_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs.cuda())
                loss = loss_fn(outputs, labels.cuda())
                pred = outputs.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(labels.cuda().view_as(pred)).sum().item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
        train_accuracy = train_correct / total
        train_loss /= total
        metrics = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy}
        if args.wandb:
            if args.test:
                test_acc, test_loss = get_acc_and_loss(model, test_loader)
                metrics["test_accuracy"] = test_acc
                metrics["test_loss"] = test_loss
            wandb.log(metrics)

    save_model(
        model,
        f"{args.dataset}-{args.model_type}{args.size}-{'bn-' if args.batch_norm else ''}"
        f"{args.width}x-{args.variant}",
    )


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=11)
parser.add_argument("--width", type=int, default=1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.08)
parser.add_argument("--variant", type=str, default="a")
parser.add_argument("--dataset", type=str, choices=["CIFAR10", "CIFAR100", "SVHN", "ImageNet"], default="CIFAR10")
parser.add_argument("--model_type", type=str, choices=["VGG", "ResNet"], default="VGG")
parser.add_argument(
    "-bn", "--batch_norm", action="store_true", help="use batch norm layers in the model (default: none)"
)
parser.add_argument("-wandb", action="store_true")
parser.add_argument(
    "-test", action="store_true", help="also evaluates test acc. and loss if set; only used when -wandb is set too"
)
parser.add_argument(
    "-cm",
    "--checkpoint-midway",
    action="store_true",
    help="Checkpoints the model every 10 epochs if set (default: only at the end)",
)  # TODO: implement

if __name__ == "__main__":
    args = parser.parse_args()
    main()
