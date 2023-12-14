import argparse
import numpy as np
from codecarbon import track_emissions
import wandb

from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from safetensors.torch import load_file

from models.VGG import VGG
from models.ResNet import ResNet18, ResNet20
from models.MLP import MLP
from src.utils import get_loaders, save_model, get_num_classes
from src.evaluate import get_acc_and_loss, evaluate_single_model

from rich import pretty, print
from rich.progress import track

pretty.install()


@track_emissions()
def main():
    if args.wandb:
        wandb.init(project="partial-model-merging", config=args)

    train_aug_loader, _, test_loader = get_loaders(args.dataset)
    if args.model_type == "VGG":
        model = VGG(
            size=args.size, width=args.width, bn=args.batch_norm, num_classes=get_num_classes(args.dataset)
        ).cuda()
    elif args.model_type == "ResNet":
        if not args.batch_norm:
            raise ValueError("ResNet must have batch norm layers (-bn/--batch_norm)")
        if args.size == 18:
            ResNet = ResNet18
        elif args.size == 20:
            ResNet = ResNet20
        else:
            raise ValueError(f"Unavailable ResNet size {args.size}")
        model = ResNet(width=args.width, num_classes=get_num_classes(args.dataset)).cuda()
    elif args.model_type == "MLP":
        model = MLP(
            size=args.size, width=args.width, bn=args.batch_norm, num_classes=get_num_classes(args.dataset)
        ).cuda()
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    if args.pretrained is not None:
        pretrained_sd = load_file(args.pretrained)
        model.load_state_dict(pretrained_sd)
        print(f"Pretrained weights loaded from {args.pretrained}")

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # this lr schedule will start and end with a lr of 0, which should have no effect on the weights,
    # but should recalibrate the batch norm layers (if they exist)
    n_iters = len(train_aug_loader)
    lr_schedule = np.interp(np.arange(1 + args.epochs * n_iters), [0, 5 * n_iters, args.epochs * n_iters], [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    for epoch in track(range(args.epochs)):
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
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        if args.wandb:
            if args.test:
                test_acc, test_loss = get_acc_and_loss(model, test_loader)
                metrics["test_accuracy"] = test_acc
                metrics["test_loss"] = test_loss
            wandb.log(metrics)

    model_name = f"{args.dataset}-{args.model_type}{args.size}-{'bn-' if args.batch_norm else ''}{int(args.width) if args.width%1 == 0 else args.width}x-{args.variant}"
    save_model(model, model_name)

    evaluate_single_model(model_name)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    choices=[
        "CIFAR10",
        "CIFAR10C",
        "CIFAR10B",
        "CIFAR10C",
        "CIFAR10D",
        "CIFAR100",
        "CIFAR100A",
        "CIFAR100B",
        "SVHN",
        "SVHNC",
        "SVHND",
        "ImageNet",
        "MNIST",
    ],
    default="CIFAR10",
)
parser.add_argument("-m", "--model_type", type=str, choices=["VGG", "ResNet", "MLP"], default="VGG")
parser.add_argument("-s", "--size", type=int, default=11)
parser.add_argument(
    "-bn", "--batch_norm", action="store_true", help="use batch norm layers in the model (default: none)"
)
parser.add_argument("-w", "--width", type=float, default=1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.08)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("-v", "--variant", type=str, default="a")

parser.add_argument("-p", "--pretrained", type=str, help="the path of a pretrained model to use as initialization")

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
    print(args)
    main()
