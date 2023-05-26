import numpy as np

import pytorch_lightning as pl
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from src.utils.utils import get_loaders_CIFAR10, save_model


class VGG(pl.LightningModule):
    def __init__(self, vgg_size: int, width_multiplier: int = 1, bn: bool = False):
        """
        A custom VGG module
        :param vgg_size: size of the VGG, one of {11, 13, 16, 19}
        :param width_multiplier: multiplier for the width of the network
        :param bn: uses batch norm if True, uses nothing if False
        """
        cfg = {
            11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }

        super(VGG, self).__init__()

        self.vgg_size = vgg_size
        self.bn = bn
        self.width_multiplier = width_multiplier
        self.features = self._make_layers(cfg[vgg_size])
        self.classifier = nn.Linear(self.width_multiplier * 512, 10)

        self.scaler = GradScaler()
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def configure_optimizers(self):
        optimizer = SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)

        n_iters = len(self.train_dataloader())  # self.trainer.num_training_batches ?
        lr_schedule = np.interp(
            np.arange(1 + self.hparams.epochs * n_iters), [0, 5 * n_iters, self.hparams.epochs * n_iters], [0, 1, 0]
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def train_dataloader(self):
        train_aug_loader, _, test_loader = get_loaders_CIFAR10()
        return train_aug_loader

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels if in_channels == 3 else self.width_multiplier * in_channels,
                        self.width_multiplier * x,
                        kernel_size=3,
                        padding=1,
                    )
                )
                if self.bn:
                    layers.append(nn.BatchNorm2d(self.width_multiplier * x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
