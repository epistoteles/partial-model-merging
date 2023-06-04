from torch import nn


class VGG(nn.Module):
    def __init__(self, size: int, width: float = 1.0, bn: bool = False, num_classes: int = 10):
        """
        A custom VGG module, adapted from https://github.com/KellerJordan/REPAIR
        :param size: size of the VGG, one of {11, 13, 16, 19}
        :param width: multiplier for the width of the network
        :param bn: uses batch norm if True, uses nothing if False
        :param num_classes: the number of output classes
        """
        cfg = {
            11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }

        super(VGG, self).__init__()
        self.size = size
        self.bn = bn
        self.num_classes = num_classes
        self.width = width
        self.features = self._make_layers(cfg[size])
        self.classifier = nn.Linear(round(self.width * 512), num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels if in_channels == 3 else round(self.width * in_channels),
                        round(self.width * x),
                        kernel_size=3,
                        padding=1,
                    )
                )
                if self.bn:
                    layers.append(nn.BatchNorm2d(round(self.width * x)))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
