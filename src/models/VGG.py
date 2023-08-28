import torch
from torch import nn


class VGG(nn.Module):
    def __init__(
        self, size: int, width: float | list[float] | torch.FloatTensor = 1.0, bn: bool = False, num_classes: int = 10
    ):
        """
        A custom VGG module, adapted from https://github.com/KellerJordan/REPAIR
        :param size: size of the VGG, one of {11, 13, 16, 19}
        :param width: multiplier for the width of the network;
                      alternatively you can provide a list or FloatTensor of length # of layers, which widens each
                      layer of the model by a different factor
        :param bn: uses batch norm if True, uses nothing if False
        :param num_classes: the number of output classes
        """
        cfg = {
            11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }

        num_layers = {key: len([x for x in cfg[key] if type(x) is int]) for key in cfg.keys()}

        assert size in cfg.keys()
        if type(width) is int:
            width = float(width)
        if type(width) is float:
            width = [width] * num_layers[size]
        assert (type(width) is list or type(width) is torch.Tensor) and len(width) == num_layers[size]

        super(VGG, self).__init__()
        self.size = size
        self.bn = bn
        self.num_classes = num_classes
        self.num_layers = num_layers[size]
        self.width = torch.FloatTensor(width)
        self.features = self._make_layers(cfg[size])
        self.classifier = nn.Linear(round(self.width[-1].item() * 512), num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        widths = self.width.tolist()
        width = None
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                prev_width = width
                width = widths.pop(0)
                conv = nn.Conv2d(
                    in_channels=in_channels if in_channels == 3 else round(prev_width * in_channels),
                    out_channels=round(width * x),
                    kernel_size=3,
                    padding=1,
                )
                conv.is_buffer = nn.Parameter(torch.zeros_like(conv.bias).bool(), requires_grad=False)
                layers.append(conv)
                if self.bn:
                    layers.append(nn.BatchNorm2d(round(width * x)))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
