import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        size: int = 3,
        width: float | list[float] | torch.FloatTensor = 1.0,
        bn: bool = False,
        num_classes: int = 10,
    ):
        """
        A custom multi-layer perceptron module
        :param size: equivalent to the number of layers (must be >=2)
        :param width: multiplier for the width of the network;
                      alternatively you can provide a list or FloatTensor of length # of layers,
                      which widens each layer of the model by a different factor
        :param bn: uses batch norm if True, uses nothing if False
        :param num_classes: the number of output classes
        """
        super(MLP, self).__init__()

        if type(width) is int:
            width = float(width)
        if type(width) is float:
            width = [width] * (size - 1)
        assert (type(width) is list or type(width) is torch.Tensor) and len(
            width
        ) == size - 1, f"Width list length ({len(width)}) does not match number of hidden layers ({size-1})"

        self.size = size
        self.num_layers = size - 1  # the number of permutable (= hidden) layers
        self.bn = bn
        self.num_classes = num_classes
        self.width = torch.FloatTensor(width)

        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, round(512 * self.width[0].item())),
            *[nn.BatchNorm1d(round(512 * self.width[0].item())), nn.ReLU()] if bn else [nn.ReLU()],
        )

        for i in range(1, self.size - 1):
            self.classifier.extend(
                nn.Sequential(
                    nn.Linear(round(512 * self.width[i - 1].item()), round(512 * self.width[i].item())),
                    *[nn.BatchNorm1d(round(512 * self.width[i].item())), nn.ReLU()] if bn else [nn.ReLU()],
                )
            )

        self.classifier.extend(
            nn.Sequential(nn.Linear(round(512 * self.width[-1].item()), num_classes), nn.LogSoftmax())
        )

        for layer in self.classifier[:-2]:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm1d):
                layer.is_buffer = nn.Parameter(torch.zeros_like(layer.bias).bool(), requires_grad=False)

    def forward(self, x):
        return self.classifier(x)
