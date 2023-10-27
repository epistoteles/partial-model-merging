import torch
from torch import nn
from src.models import MergeableModule


class MLP(MergeableModule):
    def __init__(
        self,
        size: int = 3,
        width: float | list[float] | torch.FloatTensor = 1.0,
        bn: bool = False,
        num_classes: int = 10,
    ):
        """
        A custom multi-layer perceptron module
        :param size: equivalent to the number of layers, only implemented for size == 3
        :param width: multiplier for the width of the network;
                      alternatively you can provide a list or FloatTensor of length # of layers,
                      which widens each layer of the model by a different factor
        :param bn: uses batch norm if True, uses nothing if False
        :param num_classes: the number of output classes
        """

        assert size == 3

        if type(width) is int:
            width = float(width)
        if type(width) is float:
            width = [width] * size
        assert (type(width) is list or type(width) is torch.Tensor) and len(width) == size

        super(MLP, self).__init__()
        self.size = size
        self.bn = bn
        self.num_classes = num_classes
        self.width = torch.FloatTensor(width)

        self.classifier = nn.Sequential(
            nn.Linear(32 * 32, round(512 * self.width[0].item())),
            *[nn.BatchNorm1d(round(512 * self.width[0].item())), nn.ReLU()] if bn else [nn.ReLU()],
            nn.Linear(round(512 * self.width[0].item()), round(512 * self.width[1].item())),
            *[nn.BatchNorm1d(round(512 * self.width[1].item())), nn.ReLU()] if bn else [nn.ReLU()],
            nn.Linear(round(512 * self.width[1].item()), round(512 * self.width[2].item())),
            *[nn.BatchNorm1d(round(512 * self.width[2].item())), nn.ReLU()] if bn else [nn.ReLU()],
            nn.Linear(round(512 * self.width[2].item()), num_classes),
            nn.LogSoftmax()
        )

        for layer in self.classifier[:-2]:
            if isinstance(layer, nn.Linear):
                layer.is_buffer = nn.Parameter(torch.zeros_like(layer.bias).bool(), requires_grad=False)

    def forward(self, x):
        return self.classifier(x)

    def _expand(self, expansion_factor: int | float | list[float] | torch.FloatTensor):
        """
        Returns a functionally equivalent but wider model. The appended weights and biases are all zero.
        """

    def num_layers(self):
        return self.size
