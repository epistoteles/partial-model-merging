import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    """
    A basic ResNet block, adapted from https://github.com/KellerJordan/REPAIR
    """

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1.is_buffer = nn.Parameter(torch.zeros(planes).bool(), requires_grad=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn1.is_buffer = nn.Parameter(torch.zeros(planes).bool(), requires_grad=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.is_buffer = nn.Parameter(torch.zeros(planes).bool(), requires_grad=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2.is_buffer = nn.Parameter(torch.zeros(planes).bool(), requires_grad=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes),
            )
            self.shortcut[0].is_buffer = nn.Parameter(torch.zeros(planes).bool(), requires_grad=False)
            self.shortcut[1].is_buffer = nn.Parameter(torch.zeros(planes).bool(), requires_grad=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, width: float = 1.0, num_classes: int = 10):
        """
        A custom ResNet20 module, adapted from https://github.com/KellerJordan/REPAIR
        :param width: multiplier for the width of the network
        :param num_classes: the number of output classes
        """
        super().__init__()
        self.size = 18
        self.bn = True
        self.num_classes = num_classes
        self.in_planes = round(width * 16)

        self.conv1 = nn.Conv2d(3, round(width * 16), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.is_buffer = nn.Parameter(torch.zeros(round(width * 16)).bool(), requires_grad=False)
        self.bn1 = nn.BatchNorm2d(round(width * 16))
        self.bn1.is_buffer = nn.Parameter(torch.zeros(round(width * 16)).bool(), requires_grad=False)
        self.layer1 = self._make_layer(BasicBlock, planes=round(width * 16), num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, planes=round(width * 32), num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes=round(width * 64), num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes=round(width * 128), num_blocks=2, stride=2)
        self.linear = nn.Linear(round(width * 128), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, width: float | list[float] | torch.FloatTensor = 1.0, num_classes: int = 10):
        """
        A custom ResNet18 module, adapted from https://github.com/KellerJordan/REPAIR
        :param width: multiplier for the width of the network
        :param num_classes: the number of output classes
        """
        super().__init__()
        self.size = 20
        self.bn = True
        self.num_classes = num_classes
        self.in_planes = round(width * 16)

        self.conv1 = nn.Conv2d(3, round(width * 16), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.is_buffer = nn.Parameter(torch.zeros(round(width * 16)).bool(), requires_grad=False)
        self.bn1 = nn.BatchNorm2d(round(width * 16))
        self.bn1.is_buffer = nn.Parameter(torch.zeros(round(width * 16)).bool(), requires_grad=False)
        self.layer1 = self._make_layer(BasicBlock, planes=round(width * 16), num_blocks=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, planes=round(width * 32), num_blocks=3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes=round(width * 64), num_blocks=3, stride=2)
        self.linear = nn.Linear(round(width * 64), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
