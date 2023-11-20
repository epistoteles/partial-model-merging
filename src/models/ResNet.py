import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    def __init__(
        self, in_planes: int, mid_planes: int, out_planes: int, stride: int = 1, downsample_kernel_size: int = 1
    ):
        """
        A basic ResNet block, adapted from https://github.com/KellerJordan/REPAIR
        :param downsample_kernel_size: the kernel size for the downsampling layer;
                                       can be set to 3 for reproducing the experiments of Git-ReBasin (ResNet20)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1.is_buffer = nn.Parameter(torch.zeros(mid_planes).bool(), requires_grad=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.bn1.is_buffer = nn.Parameter(torch.zeros(mid_planes).bool(), requires_grad=False)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.is_buffer = nn.Parameter(torch.zeros(out_planes).bool(), requires_grad=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn2.is_buffer = nn.Parameter(torch.zeros(out_planes).bool(), requires_grad=False)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=downsample_kernel_size,
                    stride=stride,
                    padding=(downsample_kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
            )
            self.downsample[0].is_buffer = nn.Parameter(torch.zeros(out_planes).bool(), requires_grad=False)
            self.downsample[1].is_buffer = nn.Parameter(torch.zeros(out_planes).bool(), requires_grad=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, width: float | list[float] | torch.FloatTensor = 1.0, num_classes: int = 10, norm: str = "bn"):
        """
        A custom ResNet18 module, adapted from https://github.com/KellerJordan/REPAIR
        :param width: multiplier for the width of the network; per layer if provided as iterable
        :param num_classes: the number of output classes
        :param norm: which normalization layers to use; either 'bn' (BatchNorm2d) or 'ln' (LayerNorm)  # TODO: implement
        """
        if isinstance(width, int):
            width = float(width)
        if isinstance(width, float):
            width = [width] * 17
        assert (isinstance(width, list) or isinstance(width, torch.Tensor)) and len(width) == 17
        assert (
            width[0] == width[2] == width[4]
            and width[6] == width[8]
            and width[10] == width[12]
            and width[14] == width[16]
        ), "width of residual activations must match"

        super().__init__()
        self.size = 18
        self.num_layers = 17  # number of layers which can be expanded
        self.bn = norm == "bn"
        self.ln = norm == "ln"  # . . . . . the marked self.base_sizes below  must stay the same after model expansion
        self.num_classes = num_classes  # . ╭───────┬───────╮       ╭───────╮       ╭───────╮         ╭─────────╮
        self.base_sizes = torch.LongTensor([16, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128])
        self.width = torch.FloatTensor(width)
        self.scaled_sizes = torch.round(self.base_sizes * self.width).long()
        self.in_planes = self.scaled_sizes[0]

        self.conv1 = nn.Conv2d(3, self.scaled_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.is_buffer = nn.Parameter(torch.zeros(self.scaled_sizes[0]).bool(), requires_grad=False)
        self.bn1 = nn.BatchNorm2d(self.scaled_sizes[0])
        self.bn1.is_buffer = nn.Parameter(torch.zeros(self.scaled_sizes[0]).bool(), requires_grad=False)
        self.layer1 = self._make_layer(BasicBlock, planes=self.scaled_sizes[1:5], num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, planes=self.scaled_sizes[5:9], num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes=self.scaled_sizes[9:13], num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes=self.scaled_sizes[13:], num_blocks=2, stride=2)
        self.linear = nn.Linear(self.scaled_sizes[-1], num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes: list[int] | torch.LongTensor, num_blocks: int, stride: int):
        assert len(planes) == num_blocks * 2
        assert planes[1] == planes[3]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for (p1, p2), stride in zip(zip(planes[::2], planes[1::2]), strides):
            layers.append(block(in_planes=self.in_planes, mid_planes=p1, out_planes=p2, stride=stride))
            self.in_planes = p2
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
    def __init__(self, width: float | list[float] | torch.FloatTensor = 1.0, num_classes: int = 10, norm: str = "bn"):
        """
        A custom ResNet20 module, adapted from https://github.com/KellerJordan/REPAIR
        :param width: multiplier for the width of the network
        :param num_classes: the number of output classes
        :param norm: which normalization layers to use; either 'bn' (BatchNorm2d) or 'ln' (LayerNorm)  # TODO: implement
        """
        if isinstance(width, int):
            width = float(width)
        if isinstance(width, float):
            width = [width] * 19
        assert (isinstance(width, list) or isinstance(width, torch.Tensor)) and len(width) == 19
        assert (
            width[0] == width[2] == width[4] == width[6]
            and width[8] == width[10] == width[12]
            and width[14] == width[16] == width[18]
        ), "width of residual activations must match"

        super().__init__()
        self.size = 20
        self.num_layers = 19  # number of layers which can be expanded
        self.bn = norm == "bn"
        self.ln = norm == "ln"  # . . . . . the marked self.base_sizes below  must stay the same after model expansion
        self.num_classes = num_classes  # . ╭───────┬───────┬───────╮       ╭───────┬───────╮       ╭───────┬───────╮
        self.base_sizes = torch.LongTensor([16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64])
        self.width = torch.FloatTensor(width)
        self.scaled_sizes = torch.round(self.base_sizes * self.width).long()
        self.in_planes = self.scaled_sizes[0]

        breakpoint()

        self.conv1 = nn.Conv2d(3, self.scaled_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.is_buffer = nn.Parameter(torch.zeros(self.scaled_sizes[0]).bool(), requires_grad=False)
        self.bn1 = nn.BatchNorm2d(self.scaled_sizes[0])
        self.bn1.is_buffer = nn.Parameter(torch.zeros(self.scaled_sizes[0]).bool(), requires_grad=False)
        self.layer1 = self._make_layer(BasicBlock, planes=self.scaled_sizes[1:7], num_blocks=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, planes=self.scaled_sizes[7:13], num_blocks=3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes=self.scaled_sizes[13:], num_blocks=3, stride=2)
        self.linear = nn.Linear(self.scaled_sizes[-1], num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes: list[int] | torch.LongTensor, num_blocks: int, stride: int):
        assert len(planes) == num_blocks * 2
        assert planes[1] == planes[3]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, downsample_kernel_size=3))
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
