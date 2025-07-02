import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, group=1):
    """
    3x3卷积，带padding。
    3x3 convolution with padding.
    Args:
        in_planes: 输入通道数 / input channels
        out_planes: 输出通道数 / output channels
        stride: 步长 / stride
        group: 分组数 / groups
    Returns:
        nn.Conv1d对象 / nn.Conv1d object
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=group)


def conv1x1(in_planes, out_planes, stride=1, group=1):
    """
    1x1卷积。
    1x1 convolution.
    Args:
        in_planes: 输入通道数 / input channels
        out_planes: 输出通道数 / output channels
        stride: 步长 / stride
        group: 分组数 / groups
    Returns:
        nn.Conv1d对象 / nn.Conv1d object
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation注意力模块。
    Squeeze-and-Excitation (SE) attention module.
    """
    def __init__(self, channel, reduction=16):
        """
        初始化SE模块。
        Initialize SE module.
        Args:
            channel: 输入通道数 / input channels
            reduction: 压缩比 / reduction ratio
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播。
        Forward pass.
        Args:
            x: 输入特征 / input features
        Returns:
            加权特征 / weighted features
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    """
    SE-ResNet基本残差块。
    SE-ResNet basic residual block.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        """
        初始化SE基本块。
        Initialize SE basic block.
        Args:
            inplanes: 输入通道数 / input channels
            planes: 输出通道数 / output channels
            stride: 步长 / stride
            downsample: 下采样层 / downsample layer
            groups: 分组数 / groups
            base_width: 基础宽度 / base width
            dilation: 膨胀系数 / dilation
            norm_layer: 归一化层 / normalization layer
            reduction: SE压缩比 / SE reduction ratio
        """
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        前向传播。
        Forward pass.
        Args:
            x: 输入特征 / input features
        Returns:
            输出特征 / output features
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    """
    SE-ResNet主干网络。
    SE-ResNet backbone network.
    """
    def __init__(self, block, layers, inchannel=270, activity_num=13):
        """
        初始化SE-ResNet。
        Initialize SE-ResNet.
        Args:
            block: 残差块类型 / block type
            layers: 每层的块数 / number of blocks per layer
            inchannel: 输入通道数 / input channels
            activity_num: 动作类别数（未用）/ number of activity classes (unused)
        """
        super(SEResNet, self).__init__()
        # B*270*1000
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        # output: batch*64*8
        self.out_conv = nn.Conv1d(512, 64, kernel_size=1, stride=1, bias=False)
        self.out_bn = nn.BatchNorm1d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        """
        构建残差层。
        Build residual layer.
        Args:
            block: 残差块类型 / block type
            planes: 输出通道数 / output channels
            blocks: 块数 / number of blocks
            stride: 步长 / stride
            group: 分组数 / groups
        Returns:
            nn.Sequential对象 / nn.Sequential object
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, group))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播。
        Forward pass.
        Args:
            x: 输入特征，形状(batch, inchannel, 1000) / input features, shape (batch, inchannel, 1000)
        Returns:
            输出特征，形状(batch, 64, 8) / output features, shape (batch, 64, 8)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        out = self.out_conv(c4)
        out = self.out_bn(out)
        out = self.relu(out)
        
        out = self.adaptive_pool(out)
        
        # [batch, 64, 8]
        return out