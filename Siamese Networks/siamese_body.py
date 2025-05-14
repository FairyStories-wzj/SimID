"""
The Siamese Networks that could be installed with various feature encoders
"""
import torch
import torch.nn as nn
from densenet_head import DenseNet
from resnet_head import ResNet, BasicBlock
from se_resnet_head import SEBasicBlock, SEResNet


class SiameseNetwork(nn.Module):
    def __init__(self, head):
        """
        :param head: which feature encoder to use
        """
        super().__init__()
        self.head = head
        if head == 'ResNet':  # ResNet for ResNet10
            self.future_encoder = ResNet(block=BasicBlock, layers=[1, 1, 1, 1])
            self.future_size = 512
        elif head == 'DenseNet':  # DenseNet121
            self.future_encoder = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64)
            self.future_size = 1024
        elif head == 'SENet':  # SE-ResNet10
            self.future_encoder = SEResNet(block=SEBasicBlock, layers=[1, 1, 1, 1])
            self.future_size = 512
        elif head == 'SENet18':  # SE-ResNet18
            self.future_encoder = SEResNet(block=SEBasicBlock, layers=[2, 2, 2, 2])
            self.future_size = 512
        else:
            print(f"error：not feature encoder named {head}")

        self.linear = nn.Linear(self.future_size, 1)
        self.norm = nn.BatchNorm1d(self.future_size)

    def forward(self, x1, x2):
        if x1.size(0) != x2.size(0):
            print("error：the samples are not paired")

        waves = torch.cat((x1, x2), dim=0)

        out = self.future_encoder.forward(waves)
        out1 = out[:x1.size(0)]
        out2 = out[x1.size(0):]
        # We used a similarity computation module similar to that in Prototypical Networks,
        # but the Siamese Networks will have one less normalization
        dis = (out1 - out2) ** 2
        dis = self.norm(dis)
        dis = self.linear(dis)

        return dis
