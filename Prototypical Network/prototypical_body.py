import torch
import torch.nn as nn
import torch.nn.functional as F
from densenet_head import DenseNet
from resnet_head import ResNet, BasicBlock
from se_resnet_head import SEResNet, SEBasicBlock


class PrototypicalNetwork(nn.Module):
    def __init__(self, dis_f, head):
        super().__init__()
        self.dis_f = dis_f
        self.head = head
        if head == 'ResNet':  # ResNet10
            self.feature_encoder = ResNet(block=BasicBlock, layers=[1, 1, 1, 1])
            self.feature_size = 512
        elif head == 'DenseNet':  # DenseNet121
            self.feature_encoder = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64)
            self.feature_size = 1024
        elif head == 'SENet':  # SE-ResNet10
            self.feature_encoder = SEResNet(block=SEBasicBlock, layers=[1, 1, 1, 1])
            self.feature_size = 512
        elif head == 'SENet18':  # SE-ResNet18
            self.feature_encoder = SEResNet(block=SEBasicBlock, layers=[2, 2, 2, 2])
            self.feature_size = 512
        else:
            print(f"error：no feature encoder named {head}.")

        self.linear = nn.Linear(self.feature_size, 1)

    def distance(self, x, y, f):
        """
        compute the similarity score between two matrix
        :param x: a tensor with shape [len_x, 512]
        :param y: a tensor with shape [len_y, 512]
        :param f: a string meaning the computation method
        :return:
            a tensor with shape of [len_x, len_y], representing the similarity score of:
            [(x1, y1), (x1, y2), ..., (x1, yn)],
            [(x2, y1), ..., (x2, yn)],
            ...,
            [(xn, y1), ..., (xn, yn)]
        """
        len_x, len_y = x.shape[0], y.shape[0]

        if f == "L2":  # The L2 distance
            z = x.unsqueeze(1).expand(len_x, len_y, -1) - y.unsqueeze(0).expand(len_x, len_y, -1)
            z = z.pow(2).sum(dim=2)
            return z

        if f == "Sim":  # The Sim calculation module
            z = x.unsqueeze(1).expand(len_x, len_y, -1) - y.unsqueeze(0).expand(len_x, len_y, -1)
            z = z.pow(2)
            z = F.normalize(z, p=2, dim=1)
            z = self.linear(z).squeeze(-1)
            z = F.normalize(z, p=2, dim=1)
            return z

    def forward(self, x, way, shot):
        """
        :param x:
            a tensor with shape [way * shot + batch_size, 270, 1000]
            the forward way * shot lines represent the waves of the training support set, arranged in order of the first keyword by type
            the backward batch_size lines represent the waves of the training query set
        :param way: the number of categories
        :param shot: the n_{test}
        :return:
            a tensor with shape [batch_size, way]
            Corresponding to the logarithmic value of the probability that each waveform in the query set belongs to each category
        """

        # get the feature vectors
        features = self.feature_encoder.forward(x)
        support, query = torch.split(features,
                                     [way * shot, features.size(0) - way * shot])

        support = support.reshape(way, shot, -1)
        prototypes = support.mean(dim=1)  # get the prototypes

        # compute the distance (or similarity)
        dis = self.distance(query, prototypes, self.dis_f)

        return (-dis + 1e-8).log_softmax(dim=1)
