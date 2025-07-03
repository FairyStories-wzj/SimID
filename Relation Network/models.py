# --------------- models.py ---------------
# 模型管理器，负责模型的初始化、权重初始化、优化器和调度器设置
# Model manager, responsible for model initialization, weight initialization, optimizer and scheduler setup
import math

import torch

import RelationNetworkWithResNet
import SEResNet


class ModelManager:
    """
    模型管理器类，提供模型初始化、权重初始化、优化器和调度器设置等功能。
    Model manager class, provides model initialization, weight initialization, optimizer and scheduler setup, etc.
    """
    @staticmethod
    def weights_init(m):
        """
        权重初始化方法。
        Weight initialization method.
        Args:
            m: 模型层 / model layer
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())

    @classmethod
    def initialize_models(cls, config):
        """
        初始化特征编码器和关系网络。
        Initialize feature encoder and relation network.
        Args:
            config: 配置对象 / config object
        Returns:
            (feature_encoder, relation_network): 特征编码器和关系网络 / feature encoder and relation network
        """
        feature_encoder = SEResNet.SEResNet(block=SEResNet.SEBasicBlock, layers=[1, 1, 1, 1])
        # feature_encoder = ResNet.ResNet(block=ResNet.BasicBlock, layers=[1, 1, 1, 1])
        relation_network = RelationNetworkWithResNet.RelationNetwork(config.feature_dim,config.relation_dim)

        # feature_encoder.apply(cls.weights_init)
        relation_network.apply(cls.weights_init)

        feature_encoder.cuda(config.gpu)
        relation_network.cuda(config.gpu)
        return feature_encoder, relation_network

    @classmethod
    def setup_optimizers(cls, feature_encoder, relation_network, config):
        """
        设置优化器和学习率调度器。
        Setup optimizers and learning rate schedulers.
        Args:
            feature_encoder: 特征编码器 / feature encoder
            relation_network: 关系网络 / relation network
            config: 配置对象 / config object
        Returns:
            (optimizers, schedulers): 优化器和调度器字典 / dictionaries of optimizers and schedulers
        """
        step_size = 200

        # 设置优化器 / Setup optimizers
        optimizers = {
            "feature": torch.optim.Adam(feature_encoder.parameters(), lr=config.learning_rate),
            "relation": torch.optim.Adam(relation_network.parameters(), lr=config.learning_rate, weight_decay=1e-2)
        }
        # 设置学习率调度器，动态调整 step_size / Setup learning rate schedulers, dynamic step_size
        schedulers = {
            "feature": torch.optim.lr_scheduler.StepLR(
                optimizers["feature"],
                step_size=step_size,  # 动态调整 step_size / dynamic step_size
                gamma=0.8
            ),
            "relation": torch.optim.lr_scheduler.StepLR(
                optimizers["relation"],
                step_size=step_size,  # 动态调整 step_size / dynamic step_size
                gamma=0.8
            )
        }

        return optimizers, schedulers