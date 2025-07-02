import torch
import torch.nn.functional as F
import torch.nn as nn

class RelationNetwork(nn.Module):
    """
    关系网络，用于比较样本之间的相似度。
    Relation Network for comparing similarity between samples.
    结构：2层卷积 + 2层全连接。
    Structure: 2 convolutional layers + 2 fully connected layers.
    输入: 128通道（拼接的特征对），形状为128*8。
    Input: 128 channels (concatenated feature pairs), shape 128*8.
    输出: 0-1之间的相似度分数。
    Output: similarity score between 0 and 1.
    """

    def __init__(self,input_size, hidden_size):
        """
        初始化关系网络。
        Initialize Relation Network.
        Args:
            input_size: 输入特征维度 / input feature dimension
            hidden_size: 隐藏层维度 / hidden layer dimension
        """
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.MaxPool1d(4)
                        ) # 64x2

        self.layer2 = nn.Sequential(
                        nn.Conv1d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.MaxPool1d(2)) # 64x1

        self.fc1 = nn.Linear(input_size,hidden_size) # 64x1 -> 8
        
        self.fc2 = nn.Linear(hidden_size,1) # 8 -> 1

    def forward(self,x):
        """
        前向传播。
        Forward pass.
        Args:
            x: 输入特征对，形状(batch, 128, 8) / input feature pairs, shape (batch, 128, 8)
        Returns:
            相似度分数，形状(batch, 1) / similarity score, shape (batch, 1)
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        
        out = torch.sigmoid(self.fc2(out))
        return out