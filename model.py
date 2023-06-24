from torch_geometric.nn import GAE, GCNConv, GATConv
import torch
import torch.nn.functional as F
from utils import *


# 单层GCN编码器
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        return self.conv1(x, edge_index).relu()


# 单层GCN编码器_scale
class GCNEncoder_scale(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder_scale, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, cached=True) # cached only for transductive learning

    # 按行取最大值最小值
    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled

    def forward(self, x, edge_index):
        torch.set_printoptions(profile="full")
        h = self.conv1(x, edge_index)
        h = self.scale(h)
        h = F.normalize(h)
        return h


# 单层GCN编码器_scale_大度权重
class GCNEncoder_scale_degree(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder_scale_degree, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, cached=True) # cached only for transductive learning
        self.ac = nn.ReLU()

    # 按行取最大值最小值
    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        # h = self.ac(h)
        h = self.scale(h)
        h = F.normalize(h)
        return h


# 双层GCN编码器
class GCNEncoder2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder2, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量
        # cached 因为我们只有一张图
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# 双层GCN编码器——归一化
class GCNEncoder2_scale(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder2_scale, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量
        # cached 因为我们只有一张图
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        h = self.scale(h)
        h = F.normalize(h)
        return h


# 三层GCN编码器
class GCNEncoder3(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder3, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量
        # cached 因为我们只有一张图
        self.conv1 = GCNConv(in_channels, 4 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(4 * out_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv3 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)


# 单层线性编码器
class GCNEncoder4(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder4, self).__init__()
        self.conv1 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv1(x).relu()


# 单层线性编码器
class GCNEncoder4_scale(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder4_scale, self).__init__()
        self.conv1 = torch.nn.Linear(in_channels, out_channels)

    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled

    def forward(self, x, edge_index):
        h = self.conv1(x)
        h = self.scale(h)
        h = F.normalize(h)
        return h


# 损失函数
class PairLoss(nn.Module):
    def forward(self, h, pos, neg):
        y = 0.5

        norm_pos = torch.tensor([0.0]).cuda()
        for i in pos:
            norm_pos += torch.norm(h[i[0]] - h[i[1]])

        norm_neg = torch.tensor([0.0]).cuda()
        for i in neg:
            norm_neg += max(0, y - torch.norm(h[i[0]] - h[i[1]]))

        loss = norm_pos + norm_neg

        print("loss compute finished：" + str(loss.item()))
        print("pos_loss:", norm_pos)
        print("neg_loss:", norm_neg)
        return loss

