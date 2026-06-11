import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """
    x: (B, C, N)
    return: idx (B, N, k)
    """
    B, C, N = x.shape
    x = x.transpose(2, 1)  # (B, N, C)
    dist = torch.cdist(x, x)  # (B, N, N)
    _, idx = dist.topk(k=k, dim=-1, largest=False)
    return idx

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=4):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, C, N)
        """
        B, C, N = x.shape
        idx = knn(x[:, :3, :], self.k)  # use xyz for neighbor search

        idx = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N, k)
        neighbors = torch.gather(
            x.unsqueeze(-1).expand(-1, -1, -1, self.k),
            2,
            idx
        )  # (B, C, N, k)

        # edge feature: concat (x_i, x_j - x_i)
        x_central = x.unsqueeze(-1).expand(-1, -1, -1, self.k)  # (B, C, N, k)
        edge_feat = torch.cat([x_central, neighbors - x_central], dim=1)  # (B, 2C, N, k)

        # MLP
        out = self.mlp(edge_feat)  # (B, out_channels, N, k)
        out = torch.max(out, dim=-1)[0]  # (B, out_channels, N)
        return out

# ---------------------
# DGCNN Encoder
# ---------------------
class DGCNNEncoder(nn.Module):
    """
    Simplified DGCNN Encoder
    Input : (B, 3, N)
    Output: (B, global_feat_dim)
    """
    def __init__(self, in_dim=3, global_feat_dim=1024, k=4):
        super().__init__()
        self.k = k

        # EdgeConv layers
        self.edge1 = EdgeConv(in_channels=in_dim, out_channels=64, k=k)
        self.edge2 = EdgeConv(in_channels=64, out_channels=128, k=k)

        # final MLP to global feature
        self.global_mlp = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, global_feat_dim, kernel_size=1),
            nn.BatchNorm1d(global_feat_dim),
            nn.ReLU()
        )

        self.out_dim = global_feat_dim

    def forward(self, x):
        """
        x: (B, 3, N)
        """
        x = self.edge1(x)       # (B, 64, N)
        x = self.edge2(x)       # (B, 128, N)
        x = self.global_mlp(x)  # (B, global_feat_dim, N)
        x = torch.max(x, dim=2)[0]  # (B, global_feat_dim)
        return x

