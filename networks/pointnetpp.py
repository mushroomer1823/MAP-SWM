# pointnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """
    x: (B, 3, N)
    return: idx (B, N, k)
    """
    B, _, N = x.shape
    x = x.transpose(2, 1)  # (B, N, 3)

    dist = torch.cdist(x, x)          # (B, N, N)
    _, idx = dist.topk(k=k, dim=-1, largest=False)
    return idx

class SimplePointNetSA(nn.Module):
    """
    Simplified Set Abstraction:
    - kNN grouping
    - local PointNet
    """
    def __init__(self, in_dim, out_dim, k=8):
        super().__init__()
        self.k = k

        self.local_mlp = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (B, C, N)
        return: (B, out_dim, N)
        """
        B, C, N = x.shape
        idx = knn(x[:, :3, :], self.k)      # use xyz for neighbor search

        idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
        neighbors = torch.gather(
            x.unsqueeze(-1).expand(-1, -1, -1, self.k),
            2,
            idx
        )                                   # (B, C, N, k)

        # local PointNet
        local_feat = self.local_mlp(neighbors)   # (B, out_dim, N, k)
        local_feat = torch.max(local_feat, dim=-1)[0]  # (B, out_dim, N)

        return local_feat

class PointNetPPEncoder(nn.Module):
    """
    Simplified PointNet++ Encoder
    Input : (B, 3, N)
    Output: (B, global_feat_dim)
    """
    def __init__(self, in_dim=3, global_feat_dim=1024, k=8):
        super().__init__()

        # ----- local abstraction -----
        self.sa1 = SimplePointNetSA(
            in_dim=in_dim,
            out_dim=128,
            k=k
        )

        # ----- global abstraction (PointNet-style) -----
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
        # local features
        x = self.sa1(x)                    # (B, 128, N)

        # global features
        x = self.global_mlp(x)             # (B, global_feat_dim, N)
        x = torch.max(x, dim=2)[0]         # (B, global_feat_dim)

        return x

