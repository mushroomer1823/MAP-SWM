# pointnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointMLPBlock(nn.Module):
    """
    PointMLP Residual Block
    """
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),

            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        """
        x: (B, C, N)
        """
        return self.act(x + self.mlp(x))
        
class PointMLPEncoder(nn.Module):
    """
    PointMLP-style encoder
    - deep per-point MLP
    - residual connections
    - symmetric max pooling

    Input : (B, 3, N)
    Output: (B, global_feat_dim)
    """
    def __init__(self, in_dim=3, global_feat_dim=1024):
        super().__init__()

        # ----- stem -----
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # ----- PointMLP blocks -----
        self.blocks = nn.Sequential(
            PointMLPBlock(128),
            PointMLPBlock(128),
            PointMLPBlock(128)
        )

        # ----- projection to global feature -----
        self.proj = nn.Sequential(
            nn.Conv1d(128, global_feat_dim, kernel_size=1),
            nn.BatchNorm1d(global_feat_dim),
            nn.ReLU()
        )

        self.out_dim = global_feat_dim

    def forward(self, x):
        """
        x: (B, 3, N)
        """
        x = self.stem(x)           # (B, 128, N)
        x = self.blocks(x)         # (B, 128, N)
        x = self.proj(x)           # (B, global_feat_dim, N)
        x = torch.max(x, dim=2)[0] # (B, global_feat_dim)
        return x