# pointnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseMLP(nn.Module):
    """
    PointNet-style shared MLP:
    implemented as Conv1d(kernel=1)
    """
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        prev_dim = in_dim

        for h in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, h, kernel_size=1),
                nn.BatchNorm1d(h),
                nn.ReLU()
            ])
            prev_dim = h

        layers.extend([
            nn.Conv1d(prev_dim, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, C, N)
        """
        return self.net(x)


class PointNetEncoder(nn.Module):
    """
    PointNet encoder:
    - shared point-wise MLP
    - symmetric max pooling
    """
    def __init__(self, in_dim=3, global_feat_dim=1024):
        super().__init__()

        self.point_mlp = PointwiseMLP(
            in_dim=in_dim,
            hidden_dims=[64, 128],
            out_dim=global_feat_dim
        )
        
        self.out_dim = global_feat_dim

    def forward(self, x):
        """
        x: (B, 3, N)
        """
        # ----- point-wise feature -----
        x = self.point_mlp(x)              # (B, 1024, N)

        # ----- symmetric aggregation -----
        x = torch.max(x, dim=2)[0]          # (B, 1024)

        return x

