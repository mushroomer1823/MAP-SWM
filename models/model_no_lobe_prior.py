# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.pointnet import PointNetEncoder
from networks.pointnetpp import PointNetPPEncoder
from networks.DGCNN import DGCNNEncoder
from networks.pointMLP import PointMLPEncoder

class UnifiedSWMNet(nn.Module):
    def __init__(self, atlas_roi_dims, backbone):
        super().__init__()

        # ========= 1) PointNet global feature =========
        backbone = backbone.lower()
        if backbone == "pointnet":
            # basic pointnet
            self.pointnet = PointNetEncoder(
                in_dim=3,
                global_feat_dim=1024
            )
        elif backbone == "pointnet++":
            # pointnet++
            self.pointnet = PointNetPPEncoder(
                in_dim=3,
                global_feat_dim=1024
            )
        elif backbone == "dgcnn":
            # DGCNN
            self.pointnet = DGCNNEncoder(
                in_dim=3,
                global_feat_dim=1024
            )
        elif backbone == "pointmlp":
            # pointMLP
            self.pointnet = PointMLPEncoder(
                in_dim=3,
                global_feat_dim=1024
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.global_dim=1024

        # ========= 2) endpoint feature MLP =========
        self.endpoint_dim = 256
        self.endpoint_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.endpoint_dim)
        )

        # fused dim = global + start + end
        self.fused_dim = self.global_dim + 2*self.endpoint_dim   # 1024 + 256*2 = 1536

        # ========= 3) SWM / non-SWM binary head =========
        self.swm_head = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        # ========= 6) conditional projection =========
        # Capacity-matched to model_lobe.UnifiedSWMNet: cond_proj input dim and
        # parameter count are identical; the 2*mid_embed_dim slots are fed
        # with zeros so the only difference vs. the lobe/yeo mid variant is
        # whether the mid signal itself is present.
        self.mid_embed_dim = 256
        self.cond_proj = nn.Sequential(
            nn.Linear(self.fused_dim + 2 * self.mid_embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        # ========= 7) Multi-atlas heads =========
        self.atlas_heads = nn.ModuleDict()
        for atlas, n_roi in atlas_roi_dims.items():
            self.atlas_heads[atlas] = nn.ModuleDict({
                "start": nn.Linear(256, n_roi),
                "end": nn.Linear(256, n_roi),
            })
            


    def forward(self, fiber):
        """
        fiber: (B, 3, N)
        """

        # ---------- PointNet global ----------
        global_feat = self.pointnet(fiber)     # (B, 1024)
        # print(global_feat.shape)

        # ---------- endpoints ----------
        start = fiber[:, :, 0]                 # (B, 3)
        end   = fiber[:, :, -1]                # (B, 3)

        start_feat = self.endpoint_mlp(start)  # (B, 256)
        end_feat   = self.endpoint_mlp(end)    # (B, 256)

        # ---------- concat ----------
        z = torch.cat([global_feat, start_feat, end_feat], dim=1)   # (B, 1024)

        # ---------- SWM binary ----------
        swm_logits = self.swm_head(z)

        zero_mid = torch.zeros(
            z.size(0), 2 * self.mid_embed_dim,
            device=z.device, dtype=z.dtype,
        )
        z_cond = self.cond_proj(torch.cat([z, zero_mid], dim=1))

        outputs = {
            "swm": swm_logits,
        }

        # ---------- Multi-atlas ----------
        for atlas, heads in self.atlas_heads.items():
            outputs[f"{atlas}_start"] = heads["start"](z_cond)
            outputs[f"{atlas}_end"]   = heads["end"](z_cond)
            

        return outputs