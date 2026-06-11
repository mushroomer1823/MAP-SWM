# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.pointnet import PointNetEncoder
from networks.pointnetpp import PointNetPPEncoder
from networks.DGCNN import DGCNNEncoder
from networks.pointMLP import PointMLPEncoder


class UnifiedSWMNet(nn.Module):
    """
    Hierarchical SWM classifier with a switchable intermediate ("mid") layer.

    Pipeline:
        fiber → backbone → z
        z → swm head (binary)
        z → mid_start / mid_end heads (mid_dim classes; supervised by either
            lobe labels or yeo labels depending on --mid_layer at the trainer
            side)
        softmax(mid_*).detach() → mid_mlp → mid_*_embed
        cat[z, mid_start_embed, mid_end_embed] → cond_proj → z_cond
        z_cond → per-atlas start/end heads

    mid_dim:
        14 when the trainer uses lobe labels as the mid supervision target
        7  when the trainer uses yeo labels as the mid supervision target
    """

    def __init__(self, atlas_roi_dims, backbone, mid_dim=14):
        super().__init__()

        # ========= 1) PointNet global feature =========
        backbone = backbone.lower()
        if backbone == "pointnet":
            self.pointnet = PointNetEncoder(in_dim=3, global_feat_dim=1024)
        elif backbone == "pointnet++":
            self.pointnet = PointNetPPEncoder(in_dim=3, global_feat_dim=1024)
        elif backbone == "dgcnn":
            self.pointnet = DGCNNEncoder(in_dim=3, global_feat_dim=1024)
        elif backbone == "pointmlp":
            self.pointnet = PointMLPEncoder(in_dim=3, global_feat_dim=1024)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.global_dim = 1024

        # ========= 2) endpoint feature MLP =========
        self.endpoint_dim = 256
        self.endpoint_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.endpoint_dim),
        )

        # fused dim = global + start + end
        self.fused_dim = self.global_dim + 2 * self.endpoint_dim  # 1536

        # ========= 3) SWM / non-SWM binary head =========
        self.swm_head = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        # ========= 4) Mid (lobe or yeo) prior heads =========
        self.mid_dim = mid_dim
        self.mid_start = nn.Linear(self.fused_dim, self.mid_dim)
        self.mid_end = nn.Linear(self.fused_dim, self.mid_dim)

        # ========= 5) Mid embedding projection =========
        self.mid_embed_dim = 256
        self.mid_mlp = nn.Sequential(
            nn.Linear(self.mid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.mid_embed_dim),
        )

        # ========= 6) Conditional projection =========
        self.cond_proj = nn.Sequential(
            nn.Linear(self.fused_dim + 2 * self.mid_embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        # ========= 7) Multi-atlas heads =========
        # Skip connection: each atlas head sees [z_cond, z] so the raw
        # backbone feature is never filtered through the cond_proj bottleneck.
        # If the mid signal turns out unhelpful, the head can ignore z_cond;
        # if it helps, the head still has full z capacity to combine with.
        self.atlas_head_in_dim = 256 + self.fused_dim
        self.atlas_heads = nn.ModuleDict()
        for atlas, n_roi in atlas_roi_dims.items():
            self.atlas_heads[atlas] = nn.ModuleDict({
                "start": nn.Linear(self.atlas_head_in_dim, n_roi),
                "end": nn.Linear(self.atlas_head_in_dim, n_roi),
            })

    def _build_mid_input(self, mid_logits, mid_target):
        """
        Build the mid feature fed into mid_mlp.

        Teacher forcing: when mid_target is provided (training path), valid
        labels (>= 0) are turned into one-hot vectors; invalid positions
        (IGNORE_INDEX, e.g. DWM samples) get zero vectors — matching the
        zero-mid behaviour of model_no_lobe_prior so cond_proj sees a clean,
        consistent signal across train/eval.

        When mid_target is None (eval path), fall back to the detached softmax
        of predicted mid logits.
        """
        if mid_target is None:
            return F.softmax(mid_logits, dim=-1).detach()

        valid = mid_target >= 0
        safe = mid_target.clamp(min=0)
        one_hot = F.one_hot(safe, num_classes=self.mid_dim).to(mid_logits.dtype)
        return torch.where(valid.unsqueeze(-1), one_hot, torch.zeros_like(one_hot))

    def forward(self, fiber, mid_start_target=None, mid_end_target=None):
        """
        fiber: (B, 3, N)

        mid_start_target / mid_end_target: optional (B,) long tensors.
            When provided (training), teacher-force the mid input to atlas
            heads using ground-truth labels. IGNORE_INDEX positions get zero
            mid embeddings.
            When None (eval), use detached softmax of predicted mid logits.
        """

        global_feat = self.pointnet(fiber)            # (B, 1024)

        start = fiber[:, :, 0]                        # (B, 3)
        end = fiber[:, :, -1]                         # (B, 3)
        start_feat = self.endpoint_mlp(start)         # (B, 256)
        end_feat = self.endpoint_mlp(end)             # (B, 256)

        z = torch.cat([global_feat, start_feat, end_feat], dim=1)  # (B, 1536)

        swm_logits = self.swm_head(z)

        # Detach z for the mid heads: mid CE still trains the mid_start /
        # mid_end Linear weights, but its gradient no longer reaches the
        # backbone. The backbone is now optimized by exactly the same losses
        # as model_no_lobe_prior (SWM + atlas CEs), so the lobe variant cannot
        # be hurt by mid pulling backbone capacity.
        z_for_mid = z.detach()
        mid_start_logits = self.mid_start(z_for_mid)
        mid_end_logits = self.mid_end(z_for_mid)

        mid_start_input = self._build_mid_input(mid_start_logits, mid_start_target)
        mid_end_input = self._build_mid_input(mid_end_logits, mid_end_target)

        mid_start_embed = self.mid_mlp(mid_start_input)
        mid_end_embed = self.mid_mlp(mid_end_input)

        z_cond_in = torch.cat([z, mid_start_embed, mid_end_embed], dim=1)
        z_cond = self.cond_proj(z_cond_in)

        # Skip connection: atlas heads see both the conditioned feature and
        # the raw backbone z, so the cond_proj bottleneck cannot strip atlas-
        # relevant info from z.
        atlas_in = torch.cat([z_cond, z], dim=1)

        outputs = {
            "swm": swm_logits,
            "mid_start": mid_start_logits,
            "mid_end": mid_end_logits,
        }

        for atlas, heads in self.atlas_heads.items():
            outputs[f"{atlas}_start"] = heads["start"](atlas_in)
            outputs[f"{atlas}_end"] = heads["end"](atlas_in)

        return outputs
