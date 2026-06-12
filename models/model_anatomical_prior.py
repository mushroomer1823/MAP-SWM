# model_anatomical_prior.py
#
# SWM classifier with anatomical-prior conditioning.
#
# The mid (yeo, 7 networks) signal is NOT injected into the feature path
# (no cond_proj, no mid_embed). Instead, each atlas head's logits get an
# additive log-prior term derived from a fixed anatomical overlap matrix:
#
#     atlas_logits[A, pos] = atlas_head[A, pos](z)
#                          + alpha[A, pos] * log( softmax(mid_pos_logits / tau) @ M[A] )
#
# M[A] is a fixed (mid_dim, n_roi_A) matrix of P(atlas_roi | yeo_class),
# computed offline from voxel overlap on a common cortical region.
#
# mid_*_logits are detached when forming the prior so:
#   - atlas CE never flows back into mid_head or the backbone via the prior
#   - mid_head is supervised purely by its own CE (yeo label)
#   - backbone is supervised by the same losses as model_no_lobe_prior
#     (SWM + atlas CEs), so that baseline is a strict lower bound

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.pointnet import PointNetEncoder
from networks.pointnetpp import PointNetPPEncoder
from networks.DGCNN import DGCNNEncoder
from networks.pointMLP import PointMLPEncoder


class UnifiedSWMNet(nn.Module):
    def __init__(
        self,
        atlas_roi_dims,
        backbone,
        mid_dim=7,
        overlap_dir="/home/heyifei/codes/test/atlas_overlap",
        temperature=1.0,
        alpha_init=1.0,
        eps=1e-8,
    ):
        super().__init__()

        # ========= 1) backbone =========
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

        # ========= 2) endpoint MLP =========
        self.endpoint_dim = 256
        self.endpoint_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.endpoint_dim),
        )
        self.fused_dim = self.global_dim + 2 * self.endpoint_dim  # 1536

        # ========= 3) SWM / non-SWM binary head =========
        self.swm_head = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        # ========= 4) mid (yeo) heads — independent for start / end =========
        self.mid_dim = mid_dim
        self.mid_head_start = nn.Linear(self.fused_dim, mid_dim)
        self.mid_head_end = nn.Linear(self.fused_dim, mid_dim)

        # ========= 5) atlas heads (z → ROI directly, no bottleneck) =========
        self.atlas_heads = nn.ModuleDict()
        for atlas, n_roi in atlas_roi_dims.items():
            self.atlas_heads[atlas] = nn.ModuleDict({
                "start": nn.Linear(self.fused_dim, n_roi),
                "end": nn.Linear(self.fused_dim, n_roi),
            })

        # ========= 6) per-(atlas, pos) learnable prior weight alpha (A2) =====
        # Each of the 12 entries can drift independently; an atlas+pos for
        # which the prior is unhelpful will see its alpha shrink toward 0,
        # restoring the no-prior baseline for that head.
        self.alpha = nn.ParameterDict({
            f"{atlas}_{pos}": nn.Parameter(torch.tensor(float(alpha_init)))
            for atlas in atlas_roi_dims
            for pos in ["start", "end"]
        })

        # ========= 7) anatomical overlap matrices M (fixed buffers) ==========
        # M[A]: (mid_dim, n_roi_A); each row is P(atlas_roi | yeo_class)
        # estimated by voxel overlap on a common cortical region. Registered
        # as buffer so it moves with .to(device) and is included in state_dict.
        for atlas, n_roi in atlas_roi_dims.items():
            path = os.path.join(overlap_dir, f"M_yeo_to_{atlas}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Overlap matrix not found: {path}. "
                    f"Run atlas_overlap/compute_overlap.py first."
                )
            M = np.load(path)
            if M.shape != (mid_dim, n_roi):
                raise ValueError(
                    f"Shape mismatch for {atlas}: expected ({mid_dim}, {n_roi}), "
                    f"got {M.shape}"
                )
            self.register_buffer(f"M_{atlas}", torch.from_numpy(M.astype(np.float32)))

        # ========= 8) temperature and numerical floor ========================
        # Registered as buffers so they move with .to(device) and survive
        # checkpoint round-trips. Temperature defaults to 1.0 — set >1 to
        # soften the prior (less harsh when mid is wrong), <1 to sharpen.
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("eps", torch.tensor(float(eps)))

    def _prior_log_probs(self, mid_logits, atlas):
        """
        log P_prior(atlas_roi | mid prediction, anatomy) for one
        (mid head output, atlas) pair.

        mid_logits is detached so atlas CE does not pollute mid_head or
        backbone through this path.
        """
        mid_prob = F.softmax(mid_logits.detach() / self.temperature, dim=-1)
        prior = mid_prob @ getattr(self, f"M_{atlas}")    # (B, n_roi)
        return torch.log(prior + self.eps)

    def forward(self, fiber):
        """
        fiber: (B, 3, N)
        """
        global_feat = self.pointnet(fiber)                  # (B, 1024)

        start = fiber[:, :, 0]                              # (B, 3)
        end = fiber[:, :, -1]                               # (B, 3)
        start_feat = self.endpoint_mlp(start)               # (B, 256)
        end_feat = self.endpoint_mlp(end)                   # (B, 256)

        z = torch.cat([global_feat, start_feat, end_feat], dim=1)  # (B, 1536)

        swm_logits = self.swm_head(z)
        mid_start_logits = self.mid_head_start(z)
        mid_end_logits = self.mid_head_end(z)

        outputs = {
            "swm": swm_logits,
            "mid_start": mid_start_logits,
            "mid_end": mid_end_logits,
        }

        for atlas, heads in self.atlas_heads.items():
            log_prior_start = self._prior_log_probs(mid_start_logits, atlas)
            log_prior_end = self._prior_log_probs(mid_end_logits, atlas)

            raw_start = heads["start"](z)
            raw_end = heads["end"](z)

            outputs[f"{atlas}_start"] = (
                raw_start + self.alpha[f"{atlas}_start"] * log_prior_start
            )
            outputs[f"{atlas}_end"] = (
                raw_end + self.alpha[f"{atlas}_end"] * log_prior_end
            )

        return outputs

    def alpha_snapshot(self):
        """Return a plain dict of current alpha values for logging."""
        return {k: v.detach().cpu().item() for k, v in self.alpha.items()}
