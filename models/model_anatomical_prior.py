# model_anatomical_prior.py
#
# SWM classifier with anatomical-prior conditioning.
#
# This version injects the anatomical prior as a gated residual refinement:
#
#     atlas_logits[A, pos] = base_head[A, pos](z)
#                          + sigmoid(gate[A, pos]) * alpha[A, pos]
#                            * log( softmax(mid_pos_logits / tau) @ M[A] )
#
# The direct streamline + endpoint path is therefore always preserved.  If the
# anatomical prior is unhelpful for a given atlas/position head, the learnable
# gate can stay close to 0 and the model degenerates to the no-prior baseline.
#
# M[A] is a fixed (mid_dim, n_roi_A) matrix of P(atlas_roi | yeo_class),
# computed offline from voxel overlap on a common cortical region.
#
# mid_*_logits are detached when forming the prior so:
#   - atlas CE never flows back into mid_head or the backbone via the prior
#   - mid_head is supervised purely by its own CE (yeo label)
#   - backbone still receives gradients from SWM + atlas CEs through the direct
#     base_head path.

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
        gate_init=-6.0,
        global_feat_dim=1024,
        endpoint_dim=256,
        swm_hidden_dim=256,
        eps=1e-8,
    ):
        super().__init__()

        self.global_dim = int(global_feat_dim)
        self.endpoint_dim = int(endpoint_dim)
        self.swm_hidden_dim = int(swm_hidden_dim)

        # ========= 1) backbone =========
        backbone = backbone.lower()
        if backbone == "pointnet":
            self.pointnet = PointNetEncoder(in_dim=3, global_feat_dim=self.global_dim)
        elif backbone == "pointnet++":
            self.pointnet = PointNetPPEncoder(in_dim=3, global_feat_dim=self.global_dim)
        elif backbone == "dgcnn":
            self.pointnet = DGCNNEncoder(in_dim=3, global_feat_dim=self.global_dim)
        elif backbone == "pointmlp":
            self.pointnet = PointMLPEncoder(in_dim=3, global_feat_dim=self.global_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ========= 2) endpoint MLP =========
        self.endpoint_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.endpoint_dim),
        )
        self.fused_dim = self.global_dim + 2 * self.endpoint_dim

        # ========= 3) SWM / non-SWM binary head =========
        self.swm_head = nn.Sequential(
            nn.Linear(self.fused_dim, self.swm_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.swm_hidden_dim, 2),
        )

        # ========= 4) mid (yeo) heads — independent for start / end =========
        self.mid_dim = mid_dim
        self.mid_head_start = nn.Linear(self.fused_dim, mid_dim)
        self.mid_head_end = nn.Linear(self.fused_dim, mid_dim)

        # ========= 5) direct atlas heads: baseline path z -> ROI =========
        self.atlas_heads = nn.ModuleDict()
        for atlas, n_roi in atlas_roi_dims.items():
            self.atlas_heads[atlas] = nn.ModuleDict({
                "start": nn.Linear(self.fused_dim, n_roi),
                "end": nn.Linear(self.fused_dim, n_roi),
            })

        # ========= 6) gated residual prior weights ==========================
        # alpha controls residual magnitude/sign; gate controls whether the
        # anatomical prior is used. gate_init=-6 gives sigmoid(gate)≈0.0025,
        # so the model starts very close to the direct no-prior baseline.
        self.alpha = nn.ParameterDict({
            f"{atlas}_{pos}": nn.Parameter(torch.tensor(float(alpha_init)))
            for atlas in atlas_roi_dims
            for pos in ["start", "end"]
        })
        self.gate = nn.ParameterDict({
            f"{atlas}_{pos}": nn.Parameter(torch.tensor(float(gate_init)))
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

    def _apply_gated_residual_prior(self, raw_logits, log_prior, atlas, pos):
        key = f"{atlas}_{pos}"
        gate = torch.sigmoid(self.gate[key])
        return raw_logits + gate * self.alpha[key] * log_prior

    def forward(self, fiber):
        """
        fiber: (B, 3, N)
        """
        global_feat = self.pointnet(fiber)                  # (B, global_dim)

        start = fiber[:, :, 0]                              # (B, 3)
        end = fiber[:, :, -1]                               # (B, 3)
        start_feat = self.endpoint_mlp(start)               # (B, endpoint_dim)
        end_feat = self.endpoint_mlp(end)                   # (B, endpoint_dim)

        z = torch.cat([global_feat, start_feat, end_feat], dim=1)

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

            outputs[f"{atlas}_start"] = self._apply_gated_residual_prior(
                raw_logits=raw_start,
                log_prior=log_prior_start,
                atlas=atlas,
                pos="start",
            )
            outputs[f"{atlas}_end"] = self._apply_gated_residual_prior(
                raw_logits=raw_end,
                log_prior=log_prior_end,
                atlas=atlas,
                pos="end",
            )

        return outputs

    def alpha_snapshot(self):
        """Return a plain dict of current alpha values for logging."""
        return {k: v.detach().cpu().item() for k, v in self.alpha.items()}

    def gate_snapshot(self):
        """Return raw gate and sigmoid(gate) values for logging."""
        out = {}
        for k, v in self.gate.items():
            raw = v.detach().cpu().item()
            out[k] = {"raw": raw, "sigmoid": float(torch.sigmoid(v.detach()).cpu().item())}
        return out

    def prior_weight_snapshot(self):
        """Return effective prior weights: sigmoid(gate) * alpha."""
        out = {}
        for k in self.alpha.keys():
            out[k] = (
                torch.sigmoid(self.gate[k].detach()) * self.alpha[k].detach()
            ).cpu().item()
        return out
