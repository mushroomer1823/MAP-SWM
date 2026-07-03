# model_anatomical_prior.py
#
# Unified SWM multi-atlas classifier with a Light Anatomical Bottleneck option.
#
# Recommended lightweight configuration:
#   endpoint_usage="mid_only"
#   classifier_head="prototype"
#   prior_mode="adapter"
#
# In this setting, endpoint features are used only to predict the intermediate
# anatomical labels (Yeo or lobe). The final multi-atlas heads do not directly
# consume endpoint features. Instead, the final heads receive:
#   1) the streamline global feature, and
#   2) a compact mid-layer anatomical embedding derived from mid probabilities.
# A small learnable prior adapter then produces a residual logit correction from
# the mid prediction and the fixed anatomical overlap matrix.

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.pointnet import PointNetEncoder
from networks.pointnetpp import PointNetPPEncoder
from networks.DGCNN import DGCNNEncoder
from networks.pointMLP import PointMLPEncoder


class PrototypeClassifier(nn.Module):
    """Compact cosine prototype classifier.

    It replaces a large Linear(in_dim, num_classes) head with:
        feature projection: in_dim -> proto_dim
        class prototypes:   num_classes x proto_dim
        cosine similarity in prototype space

    This is useful for lightweight multi-class classification because the
    parameter cost scales with (in_dim + num_classes) * proto_dim instead of
    in_dim * num_classes.
    """

    def __init__(self, in_dim: int, num_classes: int, proto_dim: int = 128,
                 dropout: float = 0.1, scale: float = 16.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, proto_dim),
            nn.GELU(),
            nn.LayerNorm(proto_dim),
        )
        self.prototypes = nn.Parameter(torch.empty(num_classes, proto_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(scale))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = F.normalize(z, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        # clamp scale for stability; exp(log_scale) remains trainable.
        scale = torch.exp(self.log_scale).clamp(1.0, 100.0)
        return scale * F.linear(z, p)


class CosineClassifier(nn.Module):
    """Cosine classifier with the same order of parameters as Linear."""

    def __init__(self, in_dim: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(scale))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        scale = torch.exp(self.log_scale).clamp(1.0, 100.0)
        return scale * F.linear(x, w)


class PriorAdapter(nn.Module):
    """Learnable anatomical prior adapter.

    Input:  concat(mid_prob, log_overlap_prior)
    Output: residual delta logits for one atlas endpoint head.
    """

    def __init__(self, mid_dim: int, n_roi: int, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(mid_dim + n_roi),
            nn.Dropout(dropout),
            nn.Linear(mid_dim + n_roi, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_roi),
        )
        # Start as a very small residual correction. The gate controls magnitude,
        # but this makes early training even safer.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mid_prob: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([mid_prob, log_prior], dim=-1))


class UnifiedSWMNet(nn.Module):
    def __init__(
        self,
        atlas_roi_dims: Dict[str, int],
        backbone: str,
        mid_dim: int = 7,
        mid_source: str = "yeo",
        overlap_dir: str = "./atlas_overlap",
        temperature: float = 1.0,
        gate_init: float = 0.0,
        global_feat_dim: int = 512,
        endpoint_dim: int = 64,
        swm_hidden_dim: int = 256,
        endpoint_usage: str = "mid_only",
        mid_embed_dim: int = 64,
        classifier_head: str = "prototype",
        proto_dim: int = 128,
        head_dropout: float = 0.1,
        prior_mode: str = "adapter",
        prior_hidden_dim: int = 64,
        prior_dropout: float = 0.1,
        detach_prior: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()

        endpoint_usage = endpoint_usage.lower()
        if endpoint_usage not in {"all", "mid_only", "none"}:
            raise ValueError("endpoint_usage must be one of: all, mid_only, none")
        classifier_head = classifier_head.lower()
        if classifier_head not in {"linear", "cosine", "prototype"}:
            raise ValueError("classifier_head must be one of: linear, cosine, prototype")
        prior_mode = prior_mode.lower()
        if prior_mode not in {"none", "overlap_log", "adapter", "hybrid"}:
            raise ValueError("prior_mode must be one of: none, overlap_log, adapter, hybrid")

        self.atlas_roi_dims = dict(atlas_roi_dims)
        self.global_dim = int(global_feat_dim)
        self.endpoint_dim = int(endpoint_dim)
        self.swm_hidden_dim = int(swm_hidden_dim)
        self.mid_dim = int(mid_dim)
        self.mid_source = str(mid_source)
        self.endpoint_usage = endpoint_usage
        self.mid_embed_dim = int(mid_embed_dim)
        self.classifier_head = classifier_head
        self.prior_mode = prior_mode
        self.detach_prior = bool(detach_prior)
        self.prior_enabled = prior_mode != "none"

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

        # ========= 2) endpoint encoder =========
        if self.endpoint_usage != "none":
            self.endpoint_mlp = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.endpoint_dim),
            )
        else:
            self.endpoint_mlp = None

        self.full_fused_dim = self.global_dim + 2 * self.endpoint_dim
        self.per_endpoint_mid_dim = self.global_dim + self.endpoint_dim

        if self.endpoint_usage == "all":
            self.swm_input_dim = self.full_fused_dim
            self.mid_input_dim = self.full_fused_dim
            self.final_input_dim = self.full_fused_dim
        elif self.endpoint_usage == "mid_only":
            # Endpoint features do not directly enter SWM or final atlas heads.
            self.swm_input_dim = self.global_dim
            self.mid_input_dim = self.per_endpoint_mid_dim
            self.final_input_dim = self.global_dim + self.mid_embed_dim
        else:
            self.swm_input_dim = self.global_dim
            self.mid_input_dim = self.global_dim
            self.final_input_dim = self.global_dim

        # Exposed for the trainer's print statements.
        self.fused_dim = self.final_input_dim

        # ========= 3) SWM / non-SWM binary head =========
        self.swm_head = nn.Sequential(
            nn.Linear(self.swm_input_dim, self.swm_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.swm_hidden_dim, 2),
        )

        # ========= 4) mid anatomical heads =========
        self.mid_head_start = nn.Linear(self.mid_input_dim, self.mid_dim)
        self.mid_head_end = nn.Linear(self.mid_input_dim, self.mid_dim)

        # Convert mid probabilities into compact anatomical embeddings for final heads.
        if self.endpoint_usage == "mid_only":
            self.mid_embed_start = nn.Sequential(
                nn.Linear(self.mid_dim, self.mid_embed_dim),
                nn.GELU(),
                nn.LayerNorm(self.mid_embed_dim),
            )
            self.mid_embed_end = nn.Sequential(
                nn.Linear(self.mid_dim, self.mid_embed_dim),
                nn.GELU(),
                nn.LayerNorm(self.mid_embed_dim),
            )
        else:
            self.mid_embed_start = None
            self.mid_embed_end = None

        # ========= 5) final atlas heads =========
        self.atlas_heads = nn.ModuleDict()
        for atlas, n_roi in self.atlas_roi_dims.items():
            self.atlas_heads[atlas] = nn.ModuleDict({
                "start": self._make_classifier(self.final_input_dim, n_roi, proto_dim, head_dropout),
                "end": self._make_classifier(self.final_input_dim, n_roi, proto_dim, head_dropout),
            })

        # ========= 6) overlap matrices and prior adapters =========
        for atlas, n_roi in self.atlas_roi_dims.items():
            path = os.path.join(overlap_dir, f"M_{self.mid_source}_to_{atlas}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Overlap matrix not found: {path}. "
                    f"Run atlas_overlap/compute_overlap.py --source {self.mid_source} first."
                )
            M = np.load(path)
            if M.shape != (self.mid_dim, n_roi):
                raise ValueError(
                    f"Shape mismatch for {atlas}: expected ({self.mid_dim}, {n_roi}), got {M.shape}"
                )
            self.register_buffer(f"M_{atlas}", torch.from_numpy(M.astype(np.float32)))

        self.prior_gates = nn.ParameterDict({
            f"{atlas}_{pos}": nn.Parameter(torch.tensor(float(gate_init)))
            for atlas in self.atlas_roi_dims
            for pos in ["start", "end"]
        })

        self.prior_adapters = nn.ModuleDict()
        if prior_mode in {"adapter", "hybrid"}:
            for atlas, n_roi in self.atlas_roi_dims.items():
                self.prior_adapters[atlas] = nn.ModuleDict({
                    "start": PriorAdapter(self.mid_dim, n_roi, prior_hidden_dim, prior_dropout),
                    "end": PriorAdapter(self.mid_dim, n_roi, prior_hidden_dim, prior_dropout),
                })

        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("eps", torch.tensor(float(eps)))

    def _make_classifier(self, in_dim: int, n_cls: int, proto_dim: int, dropout: float) -> nn.Module:
        if self.classifier_head == "linear":
            return nn.Linear(in_dim, n_cls)
        if self.classifier_head == "cosine":
            return CosineClassifier(in_dim, n_cls)
        if self.classifier_head == "prototype":
            return PrototypeClassifier(in_dim, n_cls, proto_dim=proto_dim, dropout=dropout)
        raise RuntimeError("unreachable")

    def set_prior_enabled(self, enabled: bool):
        self.prior_enabled = bool(enabled)
        if not enabled:
            with torch.no_grad():
                for p in self.prior_gates.values():
                    p.fill_(-50.0)
            for p in self.prior_gates.values():
                p.requires_grad = False
            for p in self.prior_adapters.parameters():
                p.requires_grad = False

    def _compute_mid_prob_and_log_prior(self, mid_logits: torch.Tensor, atlas: str):
        src = mid_logits.detach() if self.detach_prior else mid_logits
        mid_prob = F.softmax(src / self.temperature, dim=-1)
        prior = mid_prob @ getattr(self, f"M_{atlas}")
        log_prior = torch.log(prior + self.eps)
        return mid_prob, log_prior

    def _mid_embedding(self, mid_logits: torch.Tensor, pos: str) -> torch.Tensor:
        # If prior is disabled in mid_only mode, return a zero bottleneck so no
        # endpoint-derived information can leak into final heads.
        if not self.prior_enabled:
            batch = mid_logits.shape[0]
            return mid_logits.new_zeros(batch, self.mid_embed_dim)
        src = mid_logits.detach() if self.detach_prior else mid_logits
        mid_prob = F.softmax(src / self.temperature, dim=-1)
        if pos == "start":
            return self.mid_embed_start(mid_prob)
        return self.mid_embed_end(mid_prob)

    def _apply_prior(self, raw_logits: torch.Tensor, mid_logits: torch.Tensor,
                     atlas: str, pos: str) -> torch.Tensor:
        if (not self.prior_enabled) or self.prior_mode == "none":
            return raw_logits

        mid_prob, log_prior = self._compute_mid_prob_and_log_prior(mid_logits, atlas)
        if self.prior_mode == "overlap_log":
            delta = log_prior
        elif self.prior_mode == "adapter":
            delta = self.prior_adapters[atlas][pos](mid_prob, log_prior)
        elif self.prior_mode == "hybrid":
            delta = log_prior + self.prior_adapters[atlas][pos](mid_prob, log_prior)
        else:
            return raw_logits

        gate = torch.sigmoid(self.prior_gates[f"{atlas}_{pos}"])
        return raw_logits + gate * delta

    def forward(self, fiber: torch.Tensor):
        """fiber: (B, 3, N)"""
        global_feat = self.pointnet(fiber)

        start_feat = end_feat = None
        if self.endpoint_usage != "none":
            start = fiber[:, :, 0]
            end = fiber[:, :, -1]
            start_feat = self.endpoint_mlp(start)
            end_feat = self.endpoint_mlp(end)

        # ----- SWM feature -----
        if self.endpoint_usage == "all":
            swm_feat = torch.cat([global_feat, start_feat, end_feat], dim=1)
        else:
            swm_feat = global_feat
        swm_logits = self.swm_head(swm_feat)

        # ----- Mid-layer features -----
        if self.endpoint_usage == "all":
            mid_feat_start = torch.cat([global_feat, start_feat, end_feat], dim=1)
            mid_feat_end = mid_feat_start
        elif self.endpoint_usage == "mid_only":
            mid_feat_start = torch.cat([global_feat, start_feat], dim=1)
            mid_feat_end = torch.cat([global_feat, end_feat], dim=1)
        else:
            mid_feat_start = global_feat
            mid_feat_end = global_feat

        mid_start_logits = self.mid_head_start(mid_feat_start)
        mid_end_logits = self.mid_head_end(mid_feat_end)

        # ----- Final atlas features -----
        if self.endpoint_usage == "all":
            final_feat_start = torch.cat([global_feat, start_feat, end_feat], dim=1)
            final_feat_end = final_feat_start
        elif self.endpoint_usage == "mid_only":
            z_mid_start = self._mid_embedding(mid_start_logits, "start")
            z_mid_end = self._mid_embedding(mid_end_logits, "end")
            final_feat_start = torch.cat([global_feat, z_mid_start], dim=1)
            final_feat_end = torch.cat([global_feat, z_mid_end], dim=1)
        else:
            final_feat_start = global_feat
            final_feat_end = global_feat

        outputs = {
            "swm": swm_logits,
            "mid_start": mid_start_logits,
            "mid_end": mid_end_logits,
        }

        for atlas, heads in self.atlas_heads.items():
            raw_start = heads["start"](final_feat_start)
            raw_end = heads["end"](final_feat_end)
            outputs[f"{atlas}_start"] = self._apply_prior(raw_start, mid_start_logits, atlas, "start")
            outputs[f"{atlas}_end"] = self._apply_prior(raw_end, mid_end_logits, atlas, "end")

        return outputs

    def gate_snapshot(self):
        out = {}
        for k, v in self.prior_gates.items():
            raw = float(v.detach().cpu().item())
            out[k] = {"raw": raw, "sigmoid": float(torch.sigmoid(v.detach()).cpu().item())}
        return out

    def prior_weight_snapshot(self):
        return {
            k: float(torch.sigmoid(v.detach()).cpu().item())
            for k, v in self.prior_gates.items()
        }
