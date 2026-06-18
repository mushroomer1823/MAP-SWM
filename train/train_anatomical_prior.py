# train_anatomical_prior.py
#
# Trainer for model_anatomical_prior.UnifiedSWMNet.
#
# Main features:
#   1. Uses anatomical prior as a gated residual refinement in the model.
#   2. Supports configurable feature dimensions through --global_feat_dim and
#      --endpoint_dim, so changing 1024 -> 512 does not require editing model code.
#   3. Saves every run under /data/hyf/swm_identification/train_result/<param-tag>/,
#      including config, best/final checkpoints, per-epoch metrics, test metrics,
#      convergence curves, parameter statistics, and learned gate values.

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

# Avoid "OSError: [Errno 24] too many open files" when running k-fold with
# many DataLoader workers. PyTorch's default tensor-sharing strategy
# ('file_descriptor') opens one fd per shared tensor; with num_workers=64,
# prefetch_factor=4, and 3 simultaneously-alive loaders (train/val/test),
# the fd count blows past the default ulimit (1024). Switching to
# 'file_system' uses /dev/shm-backed tmpfs files instead — minor IPC slowdown,
# but no fd explosion. Must be set BEFORE any DataLoader spawns workers.
torch.multiprocessing.set_sharing_strategy("file_system")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from dataset_demo_kfold import (  # noqa: E402
    build_demo_datasets,
    build_demo_kfold_datasets,
    ATLAS_LIST,
)
from models.model_anatomical_prior import UnifiedSWMNet  # noqa: E402


# =====================================================
# Args
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train UnifiedSWMNet with gated-residual anatomical-prior conditioning. "
            "Mid layer can be yeo (7 networks) or lobe (14 hemi-lobes) via --mid_layer; "
            "the corresponding M_{mid_layer}_to_{atlas}.npy overlap matrices are loaded "
            "from {overlap_dir}/{mid_layer}/."
        )
    )

    # ===== data =====
    parser.add_argument(
        "--h5_path", type=str,
        default="/data/hyf/swm_identification/data/demo/All_swm_dwm_streamlines.h5",
    )
    parser.add_argument(
        "--csv_path", type=str,
        default="/data/hyf/swm_identification/data/demo/All_swm_dwm_atlas_start_end_selected_with_lobe.csv",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # ===== K-fold =====
    parser.add_argument(
        "--k_fold", type=int, default=1,
        help="Number of folds. <=1 uses the original single train/val/test split.",
    )
    parser.add_argument(
        "--stratify_by_swm",
        action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--group_by_subject",
        action=argparse.BooleanOptionalAction, default=True,
    )

    # ===== training =====
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument(
        "--early_stop_patience", type=int, default=10,
        help=(
            "Stop training if val_loss does not improve for this many "
            "consecutive epochs. Set to 0 (or any value <=0) to disable."
        ),
    )

    # ===== device =====
    parser.add_argument("--gpu", type=int, default=2)

    parser.add_argument(
        "--model_type", type=str, default="dgcnn",
        choices=["pointnet", "pointnet++", "pointmlp", "dgcnn"],
    )

    # ===== dimension control =====
    parser.add_argument(
        "--global_feat_dim", type=int, default=1024,
        help="Backbone global feature dimension, e.g. 1024 or 512.",
    )
    parser.add_argument(
        "--endpoint_dim", type=int, default=256,
        help="Endpoint feature dimension after endpoint_mlp. Ignored when --no-use_endpoint is set.",
    )
    parser.add_argument(
        "--use_endpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to concatenate start/end endpoint features with the "
            "streamline backbone feature. Use --no-use_endpoint for the "
            "streamline-only baseline without endpoint features."
        ),
    )
    parser.add_argument(
        "--swm_hidden_dim", type=int, default=256,
        help="Hidden dimension of the SWM binary classification head.",
    )

    # ===== atlas loss control =====
    parser.add_argument(
        "--train_atlases", type=str, default="all",
        help="Atlases used for backprop. 'all' or comma-separated names.",
    )

    # ===== anatomical-prior specific =====
    parser.add_argument(
        "--mid_layer", type=str, default="yeo",
        choices=["yeo", "lobe"],
        help=(
            "Source of the mid-layer prior. 'yeo' uses 7 yeo-network labels "
            "from atlas_targets and M_yeo_to_{atlas}.npy under "
            "{overlap_dir}/yeo/. 'lobe' uses 14 hemisphere-lobe labels from "
            "lobe_targets and M_lobe_to_{atlas}.npy under {overlap_dir}/lobe/. "
            "mid_dim is set automatically: yeo->7, lobe->14."
        ),
    )
    parser.add_argument(
        "--overlap_dir", type=str,
        default="/home/heyifei/codes/test/atlas_overlap",
        help=(
            "Parent directory holding the per-source overlap subfolders "
            "{yeo,lobe}/. The actual matrix path is "
            "{overlap_dir}/{mid_layer}/M_{mid_layer}_to_{atlas}.npy."
        ),
    )
    parser.add_argument(
        "--lambda_mid", type=float, default=1.0,
        help="Weight on the mid CE that supervises mid_head_start/end.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help=(
            "Temperature applied to mid logits before softmax when forming "
            "the prior. >1 softens the prior, <1 sharpens it."
        ),
    )
    parser.add_argument(
        "--gate_init", type=float, default=-6.0,
        help=(
            "Initial raw gate value. sigmoid(-6)≈0.0025, so the model starts "
            "close to the no-prior baseline. Use 0 for stronger initial prior."
        ),
    )
    parser.add_argument(
        "--no_prior", action="store_true", default=False,
        help=(
            "Ablation mode: freeze gate at a large negative value so the prior "
            "residual is exactly 0, and skip mid CE (effectively lambda_mid=0). "
            "Architecture is unchanged so logging and result format stay "
            "identical, but the prior path contributes nothing and the mid "
            "head receives no supervision. Folder name is prefixed with "
            "'noPriorAblation' for easy separation."
        ),
    )

    # ===== save / result logging =====
    parser.add_argument(
        "--result_root", type=str,
        default="/data/hyf/swm_identification/train_result",
        help="Root directory for all training outputs.",
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Optional manual run folder name. If not set, a parameter-based name is generated.",
    )
    parser.add_argument(
        "--save_path", type=str, default=None,
        help=(
            "Optional explicit best model path. If omitted, best_model.pth is "
            "saved inside the generated result directory."
        ),
    )

    return parser.parse_args()


# =====================================================
# Utilities
# =====================================================
def parse_train_atlases(args):
    if args.train_atlases == "all":
        train_atlases = list(ATLAS_LIST)
    else:
        train_atlases = [a.strip() for a in args.train_atlases.split(",") if a.strip()]

    invalid = [a for a in train_atlases if a not in ATLAS_LIST]
    if invalid:
        raise ValueError(f"Invalid atlas names: {invalid}. Available: {ATLAS_LIST}")
    if not train_atlases:
        raise ValueError("train_atlases is empty.")
    return train_atlases


def compute_classification_metrics(pred_list, label_list, average="macro"):
    if len(pred_list) == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    preds = torch.cat(pred_list).numpy()
    labels = torch.cat(label_list).numpy()

    acc = (preds == labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=average, zero_division=0,
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# =====================================================
# Subject-level metrics
# =====================================================
# Pooled metrics (above) combine every fiber from every subject into one big
# pool and produce one number per head. That hides between-subject variance.
# The functions below compute the same four metrics per subject and then
# aggregate as mean / std across subjects.
#
# Single-class-only subjects in macro F1: when a subject only sees a small
# subset of classes, sklearn computes macro over the classes present (those
# given via `labels=` are kept; with zero_division=0 absent classes contribute
# 0). We restrict the label set per call to the union actually observed, which
# matches what a per-subject report should reflect.
_METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def _nan_metric_block():
    return {m: float("nan") for m in _METRIC_NAMES}


def compute_per_subject_metrics(preds, labels, subj_ids, average="macro"):
    """Group fiber-level (preds, labels) by subject_id, compute one set of
    metrics per subject, then return per-subject details + mean/std across
    subjects.

    Returns:
        {
          "per_subject": {subj_id: {accuracy, precision, recall, f1, n_samples}},
          "mean":       {accuracy, precision, recall, f1},  # across subjects
          "std":        {accuracy, precision, recall, f1},  # across subjects (ddof=0)
          "n_subjects": int,
          "n_samples":  int,
        }

    Notes:
        - average="binary" for SWM head, "macro" for atlas/mid heads.
        - std is population std (ddof=0); switch to ddof=1 if you want sample std.
        - subjects with 0 samples contribute nothing.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    subj_ids = np.asarray(subj_ids)

    n = len(preds)
    if not (len(labels) == len(subj_ids) == n):
        raise ValueError(
            f"length mismatch: preds={n} labels={len(labels)} subj={len(subj_ids)}"
        )
    if n == 0:
        return {
            "per_subject": {},
            "mean": _nan_metric_block(),
            "std": _nan_metric_block(),
            "n_subjects": 0,
            "n_samples": 0,
        }

    per_subj = {}
    for s in np.unique(subj_ids):
        m = subj_ids == s
        p = preds[m]
        l = labels[m]
        if len(p) == 0:
            continue
        acc = float((p == l).mean())
        prec, rec, f1, _ = precision_recall_fscore_support(
            l, p, average=average, zero_division=0,
        )
        per_subj[str(s)] = {
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "n_samples": int(m.sum()),
        }

    means, stds = {}, {}
    for m_name in _METRIC_NAMES:
        vals = np.array([v[m_name] for v in per_subj.values()], dtype=np.float64)
        means[m_name] = float(vals.mean()) if vals.size else float("nan")
        stds[m_name] = float(vals.std(ddof=0)) if vals.size else float("nan")

    return {
        "per_subject": per_subj,
        "mean": means,
        "std": stds,
        "n_subjects": len(per_subj),
        "n_samples": int(n),
    }


def write_per_subject_summary_csv(per_subject_by_head, csv_path):
    """One row per head: mean ± std across subjects for acc/precision/recall/f1."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["head", "n_subjects", "n_samples"]
    for m in _METRIC_NAMES:
        fieldnames.extend([f"{m}_mean", f"{m}_std"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for head, info in per_subject_by_head.items():
            row = {
                "head": head,
                "n_subjects": info["n_subjects"],
                "n_samples": info["n_samples"],
            }
            for m in _METRIC_NAMES:
                row[f"{m}_mean"] = f"{info['mean'][m]:.6f}"
                row[f"{m}_std"] = f"{info['std'][m]:.6f}"
            writer.writerow(row)
    print(f"Saved per-subject summary to: {csv_path}")


def write_per_subject_detail_csv(per_subject_by_head, csv_path):
    """One row per (subject, head): per-subject metric values."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["subject_id", "head", "n_samples"] + list(_METRIC_NAMES)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for head, info in per_subject_by_head.items():
            for subj, m in info["per_subject"].items():
                row = {"subject_id": subj, "head": head, "n_samples": m["n_samples"]}
                for k in _METRIC_NAMES:
                    row[k] = f"{m[k]:.6f}"
                writer.writerow(row)
    print(f"Saved per-subject details to: {csv_path}")


def print_per_subject_summary(per_subject_by_head, label="Test", indent="  "):
    """Pretty-print mean ± std across subjects for each head."""
    print(f"{label} per-subject metrics (mean ± std across N subjects):")
    for head, info in per_subject_by_head.items():
        n = info["n_subjects"]
        m, s = info["mean"], info["std"]
        print(
            f"{indent}{head:<20} N={n:>3}  "
            f"acc={m['accuracy']:.4f}±{s['accuracy']:.4f}  "
            f"f1={m['f1']:.4f}±{s['f1']:.4f}  "
            f"prec={m['precision']:.4f}±{s['precision']:.4f}  "
            f"rec={m['recall']:.4f}±{s['recall']:.4f}"
        )


# =====================================================
# Pair-level (start × end composite class) metrics
# =====================================================
# Per-position metrics treat start_endpoint and end_endpoint as two independent
# classification problems. A fiber is fully recovered only when BOTH endpoints
# are correctly classified. To capture that, we form a composite class
#     composite_class = start_class * n_classes + end_class
# and compute the usual ACC + macro precision/recall/f1 on that.
#
# Pair-ACC = (start correct) AND (end correct), per fiber.
# Pair-F1/precision/recall are macro across the composite classes that actually
# appear in the data (sklearn defaults). The composite space is up to
# n_classes^2 wide (BN: 60516), but only a small fraction of those pairs are
# anatomically plausible so the active set is much smaller.


def _pair_composite(start_arr, end_arr, n_classes):
    """Encode (start, end) into a single composite class id."""
    s = np.asarray(start_arr).astype(np.int64)
    e = np.asarray(end_arr).astype(np.int64)
    if s.shape != e.shape:
        raise ValueError(f"start/end shape mismatch: {s.shape} vs {e.shape}")
    return s * int(n_classes) + e


def compute_pooled_pair_metrics(start_raw, end_raw, n_classes, average="macro"):
    """Fiber-level pooled metrics over the composite (start, end) class."""
    n = len(start_raw["preds"])
    if not (len(start_raw["labels"]) == len(end_raw["preds"]) == len(end_raw["labels"]) == n):
        raise ValueError(
            f"length mismatch across start/end raw arrays: "
            f"sp={n} sl={len(start_raw['labels'])} "
            f"ep={len(end_raw['preds'])} el={len(end_raw['labels'])}"
        )
    if n == 0:
        return _nan_metric_block()
    comp_pred = _pair_composite(start_raw["preds"], end_raw["preds"], n_classes)
    comp_label = _pair_composite(start_raw["labels"], end_raw["labels"], n_classes)
    acc = float((comp_pred == comp_label).mean())
    prec, rec, f1, _ = precision_recall_fscore_support(
        comp_label, comp_pred, average=average, zero_division=0,
    )
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def _pair_raw_from_start_end(start_raw, end_raw, n_classes):
    """Build a composite-class raw dict reusable by compute_per_subject_metrics.

    Requires the same subject list on both sides (true for atlas/mid heads in
    eval_one_epoch: both share the SWM-positive mask and the same iteration
    order, so subj_pos is the same list).
    """
    if start_raw["subj"] != end_raw["subj"]:
        # Fall back to length check + alignment-by-position; eval_one_epoch
        # guarantees identical lists, so this is just defensive.
        if len(start_raw["subj"]) != len(end_raw["subj"]):
            raise ValueError("start/end subject lists differ in length")
    comp_pred = _pair_composite(start_raw["preds"], end_raw["preds"], n_classes)
    comp_label = _pair_composite(start_raw["labels"], end_raw["labels"], n_classes)
    return {"preds": comp_pred, "labels": comp_label, "subj": list(start_raw["subj"])}


def compute_all_pooled_pair_metrics(raw, mid_layer):
    """{f'{atlas}_pair': metrics, f'mid_{mid_layer}_pair': metrics}."""
    out = {}
    for atlas, n in ATLAS_ROI_DIMS.items():
        s_key, e_key = f"{atlas}_start", f"{atlas}_end"
        if s_key in raw and e_key in raw:
            out[f"{atlas}_pair"] = compute_pooled_pair_metrics(
                raw[s_key], raw[e_key], n
            )
    if "mid_start" in raw and "mid_end" in raw:
        out[f"mid_{mid_layer}_pair"] = compute_pooled_pair_metrics(
            raw["mid_start"], raw["mid_end"], MID_DIMS[mid_layer]
        )
    return out


def print_pair_metrics_summary(pair_metrics, label="Test", indent="  "):
    """Pooled pair-level metrics — one line per head."""
    print(f"{label} pair-level metrics (start × end as composite class):")
    for head, m in pair_metrics.items():
        print(
            f"{indent}{head:<22}  "
            f"acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
            f"prec={m['precision']:.4f}  rec={m['recall']:.4f}"
        )


def write_pair_metrics_csv(pair_metrics, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["head", "accuracy", "precision", "recall", "f1"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for head, m in pair_metrics.items():
            row = {"head": head}
            for k in fieldnames[1:]:
                v = m[k]
                row[k] = f"{v:.6f}" if isinstance(v, float) else v
            writer.writerow(row)
    print(f"Saved pair-level pooled metrics to: {csv_path}")


MID_DIMS = {"yeo": 7, "lobe": 14}

# Number of ROI classes per atlas (matches what the model heads output).
# Kept at module scope so pair-level metric helpers can look up the composite
# class space size (= n_classes * n_classes).
ATLAS_ROI_DIMS = {
    "yeo": 7, "DK": 70, "Brainnetome": 246,
    "AAL": 116, "schaefer_100": 100, "Destrieux": 75,
}


def resolve_overlap_dir(args):
    """Auto-route --overlap_dir to the {mid_layer}/ subfolder."""
    return os.path.join(args.overlap_dir, args.mid_layer)


def build_model(args, device):
    mid_dim = MID_DIMS[args.mid_layer]
    model = UnifiedSWMNet(
        atlas_roi_dims=ATLAS_ROI_DIMS,
        backbone=args.model_type,
        mid_dim=mid_dim,
        mid_source=args.mid_layer,
        overlap_dir=resolve_overlap_dir(args),
        temperature=args.temperature,
        gate_init=args.gate_init,
        global_feat_dim=args.global_feat_dim,
        endpoint_dim=args.endpoint_dim,
        swm_hidden_dim=args.swm_hidden_dim,
        use_endpoint=args.use_endpoint,
    ).to(device)
    return model


def _mid_label_source(mid_layer, pos, atlas_targets, lobe_targets):
    """Look up the per-position mid supervision label tensor.

    mid_layer == 'yeo'  -> atlas_targets['yeo_{pos}']   (7 classes)
    mid_layer == 'lobe' -> lobe_targets['lobe_{pos}']   (14 classes)
    """
    if mid_layer == "yeo":
        return atlas_targets[f"yeo_{pos}"]
    if mid_layer == "lobe":
        return lobe_targets[f"lobe_{pos}"]
    raise ValueError(f"Unknown mid_layer: {mid_layer}")


def format_duration(seconds):
    """Format a positive float number of seconds as 'Xh Ym Zs' (or 'Ym Zs' / 'Zs')."""
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def safe_tag(x):
    x = str(x)
    x = x.replace("+", "p").replace("-", "m").replace(".", "p")
    x = re.sub(r"[^A-Za-z0-9_=,.-]+", "_", x)
    return x.strip("_")


def make_run_dir(args):
    if args.run_name:
        name = safe_tag(args.run_name)
    else:
        atlas_tag = "all" if args.train_atlases == "all" else args.train_atlases.replace(",", "-")
        prefix = "noPriorAblation" if args.no_prior else "anatomicalPriorGated"
        endpoint_tag = "withEndpoint" if args.use_endpoint else "streamlineOnly_noEndpoint"
        name = "_".join([
            prefix,
            f"mid={args.mid_layer}",
            endpoint_tag,
            f"model={args.model_type}",
            f"gdim={args.global_feat_dim}",
            f"edim={args.endpoint_dim}",
            f"swmhid={args.swm_hidden_dim}",
            f"lr={args.lr}",
            f"wd={args.weight_decay}",
            f"bs={args.batch_size}",
            f"ep={args.epochs}",
            f"lambdaMid={args.lambda_mid}",
            f"temp={args.temperature}",
            f"gate={args.gate_init}",
            f"atlas={atlas_tag}",
            f"seed={args.seed}",
        ])
        name = safe_tag(name)

    # Add timestamp to avoid accidental overwrite while still keeping parameters in the folder name.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.result_root, f"{name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def make_save_path(args, run_dir, fold=None):
    if args.save_path is None:
        filename = "best_model"
        if fold is not None:
            filename += f"_fold{fold}"
        filename += ".pth"
        return os.path.join(run_dir, filename)

    if fold is None:
        return args.save_path
    base, ext = os.path.splitext(args.save_path)
    if ext == "":
        ext = ".pth"
    return f"{base}_fold{fold}{ext}"


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def save_parameter_summary(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary = count_parameters(model)

    csv_path = os.path.join(output_dir, "model_parameters.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "shape", "numel", "requires_grad"]
        )
        writer.writeheader()
        for name, param in model.named_parameters():
            writer.writerow({
                "name": name,
                "shape": list(param.shape),
                "numel": param.numel(),
                "requires_grad": param.requires_grad,
            })

    txt_path = os.path.join(output_dir, "model_summary.txt")
    with open(txt_path, "w") as f:
        f.write("Model parameter summary\n")
        f.write("=======================\n")
        f.write(f"total_params: {summary['total_params']}\n")
        f.write(f"trainable_params: {summary['trainable_params']}\n\n")
        f.write(str(model))
        f.write("\n")

    return summary


def mean_metric(metrics, metric_name="f1", include_mid=False):
    values = []
    for atlas in ATLAS_LIST:
        for pos in ["start", "end"]:
            key = f"{atlas}_{pos}"
            if key in metrics:
                v = metrics[key].get(metric_name, float("nan"))
                if v == v:  # not nan
                    values.append(v)
    if include_mid:
        for pos in ["start", "end"]:
            key = f"mid_{pos}"
            if key in metrics:
                v = metrics[key].get(metric_name, float("nan"))
                if v == v:
                    values.append(v)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def flatten_metrics(metrics, prefix):
    row = {}
    for key, m in metrics.items():
        for metric_name, value in m.items():
            try:
                row[f"{prefix}_{key}_{metric_name}"] = float(value)
            except Exception:
                row[f"{prefix}_{key}_{metric_name}"] = value
    row[f"{prefix}_atlas_mean_acc"] = mean_metric(metrics, "accuracy")
    row[f"{prefix}_atlas_mean_precision"] = mean_metric(metrics, "precision")
    row[f"{prefix}_atlas_mean_recall"] = mean_metric(metrics, "recall")
    row[f"{prefix}_atlas_mean_f1"] = mean_metric(metrics, "f1")
    return row


def write_history_csv(history, path):
    if not history:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in history:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_curves(history, output_dir, fold=None):
    if not history:
        return
    os.makedirs(output_dir, exist_ok=True)
    tag = "single" if fold is None else f"fold{fold}"
    epochs = [r["epoch"] for r in history]

    def plot_curve(y_keys, ylabel, filename, ylim=None):
        plt.figure(figsize=(8, 5))
        for key in y_keys:
            y = [r.get(key, float("nan")) for r in history]
            plt.plot(epochs, y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} curve ({tag})")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    plot_curve(["train_loss", "val_loss"], "Loss", f"convergence_loss_{tag}.png")
    plot_curve(
        ["train_swm_f1", "val_swm_f1", "train_atlas_mean_f1", "val_atlas_mean_f1"],
        "F1",
        f"convergence_f1_{tag}.png",
        ylim=(0.9, 1.0),
    )
    plot_curve(
        ["train_atlas_mean_acc", "val_atlas_mean_acc"],
        "Accuracy",
        f"convergence_accuracy_{tag}.png",
        ylim=(0.9, 1.0),
    )


def write_snapshot_csv(snapshot_rows, path):
    if not snapshot_rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for r in snapshot_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in snapshot_rows:
            writer.writerow(r)


def print_metrics_summary(metrics, label="Test", include_mid=True, mid_label="mid_yeo"):
    """SWM binary + per-atlas mean(start, end) metrics, with optional mid row.

    mid_label is the display name for the mid row (e.g. 'mid_yeo' or 'mid_lobe').
    """
    swm = metrics.get("swm")
    if swm is not None:
        print(
            f"  {label} SWM (binary): acc={swm['accuracy']:.4f}, "
            f"f1={swm['f1']:.4f}, precision={swm['precision']:.4f}, "
            f"recall={swm['recall']:.4f}"
        )

    rows = [(atlas, atlas) for atlas in ATLAS_LIST]
    if include_mid:
        rows.append((mid_label, "mid"))

    print(f"  {label} per-atlas mean(start, end):")
    for display, prefix in rows:
        s_key = f"{prefix}_start"
        e_key = f"{prefix}_end"
        if s_key not in metrics or e_key not in metrics:
            continue
        s = metrics[s_key]
        e = metrics[e_key]
        print(
            f"    {display}: "
            f"acc={0.5 * (s['accuracy'] + e['accuracy']):.4f}, "
            f"precision={0.5 * (s['precision'] + e['precision']):.4f}, "
            f"recall={0.5 * (s['recall'] + e['recall']):.4f}, "
            f"f1={0.5 * (s['f1'] + e['f1']):.4f} "
            f"(start_acc={s['accuracy']:.4f}, end_acc={e['accuracy']:.4f})"
        )


def format_gate(gate_dict):
    parts = []
    for atlas in ATLAS_LIST:
        for pos in ["start", "end"]:
            key = f"{atlas}_{pos}"
            g = gate_dict[key]["sigmoid"]
            parts.append(f"{key}={g:.4f}")
    return ", ".join(parts)


def metrics_to_csv_rows(metrics, fold=None, split="test", include_mid=True, mid_label="mid_yeo"):
    fold_str = "single" if fold is None else f"fold{fold}"
    pairs = [(atlas, atlas) for atlas in ATLAS_LIST]
    if include_mid:
        pairs.append((mid_label, "mid"))

    rows = []
    swm = metrics.get("swm")
    if swm is not None:
        rows.append({
            "split": split, "fold": fold_str, "atlas": "swm", "position": "binary",
            "accuracy": swm["accuracy"], "f1": swm["f1"],
            "precision": swm["precision"], "recall": swm["recall"],
        })

    for atlas_name, prefix in pairs:
        for pos in ["start", "end"]:
            key = f"{prefix}_{pos}"
            if key not in metrics:
                continue
            m = metrics[key]
            rows.append({
                "split": split, "fold": fold_str, "atlas": atlas_name, "position": pos,
                "accuracy": m["accuracy"], "f1": m["f1"],
                "precision": m["precision"], "recall": m["recall"],
            })
    return rows


def write_metrics_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["split", "fold", "atlas", "position", "accuracy", "f1", "precision", "recall"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            for k in ("accuracy", "f1", "precision", "recall"):
                v = row_out.get(k)
                row_out[k] = f"{v:.6f}" if isinstance(v, float) else v
            writer.writerow(row_out)
    print(f"Saved metrics to: {csv_path}")


def mean_std_ignore_nan(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    valid = tensor[~torch.isnan(tensor)]
    if valid.numel() == 0:
        return float("nan"), float("nan")
    return valid.mean().item(), valid.std(unbiased=False).item()


def summarize_kfold_results(fold_results):
    print("\n" + "=" * 80)
    print("K-fold cross validation summary")
    print("=" * 80)

    test_loss_mean, test_loss_std = mean_std_ignore_nan(
        [r["test_loss"] for r in fold_results]
    )
    print(f"Test Loss: {test_loss_mean:.4f} ± {test_loss_std:.4f}")

    metric_keys = list(fold_results[0]["test_metrics"].keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]

    for key in metric_keys:
        parts = []
        for metric_name in metric_names:
            m_value, s_value = mean_std_ignore_nan(
                [r["test_metrics"][key][metric_name] for r in fold_results]
            )
            parts.append(f"{metric_name}={m_value:.4f}±{s_value:.4f}")
        print(f"{key}: " + ", ".join(parts))


# =====================================================
# Train / Eval
# =====================================================
def train_one_epoch(model, loader, optimizer, criterion, device, train_atlases, lambda_mid, mid_layer):
    model.train()
    total_loss = 0.0
    total_samples = 0

    all_preds = {f"{a}_{p}": [] for a in ATLAS_LIST for p in ["start", "end"]}
    all_labels = {f"{a}_{p}": [] for a in ATLAS_LIST for p in ["start", "end"]}
    all_swm_preds, all_swm_labels = [], []
    all_mid_preds = {"start": [], "end": []}
    all_mid_labels = {"start": [], "end": []}

    for X, atlas_targets, lobe_targets, _ in loader:
        X = X.to(device, non_blocking=True)
        atlas_targets = {k: v.to(device, non_blocking=True) for k, v in atlas_targets.items()}
        lobe_targets = {k: v.to(device, non_blocking=True) for k, v in lobe_targets.items()}

        optimizer.zero_grad()
        outputs = model(X)
        swm_mask = atlas_targets["swm"] == 1

        batch_loss = criterion(outputs["swm"], atlas_targets["swm"])

        all_swm_preds.append(outputs["swm"].argmax(dim=1).cpu())
        all_swm_labels.append(atlas_targets["swm"].cpu())

        if swm_mask.sum().item() > 0:
            # ----- mid CE: supervise mid heads with yeo or lobe labels -----
            for pos in ["start", "end"]:
                mid_target = _mid_label_source(mid_layer, pos, atlas_targets, lobe_targets)
                batch_loss += lambda_mid * criterion(
                    outputs[f"mid_{pos}"][swm_mask], mid_target[swm_mask],
                )
                preds = outputs[f"mid_{pos}"][swm_mask].argmax(dim=1)
                all_mid_preds[pos].append(preds.cpu())
                all_mid_labels[pos].append(mid_target[swm_mask].cpu())

            # ----- atlas CE: only train_atlases for backprop, all for metric -----
            for atlas in train_atlases:
                for pos in ["start", "end"]:
                    key = f"{atlas}_{pos}"
                    batch_loss += criterion(
                        outputs[key][swm_mask], atlas_targets[key][swm_mask],
                    )

            for atlas in ATLAS_LIST:
                for pos in ["start", "end"]:
                    key = f"{atlas}_{pos}"
                    preds = outputs[key][swm_mask].argmax(dim=1)
                    all_preds[key].append(preds.cpu())
                    all_labels[key].append(atlas_targets[key][swm_mask].cpu())

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * X.size(0)
        total_samples += X.size(0)

    avg_loss = total_loss / total_samples

    metrics = {}
    for atlas in ATLAS_LIST:
        for pos in ["start", "end"]:
            key = f"{atlas}_{pos}"
            metrics[key] = compute_classification_metrics(
                all_preds[key], all_labels[key], average="macro",
            )
    for pos in ["start", "end"]:
        metrics[f"mid_{pos}"] = compute_classification_metrics(
            all_mid_preds[pos], all_mid_labels[pos], average="macro",
        )
    metrics["swm"] = compute_classification_metrics(
        all_swm_preds, all_swm_labels, average="binary",
    )

    return avg_loss, metrics


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, train_atlases, mid_layer,
                   collect_raw=False):
    """Evaluate over `loader`. Always returns (avg_loss, metrics, raw).

    `raw` is None unless collect_raw=True; when truthy, it carries concatenated
    fiber-level (preds, labels, subject_ids) per head, suitable for downstream
    subject-level aggregation. We only collect this at test time to avoid the
    extra memory traffic during every-epoch validation.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = {f"{a}_{p}": [] for a in ATLAS_LIST for p in ["start", "end"]}
    all_labels = {f"{a}_{p}": [] for a in ATLAS_LIST for p in ["start", "end"]}
    all_swm_preds, all_swm_labels = [], []
    all_mid_preds = {"start": [], "end": []}
    all_mid_labels = {"start": [], "end": []}

    # Subject-id tracks (only populated when collect_raw=True):
    #   subj_swm  -> per all-samples row (aligns with all_swm_*)
    #   subj_pos  -> per SWM-positive row (aligns with atlas/mid arrays)
    subj_swm, subj_pos = [], []

    for X, atlas_targets, lobe_targets, subject_ids in loader:
        X = X.to(device, non_blocking=True)
        atlas_targets = {k: v.to(device, non_blocking=True) for k, v in atlas_targets.items()}
        lobe_targets = {k: v.to(device, non_blocking=True) for k, v in lobe_targets.items()}

        outputs = model(X)
        swm_mask = atlas_targets["swm"] == 1

        # Val/test loss for checkpoint selection: SWM + atlas CEs only (NO mid CE),
        # matching the convention in no-prior baselines.
        batch_loss = criterion(outputs["swm"], atlas_targets["swm"])

        all_swm_preds.append(outputs["swm"].argmax(dim=1).cpu())
        all_swm_labels.append(atlas_targets["swm"].cpu())

        if collect_raw:
            batch_subj = _batch_subject_list(subject_ids)
            subj_swm.extend(batch_subj)

        if swm_mask.sum().item() > 0:
            if collect_raw:
                mask_np = swm_mask.cpu().numpy()
                subj_pos.extend([s for s, m in zip(batch_subj, mask_np) if m])

            for pos in ["start", "end"]:
                mid_target = _mid_label_source(mid_layer, pos, atlas_targets, lobe_targets)
                preds = outputs[f"mid_{pos}"][swm_mask].argmax(dim=1)
                all_mid_preds[pos].append(preds.cpu())
                all_mid_labels[pos].append(mid_target[swm_mask].cpu())

            for atlas in train_atlases:
                for pos in ["start", "end"]:
                    key = f"{atlas}_{pos}"
                    batch_loss += criterion(
                        outputs[key][swm_mask], atlas_targets[key][swm_mask],
                    )

            for atlas in ATLAS_LIST:
                for pos in ["start", "end"]:
                    key = f"{atlas}_{pos}"
                    preds = outputs[key][swm_mask].argmax(dim=1)
                    all_preds[key].append(preds.cpu())
                    all_labels[key].append(atlas_targets[key][swm_mask].cpu())

        total_loss += batch_loss.item() * X.size(0)
        total_samples += X.size(0)

    avg_loss = total_loss / total_samples

    metrics = {}
    for atlas in ATLAS_LIST:
        for pos in ["start", "end"]:
            key = f"{atlas}_{pos}"
            metrics[key] = compute_classification_metrics(
                all_preds[key], all_labels[key], average="macro",
            )
    for pos in ["start", "end"]:
        metrics[f"mid_{pos}"] = compute_classification_metrics(
            all_mid_preds[pos], all_mid_labels[pos], average="macro",
        )
    metrics["swm"] = compute_classification_metrics(
        all_swm_preds, all_swm_labels, average="binary",
    )

    raw = None
    if collect_raw:
        def _cat(t_list):
            if not t_list:
                return np.array([], dtype=np.int64)
            return torch.cat(t_list).numpy()

        raw = {"swm": {"preds": _cat(all_swm_preds),
                       "labels": _cat(all_swm_labels),
                       "subj": list(subj_swm)}}
        for atlas in ATLAS_LIST:
            for pos in ["start", "end"]:
                key = f"{atlas}_{pos}"
                raw[key] = {"preds": _cat(all_preds[key]),
                            "labels": _cat(all_labels[key]),
                            "subj": list(subj_pos)}
        for pos in ["start", "end"]:
            key = f"mid_{pos}"
            raw[key] = {"preds": _cat(all_mid_preds[pos]),
                        "labels": _cat(all_mid_labels[pos]),
                        "subj": list(subj_pos)}

    return avg_loss, metrics, raw


def _batch_subject_list(subject_ids):
    """Normalize DataLoader's collated subject_ids into a plain list of strings.

    Strings come back from default_collate as a list/tuple; if some upstream
    change ever ships them as tensors, we still cope.
    """
    if isinstance(subject_ids, torch.Tensor):
        return [str(x) for x in subject_ids.cpu().tolist()]
    return [str(s) for s in subject_ids]


def build_per_subject_from_raw(raw, mid_layer):
    """Apply compute_per_subject_metrics to every head in `raw`, plus pair
    entries that combine start/end into a composite class per atlas + mid.

    SWM head uses average='binary'; everything else (including pair entries)
    uses 'macro'. The mid head key is renamed to f'mid_{mid_layer}' (e.g.
    'mid_yeo' / 'mid_lobe') so the summary lines up with how the pooled metrics
    are labelled elsewhere. Pair entries are keyed as f'{atlas}_pair' /
    f'mid_{mid_layer}_pair'.
    """
    out = {}
    for head, d in raw.items():
        if head == "swm":
            avg = "binary"
            display = "swm"
        elif head.startswith("mid_"):
            avg = "macro"
            pos = head.split("_", 1)[1]
            display = f"mid_{mid_layer}_{pos}"
        else:
            avg = "macro"
            display = head
        out[display] = compute_per_subject_metrics(
            d["preds"], d["labels"], d["subj"], average=avg,
        )

    # Pair entries: composite (start, end) class per atlas + mid head.
    for atlas, n in ATLAS_ROI_DIMS.items():
        s_key, e_key = f"{atlas}_start", f"{atlas}_end"
        if s_key in raw and e_key in raw:
            pr = _pair_raw_from_start_end(raw[s_key], raw[e_key], n)
            out[f"{atlas}_pair"] = compute_per_subject_metrics(
                pr["preds"], pr["labels"], pr["subj"], average="macro",
            )
    if "mid_start" in raw and "mid_end" in raw:
        pr = _pair_raw_from_start_end(
            raw["mid_start"], raw["mid_end"], MID_DIMS[mid_layer],
        )
        out[f"mid_{mid_layer}_pair"] = compute_per_subject_metrics(
            pr["preds"], pr["labels"], pr["subj"], average="macro",
        )
    return out


# =====================================================
# One split / one fold
# =====================================================
def run_one_split(args, train_set, val_set, test_set, device, save_path, train_atlases, run_dir, fold=None):
    # persistent_workers keeps the 64 dataloader processes alive across epochs
    # so we don't pay the spawn cost every epoch. prefetch_factor=4 lets each
    # worker keep a few batches queued. Both options require num_workers > 0,
    # so we only enable them in that case.
    loader_extra = (
        dict(persistent_workers=True, prefetch_factor=4)
        if args.num_workers > 0 else {}
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        **loader_extra,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        **loader_extra,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        **loader_extra,
    )
    print("Dataloaders ready")

    model = build_model(args, device)
    print("Model initialized (gated-residual anatomical prior)")
    print(
        f"  use_endpoint={args.use_endpoint} "
        f"global_feat_dim={args.global_feat_dim} endpoint_dim={args.endpoint_dim} "
        f"fused_dim={model.fused_dim}"
    )

    # ----- Ablation: --no_prior pins the gate to a large negative value so
    # sigmoid(gate) ≈ 0 and the prior residual is exactly 0, and skips the
    # mid CE. Architecture and logging are unchanged. Must happen BEFORE the
    # optimizer is constructed so Adam sees the frozen state of the gate.
    effective_lambda_mid = args.lambda_mid
    if args.no_prior:
        print(
            "  NO PRIOR ABLATION MODE: freezing gate at -50 (sigmoid≈1.9e-22), "
            "skipping mid CE (lambda_mid -> 0). Architecture is unchanged."
        )
        with torch.no_grad():
            for k in list(model.gate.keys()):
                model.gate[k].fill_(-50.0)
                model.gate[k].requires_grad = False
        effective_lambda_mid = 0.0

    print(
        f"  temperature={args.temperature}  lambda_mid={effective_lambda_mid}"
        f"{' (overridden by --no_prior)' if args.no_prior else ''}  "
        f"gate_init={args.gate_init}"
    )

    param_summary = save_parameter_summary(model, run_dir)
    print(
        f"  total_params={param_summary['total_params']} "
        f"trainable_params={param_summary['trainable_params']}"
    )

    # Two parameter groups: gate has weight_decay=0 (a learnable scalar should
    # not be pulled toward 0 by L2 regularization), everything else uses
    # args.weight_decay. Frozen gate (in --no_prior mode) still lives in the
    # optimizer; Adam skips it since requires_grad=False.
    gate_params, other_params = [], []
    for n, p in model.named_parameters():
        if n.startswith("gate."):
            gate_params.append(p)
        else:
            other_params.append(p)
    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "weight_decay": args.weight_decay},
            {"params": gate_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Save the best model to: {save_path}")

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    early_stop_triggered = False
    history = []
    snapshot_rows = []

    for epoch in range(1, args.epochs + 1):
        if fold is None:
            print(f"\nEpoch [{epoch}/{args.epochs}]")
        else:
            print(f"\nFold {fold}/{args.k_fold} | Epoch [{epoch}/{args.epochs}]")

        train_loss, train_metrics = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            criterion=criterion, device=device,
            train_atlases=train_atlases, lambda_mid=effective_lambda_mid,
            mid_layer=args.mid_layer,
        )
        print(f"Train Loss: {train_loss:.4f}")
        print_metrics_summary(train_metrics, label="Train", mid_label=f"mid_{args.mid_layer}")

        val_loss, val_metrics, _ = eval_one_epoch(
            model=model, loader=val_loader, criterion=criterion,
            device=device, train_atlases=train_atlases,
            mid_layer=args.mid_layer,
        )
        print(f"Val Loss: {val_loss:.4f}")
        print_metrics_summary(val_metrics, label="Val", mid_label=f"mid_{args.mid_layer}")

        gate = model.gate_snapshot()
        prior_weight = model.prior_weight_snapshot()
        print(f"  gate(sigmoid) = effective prior weight: {format_gate(gate)}")

        row = {
            "fold": "single" if fold is None else fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_swm_acc": train_metrics["swm"]["accuracy"],
            "train_swm_precision": train_metrics["swm"]["precision"],
            "train_swm_recall": train_metrics["swm"]["recall"],
            "train_swm_f1": train_metrics["swm"]["f1"],
            "val_swm_acc": val_metrics["swm"]["accuracy"],
            "val_swm_precision": val_metrics["swm"]["precision"],
            "val_swm_recall": val_metrics["swm"]["recall"],
            "val_swm_f1": val_metrics["swm"]["f1"],
            "train_atlas_mean_acc": mean_metric(train_metrics, "accuracy"),
            "train_atlas_mean_precision": mean_metric(train_metrics, "precision"),
            "train_atlas_mean_recall": mean_metric(train_metrics, "recall"),
            "train_atlas_mean_f1": mean_metric(train_metrics, "f1"),
            "val_atlas_mean_acc": mean_metric(val_metrics, "accuracy"),
            "val_atlas_mean_precision": mean_metric(val_metrics, "precision"),
            "val_atlas_mean_recall": mean_metric(val_metrics, "recall"),
            "val_atlas_mean_f1": mean_metric(val_metrics, "f1"),
        }
        row.update(flatten_metrics(train_metrics, "train"))
        row.update(flatten_metrics(val_metrics, "val"))
        history.append(row)

        snap = {"fold": "single" if fold is None else fold, "epoch": epoch}
        for k, v in gate.items():
            snap[f"gate_raw_{k}"] = v["raw"]
            snap[f"gate_sigmoid_{k}"] = v["sigmoid"]
        for k, v in prior_weight.items():
            snap[f"effective_prior_weight_{k}"] = v
        snapshot_rows.append(snap)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
                "param_summary": param_summary,
            }, save_path)
            print(f"Saved new best model at epoch {epoch} (Val Loss: {val_loss:.4f})")
        else:
            epochs_since_improvement += 1
            if args.early_stop_patience > 0:
                print(
                    f"  No val_loss improvement for {epochs_since_improvement}/"
                    f"{args.early_stop_patience} epochs "
                    f"(best={best_val_loss:.4f} @ epoch {best_epoch})"
                )

        # CSVs are cheap (text append) so we still rewrite them every epoch —
        # an interrupted run keeps useful logs. Convergence PNGs are dpi=300
        # and not worth regenerating every epoch; we render them once at the
        # end of training instead.
        tag = "single" if fold is None else f"fold{fold}"
        write_history_csv(history, os.path.join(run_dir, f"epoch_metrics_{tag}.csv"))
        write_snapshot_csv(snapshot_rows, os.path.join(run_dir, f"prior_gate_{tag}.csv"))

        # Early stop check at the end of the epoch — after CSVs are flushed.
        if (
            args.early_stop_patience > 0
            and epochs_since_improvement >= args.early_stop_patience
        ):
            print(
                f"\nEarly stopping at epoch {epoch}: "
                f"val_loss did not improve for {args.early_stop_patience} "
                f"consecutive epochs. Best={best_val_loss:.4f} @ epoch {best_epoch}."
            )
            early_stop_triggered = True
            break

    # Render the final convergence curves once after all epochs are done
    # (or early-stopped).
    save_curves(history, run_dir, fold=fold)

    # `epoch` is whatever the loop variable was last set to — either args.epochs
    # if training completed, or the early-stop epoch. Don't hardcode args.epochs.
    final_path = os.path.join(run_dir, "final_model.pth" if fold is None else f"final_model_fold{fold}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "param_summary": param_summary,
        "early_stop_triggered": early_stop_triggered,
    }, final_path)
    print(f"Saved final model to: {final_path}")

    print("\nTesting best model")
    checkpoint = torch.load(save_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Backward compatibility with old pure state_dict checkpoints.
        model.load_state_dict(checkpoint)

    test_loss, test_metrics, test_raw = eval_one_epoch(
        model=model, loader=test_loader, criterion=criterion,
        device=device, train_atlases=train_atlases,
        mid_layer=args.mid_layer,
        collect_raw=True,
    )
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics_summary(test_metrics, label="Test", mid_label=f"mid_{args.mid_layer}")
    print(f"  final gate(sigmoid): {format_gate(model.gate_snapshot())}")

    fold_tag = "" if fold is None else f"_fold{fold}"

    # ---- pair-level (start × end composite class) pooled metrics ----------
    pair_metrics = compute_all_pooled_pair_metrics(test_raw, mid_layer=args.mid_layer)
    # Merge into test_metrics so k-fold summarization picks them up
    # alongside per-position metrics.
    test_metrics.update(pair_metrics)
    print_pair_metrics_summary(pair_metrics, label="Test")
    write_pair_metrics_csv(
        pair_metrics,
        os.path.join(run_dir, f"test_metrics_pair{fold_tag}.csv"),
    )

    # ---- subject-level metrics (per fiber -> per subject -> mean ± std) ----
    # build_per_subject_from_raw also emits *_pair entries computed from the
    # composite class, so per-subject CSVs naturally include them.
    per_subject = build_per_subject_from_raw(test_raw, mid_layer=args.mid_layer)
    print_per_subject_summary(per_subject, label="Test")
    write_per_subject_summary_csv(
        per_subject,
        os.path.join(run_dir, f"test_metrics_per_subject_summary{fold_tag}.csv"),
    )
    write_per_subject_detail_csv(
        per_subject,
        os.path.join(run_dir, f"test_metrics_per_subject_detail{fold_tag}.csv"),
    )

    # Strip the bulky per-subject dict before stashing in result_summary.json
    # so the JSON stays small; the full per-subject table lives in the CSVs.
    per_subject_summary = {
        head: {
            "n_subjects": info["n_subjects"],
            "n_samples": info["n_samples"],
            "mean": info["mean"],
            "std": info["std"],
        }
        for head, info in per_subject.items()
    }

    result_json = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "save_path": save_path,
        "final_path": final_path,
        "last_epoch": epoch,
        "early_stop_triggered": early_stop_triggered,
        "early_stop_patience": args.early_stop_patience,
        "gate": model.gate_snapshot(),
        "effective_prior_weight": model.prior_weight_snapshot(),
        "param_summary": param_summary,
        "test_pair_metrics": pair_metrics,
        "test_per_subject_summary": per_subject_summary,
    }
    save_json(result_json, os.path.join(run_dir, "result_summary.json" if fold is None else f"result_summary_fold{fold}.json"))

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_pair_metrics": pair_metrics,
        "test_per_subject": per_subject,
        "save_path": save_path,
        "gate": model.gate_snapshot(),
        "effective_prior_weight": model.prior_weight_snapshot(),
    }


# =====================================================
# Main
# =====================================================
def main():
    args = parse_args()

    run_dir = make_run_dir(args)
    print(f"Result directory: {run_dir}")
    save_json(vars(args), os.path.join(run_dir, "config.json"))

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)
    print(f"Using model type: {args.model_type}")
    print(
        f"Mid layer: {args.mid_layer} "
        f"(mid_dim={MID_DIMS[args.mid_layer]}, "
        f"overlap matrices loaded from {resolve_overlap_dir(args)})"
    )

    train_atlases = parse_train_atlases(args)
    print("Atlases used for training loss:", train_atlases)
    print("Metrics will be calculated for all atlases:", ATLAS_LIST)

    # Total wall-clock timing — wraps the entire training body (single split or
    # k-fold). try/finally guarantees we still log elapsed time and write
    # runtime.json on exception/KeyboardInterrupt.
    t_start = time.time()
    interrupted = False
    error = None
    try:
        _run_training(args, run_dir, device, train_atlases)
    except KeyboardInterrupt:
        interrupted = True
        raise
    except Exception as e:
        error = repr(e)
        raise
    finally:
        t_end = time.time()
        elapsed = t_end - t_start
        elapsed_str = format_duration(elapsed)
        print(f"\nTotal training time: {elapsed_str} ({elapsed:.2f}s)")
        runtime_path = os.path.join(run_dir, "runtime.json")
        save_json({
            "start_iso": datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
            "end_iso": datetime.fromtimestamp(t_end).isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed,
            "elapsed_human": elapsed_str,
            "interrupted": interrupted,
            "error": error,
        }, runtime_path)
        print(f"Runtime info saved to: {runtime_path}")


def _run_training(args, run_dir, device, train_atlases):
    """Inner training body, factored out so main() can time it cleanly."""
    if args.k_fold <= 1:
        print("k_fold <= 1, using original train/val/test split")
        train_set, val_set, test_set = build_demo_datasets(
            h5_path=args.h5_path, csv_path=args.csv_path,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            seed=args.seed,
            stratify_by_swm=args.stratify_by_swm,
            group_by_subject=args.group_by_subject,
        )
        print("Dataset built")

        save_path = make_save_path(args, run_dir=run_dir, fold=None)
        result = run_one_split(
            args=args, train_set=train_set, val_set=val_set, test_set=test_set,
            device=device, save_path=save_path,
            train_atlases=train_atlases, run_dir=run_dir, fold=None,
        )

        rows = metrics_to_csv_rows(result["test_metrics"], fold=None, split="test", mid_label=f"mid_{args.mid_layer}")
        write_metrics_csv(rows, os.path.join(run_dir, "test_metrics.csv"))
        print(f"All outputs saved under: {run_dir}")
        return

    print(f"Using {args.k_fold}-fold cross validation")
    folds = build_demo_kfold_datasets(
        h5_path=args.h5_path, csv_path=args.csv_path,
        k_fold=args.k_fold, val_ratio=args.val_ratio,
        seed=args.seed, shuffle=True,
        stratify_by_swm=args.stratify_by_swm,
        group_by_subject=args.group_by_subject,
    )

    fold_results = []
    for fold_info in folds:
        fold = fold_info["fold"]
        print("\n" + "=" * 80)
        print(f"Running fold {fold}/{args.k_fold}")
        print("=" * 80)

        save_path = make_save_path(args, run_dir=run_dir, fold=fold)
        result = run_one_split(
            args=args,
            train_set=fold_info["train_set"],
            val_set=fold_info["val_set"],
            test_set=fold_info["test_set"],
            device=device, save_path=save_path,
            train_atlases=train_atlases,
            run_dir=run_dir,
            fold=fold,
        )
        fold_results.append(result)

    summarize_kfold_results(fold_results)

    all_rows = []
    for r in fold_results:
        all_rows.extend(metrics_to_csv_rows(r["test_metrics"], fold=r["fold"], split="test", mid_label=f"mid_{args.mid_layer}"))
    write_metrics_csv(all_rows, os.path.join(run_dir, "test_metrics_all_folds.csv"))

    # --- Pooled pair metrics across folds (one row per fold per head) -------
    pair_csv_path = os.path.join(run_dir, "test_metrics_pair_all_folds.csv")
    os.makedirs(os.path.dirname(pair_csv_path), exist_ok=True)
    with open(pair_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "head", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        for r in fold_results:
            for head, m in r["test_pair_metrics"].items():
                row = {"fold": r["fold"], "head": head}
                for k in ("accuracy", "precision", "recall", "f1"):
                    v = m[k]
                    row[k] = f"{v:.6f}" if isinstance(v, float) else v
                writer.writerow(row)
    print(f"Saved per-fold pair metrics to: {pair_csv_path}")

    # Aggregate pooled pair metrics across folds (mean ± std across folds).
    pair_heads = list(fold_results[0]["test_pair_metrics"].keys())
    pair_across_folds = {}
    for head in pair_heads:
        agg = {}
        for m_name in _METRIC_NAMES:
            vals = np.array(
                [r["test_pair_metrics"][head][m_name] for r in fold_results],
                dtype=np.float64,
            )
            valid = vals[~np.isnan(vals)]
            agg[f"{m_name}_mean"] = float(valid.mean()) if valid.size else float("nan")
            agg[f"{m_name}_std"] = float(valid.std(ddof=0)) if valid.size else float("nan")
        pair_across_folds[head] = agg

    # --- Per-subject metrics aggregated across all folds ---------------------
    # With --group_by_subject (default True), each subject is in exactly one
    # fold's test set, so the union of per-fold per-subject tables covers every
    # subject exactly once. Merge them into a single per-subject view and
    # recompute mean/std across the full subject population.
    combined_per_subject = {}
    for r in fold_results:
        for head, info in r["test_per_subject"].items():
            block = combined_per_subject.setdefault(
                head, {"per_subject": {}, "n_samples": 0},
            )
            # Detect duplicate subject ids across folds (would imply
            # group_by_subject was off — surface it loudly rather than silently
            # averaging the same subject twice).
            dup = set(block["per_subject"]).intersection(info["per_subject"])
            if dup:
                raise RuntimeError(
                    f"Subject(s) {sorted(dup)} appear in more than one fold "
                    f"for head '{head}'. Did --group_by_subject get disabled?"
                )
            block["per_subject"].update(info["per_subject"])
            block["n_samples"] += info["n_samples"]

    for head, block in combined_per_subject.items():
        means, stds = {}, {}
        for m_name in _METRIC_NAMES:
            vals = np.array(
                [v[m_name] for v in block["per_subject"].values()], dtype=np.float64,
            )
            means[m_name] = float(vals.mean()) if vals.size else float("nan")
            stds[m_name] = float(vals.std(ddof=0)) if vals.size else float("nan")
        block["mean"] = means
        block["std"] = stds
        block["n_subjects"] = len(block["per_subject"])

    print_per_subject_summary(combined_per_subject, label="Test (all folds)")
    write_per_subject_summary_csv(
        combined_per_subject,
        os.path.join(run_dir, "test_metrics_per_subject_summary_all_folds.csv"),
    )
    write_per_subject_detail_csv(
        combined_per_subject,
        os.path.join(run_dir, "test_metrics_per_subject_detail_all_folds.csv"),
    )

    save_json({
        "fold_results": [
            {
                "fold": r["fold"],
                "best_epoch": r["best_epoch"],
                "best_val_loss": r["best_val_loss"],
                "test_loss": r["test_loss"],
                "save_path": r["save_path"],
                "gate": r["gate"],
                "effective_prior_weight": r["effective_prior_weight"],
                "test_pair_metrics": r["test_pair_metrics"],
            }
            for r in fold_results
        ],
        "test_pair_metrics_across_folds": pair_across_folds,
        "test_per_subject_summary_all_folds": {
            head: {
                "n_subjects": block["n_subjects"],
                "n_samples": block["n_samples"],
                "mean": block["mean"],
                "std": block["std"],
            }
            for head, block in combined_per_subject.items()
        },
    }, os.path.join(run_dir, "kfold_result_summary.json"))
    print(f"All outputs saved under: {run_dir}")


if __name__ == "__main__":
    main()
