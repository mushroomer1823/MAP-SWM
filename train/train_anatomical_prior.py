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
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

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
            "Train UnifiedSWMNet with gated-residual anatomical-prior conditioning "
            "(yeo mid + fixed overlap matrices M)."
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
        help="Endpoint feature dimension after endpoint_mlp.",
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
        "--overlap_dir", type=str,
        default="/home/hyf/swm_identification/multi-atlas-swm/atlas_overlap",
        help="Directory holding M_yeo_to_{atlas}.npy.",
    )
    parser.add_argument(
        "--lambda_mid", type=float, default=1.0,
        help="Weight on the yeo CE that supervises mid_head_start/end.",
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


def build_model(args, device):
    atlas_roi_dims = {
        "yeo": 7, "DK": 70, "Brainnetome": 246,
        "AAL": 116, "schaefer_100": 100, "Destrieux": 75,
    }
    model = UnifiedSWMNet(
        atlas_roi_dims=atlas_roi_dims,
        backbone=args.model_type,
        mid_dim=7,
        overlap_dir=args.overlap_dir,
        temperature=args.temperature,
        gate_init=args.gate_init,
        global_feat_dim=args.global_feat_dim,
        endpoint_dim=args.endpoint_dim,
        swm_hidden_dim=args.swm_hidden_dim,
    ).to(device)
    return model


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
        name = "_".join([
            prefix,
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


def print_metrics_summary(metrics, label="Test", include_mid=True):
    """SWM binary + per-atlas mean(start, end) metrics, with optional mid_yeo row."""
    swm = metrics.get("swm")
    if swm is not None:
        print(
            f"  {label} SWM (binary): acc={swm['accuracy']:.4f}, "
            f"f1={swm['f1']:.4f}, precision={swm['precision']:.4f}, "
            f"recall={swm['recall']:.4f}"
        )

    rows = [(atlas, atlas) for atlas in ATLAS_LIST]
    if include_mid:
        rows.append(("mid_yeo", "mid"))

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


def metrics_to_csv_rows(metrics, fold=None, split="test", include_mid=True):
    fold_str = "single" if fold is None else f"fold{fold}"
    pairs = [(atlas, atlas) for atlas in ATLAS_LIST]
    if include_mid:
        pairs.append(("mid_yeo", "mid"))

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
def train_one_epoch(model, loader, optimizer, criterion, device, train_atlases, lambda_mid):
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

        optimizer.zero_grad()
        outputs = model(X)
        swm_mask = atlas_targets["swm"] == 1

        batch_loss = criterion(outputs["swm"], atlas_targets["swm"])

        all_swm_preds.append(outputs["swm"].argmax(dim=1).cpu())
        all_swm_labels.append(atlas_targets["swm"].cpu())

        if swm_mask.sum().item() > 0:
            # ----- mid CE: supervise mid heads with yeo labels -----
            for pos in ["start", "end"]:
                mid_label = atlas_targets[f"yeo_{pos}"]
                batch_loss += lambda_mid * criterion(
                    outputs[f"mid_{pos}"][swm_mask], mid_label[swm_mask],
                )
                preds = outputs[f"mid_{pos}"][swm_mask].argmax(dim=1)
                all_mid_preds[pos].append(preds.cpu())
                all_mid_labels[pos].append(mid_label[swm_mask].cpu())

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
def eval_one_epoch(model, loader, criterion, device, train_atlases):
    model.eval()
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

        outputs = model(X)
        swm_mask = atlas_targets["swm"] == 1

        # Val/test loss for checkpoint selection: SWM + atlas CEs only (NO mid CE),
        # matching the convention in no-prior baselines.
        batch_loss = criterion(outputs["swm"], atlas_targets["swm"])

        all_swm_preds.append(outputs["swm"].argmax(dim=1).cpu())
        all_swm_labels.append(atlas_targets["swm"].cpu())

        if swm_mask.sum().item() > 0:
            for pos in ["start", "end"]:
                mid_label = atlas_targets[f"yeo_{pos}"]
                preds = outputs[f"mid_{pos}"][swm_mask].argmax(dim=1)
                all_mid_preds[pos].append(preds.cpu())
                all_mid_labels[pos].append(mid_label[swm_mask].cpu())

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

    return avg_loss, metrics


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
        f"  global_feat_dim={args.global_feat_dim} endpoint_dim={args.endpoint_dim} "
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
        )
        print(f"Train Loss: {train_loss:.4f}")
        print_metrics_summary(train_metrics, label="Train")

        val_loss, val_metrics = eval_one_epoch(
            model=model, loader=val_loader, criterion=criterion,
            device=device, train_atlases=train_atlases,
        )
        print(f"Val Loss: {val_loss:.4f}")
        print_metrics_summary(val_metrics, label="Val")

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

    test_loss, test_metrics = eval_one_epoch(
        model=model, loader=test_loader, criterion=criterion,
        device=device, train_atlases=train_atlases,
    )
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics_summary(test_metrics, label="Test")
    print(f"  final gate(sigmoid): {format_gate(model.gate_snapshot())}")

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
    }
    save_json(result_json, os.path.join(run_dir, "result_summary.json" if fold is None else f"result_summary_fold{fold}.json"))

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
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
    print(f"Overlap dir: {args.overlap_dir}")

    train_atlases = parse_train_atlases(args)
    print("Atlases used for training loss:", train_atlases)
    print("Metrics will be calculated for all atlases:", ATLAS_LIST)

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

        rows = metrics_to_csv_rows(result["test_metrics"], fold=None, split="test")
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
        all_rows.extend(metrics_to_csv_rows(r["test_metrics"], fold=r["fold"], split="test"))
    write_metrics_csv(all_rows, os.path.join(run_dir, "test_metrics_all_folds.csv"))
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
            }
            for r in fold_results
        ]
    }, os.path.join(run_dir, "kfold_result_summary.json"))
    print(f"All outputs saved under: {run_dir}")


if __name__ == "__main__":
    main()
