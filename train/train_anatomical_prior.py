# train_anatomical_prior.py
#
# Trainer for model_anatomical_prior.UnifiedSWMNet.
#
# Differences vs. train_demo_no_lobe.py:
#   1. Loads UnifiedSWMNet from models.model_anatomical_prior, passing
#      overlap_dir / temperature / alpha_init.
#   2. Adds an extra yeo-CE auxiliary loss to supervise the mid heads
#      (mid_head_start / mid_head_end) with weight --lambda_mid.
#   3. Reports the mid heads' yeo accuracy as 'mid_yeo' under per-atlas
#      metrics, mirroring train_atlas_ablation_kfold.py's format.
#   4. Logs the 12 learnable alpha values after every epoch.
#   5. Default save directory is .../anatomical_prior/ so checkpoints do
#      not collide with no_lobe / lobe experiments.
#
# val_loss (used for early stopping) is SWM_CE + sum_atlas atlas_CE
# (NO mid CE), matching train_demo_no_lobe.py / train_atlas_ablation_kfold.py
# so best checkpoints are directly comparable across all three setups.

import argparse
import csv
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

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
            "Train UnifiedSWMNet with anatomical-prior conditioning "
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

    # ===== device =====
    parser.add_argument("--gpu", type=int, default=2)

    parser.add_argument(
        "--model_type", type=str, default="dgcnn",
        choices=["pointnet", "pointnet++", "pointmlp", "dgcnn"],
    )

    # ===== atlas loss control =====
    parser.add_argument(
        "--train_atlases", type=str, default="all",
        help="Atlases used for backprop. 'all' or comma-separated names.",
    )

    # ===== anatomical-prior specific =====
    parser.add_argument(
        "--overlap_dir", type=str,
        default="/home/heyifei/codes/test/atlas_overlap",
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
            "the prior. >1 softens the prior (more forgiving when mid is "
            "wrong), <1 sharpens it. Default 1.0 = no temperature scaling."
        ),
    )
    parser.add_argument(
        "--alpha_init", type=float, default=1.0,
        help="Initial value for every learnable alpha[atlas_pos].",
    )

    # ===== save =====
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="Path to save best model. For K-fold _fold{i} is appended.",
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
        alpha_init=args.alpha_init,
    ).to(device)
    return model


def make_save_path(args, fold=None):
    if args.save_path is None:
        save_dir = "/data/hyf/swm_identification/models/anatomical_prior"
        filename = f"best_unified_{args.model_type}_yeoPrior"
        if fold is not None:
            filename += f"_fold{fold}"
        filename += ".pth"
        return os.path.join(save_dir, filename)

    if fold is None:
        return args.save_path
    base, ext = os.path.splitext(args.save_path)
    if ext == "":
        ext = ".pth"
    return f"{base}_fold{fold}{ext}"


def print_metrics_summary(metrics, label="Test", include_mid=True):
    """SWM binary + per-atlas mean(start, end) acc, with optional mid_yeo row."""
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

    print(f"  {label} per-atlas mean(start, end) accuracy:")
    for display, prefix in rows:
        s_key = f"{prefix}_start"
        e_key = f"{prefix}_end"
        if s_key not in metrics or e_key not in metrics:
            continue
        s_acc = metrics[s_key]["accuracy"]
        e_acc = metrics[e_key]["accuracy"]
        print(
            f"    {display}: mean_acc={0.5 * (s_acc + e_acc):.4f} "
            f"(start={s_acc:.4f}, end={e_acc:.4f})"
        )


def format_alpha(alpha_dict):
    """One-line dump of all 12 alpha values, ordered by ATLAS_LIST."""
    parts = []
    for atlas in ATLAS_LIST:
        for pos in ["start", "end"]:
            key = f"{atlas}_{pos}"
            parts.append(f"{key}={alpha_dict[key]:+.3f}")
    return ", ".join(parts)


def metrics_to_csv_rows(metrics, fold=None, include_mid=True):
    fold_str = "single" if fold is None else f"fold{fold}"
    pairs = [(atlas, atlas) for atlas in ATLAS_LIST]
    if include_mid:
        pairs.append(("mid_yeo", "mid"))

    rows = []
    swm = metrics.get("swm")
    if swm is not None:
        rows.append({
            "fold": fold_str, "atlas": "swm", "position": "binary",
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
                "fold": fold_str, "atlas": atlas_name, "position": pos,
                "accuracy": m["accuracy"], "f1": m["f1"],
                "precision": m["precision"], "recall": m["recall"],
            })
    return rows


def write_metrics_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["fold", "atlas", "position", "accuracy", "f1", "precision", "recall"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            for k in ("accuracy", "f1", "precision", "recall"):
                v = row_out.get(k)
                row_out[k] = f"{v:.6f}" if isinstance(v, float) else v
            writer.writerow(row_out)
    print(f"Saved test metrics to: {csv_path}")


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

        # Val loss for checkpoint selection: SWM + atlas CEs only (NO mid CE),
        # matching the convention in train_demo_no_lobe / train_atlas_ablation_kfold.
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
def run_one_split(args, train_set, val_set, test_set, device, save_path, train_atlases, fold=None):
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print("Dataloaders ready")

    model = build_model(args, device)
    print("Model initialized (anatomical-prior)")
    print(f"  temperature={args.temperature}  lambda_mid={args.lambda_mid}  alpha_init={args.alpha_init}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Save the trained model to: {save_path}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        if fold is None:
            print(f"\nEpoch [{epoch}/{args.epochs}]")
        else:
            print(f"\nFold {fold}/{args.k_fold} | Epoch [{epoch}/{args.epochs}]")

        train_loss, train_metrics = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            criterion=criterion, device=device,
            train_atlases=train_atlases, lambda_mid=args.lambda_mid,
        )
        print(f"Train Loss: {train_loss:.4f}")
        print_metrics_summary(train_metrics, label="Train")

        val_loss, val_metrics = eval_one_epoch(
            model=model, loader=val_loader, criterion=criterion,
            device=device, train_atlases=train_atlases,
        )
        print(f"Val Loss: {val_loss:.4f}")
        print_metrics_summary(val_metrics, label="Val")

        # Always print alpha values — main diagnostic for whether the
        # model has decided to trust the prior on each (atlas, pos) head.
        print(f"  alpha: {format_alpha(model.alpha_snapshot())}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model at epoch {epoch} (Val Loss: {val_loss:.4f})")

    print("\nTesting best model")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_metrics = eval_one_epoch(
        model=model, loader=test_loader, criterion=criterion,
        device=device, train_atlases=train_atlases,
    )
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics_summary(test_metrics, label="Test")
    print(f"  final alpha: {format_alpha(model.alpha_snapshot())}")

    return {
        "fold": fold,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "save_path": save_path,
        "alpha": model.alpha_snapshot(),
    }


# =====================================================
# Main
# =====================================================
def main():
    args = parse_args()

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

        save_path = make_save_path(args, fold=None)
        result = run_one_split(
            args=args, train_set=train_set, val_set=val_set, test_set=test_set,
            device=device, save_path=save_path,
            train_atlases=train_atlases, fold=None,
        )

        csv_path = os.path.join(
            os.path.dirname(save_path),
            f"test_metrics_{args.model_type}_yeoPrior.csv",
        )
        rows = metrics_to_csv_rows(result["test_metrics"], fold=None)
        write_metrics_csv(rows, csv_path)
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

        save_path = make_save_path(args, fold=fold)
        result = run_one_split(
            args=args,
            train_set=fold_info["train_set"],
            val_set=fold_info["val_set"],
            test_set=fold_info["test_set"],
            device=device, save_path=save_path,
            train_atlases=train_atlases, fold=fold,
        )
        fold_results.append(result)

    summarize_kfold_results(fold_results)

    csv_path = os.path.join(
        os.path.dirname(make_save_path(args, fold=fold_results[0]["fold"])),
        f"test_metrics_{args.model_type}_yeoPrior.csv",
    )
    all_rows = []
    for r in fold_results:
        all_rows.extend(metrics_to_csv_rows(r["test_metrics"], fold=r["fold"]))
    write_metrics_csv(all_rows, csv_path)


if __name__ == "__main__":
    main()
