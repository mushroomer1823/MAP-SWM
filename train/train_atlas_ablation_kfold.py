# train_multi_atlas_pointnet_kfold.py

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
from models.model_lobe import UnifiedSWMNet  # noqa: E402


# =====================================================
# Args
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Unified SWM model with multi-atlas supervision and optional K-fold CV"
    )

    # ===== data =====
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/hyf/swm_identification/data/subject_tractogram",
        help="Root directory of subject tractogram data. Kept for compatibility."
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="/data/hyf/swm_identification/data/demo/All_swm_dwm_streamlines.h5",
        help="Path to h5 file containing streamlines."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/data/hyf/swm_identification/data/demo/All_swm_dwm_atlas_start_end_selected_with_lobe.csv",
        help="Path to csv file containing atlas and lobe labels."
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # ===== K-fold =====
    parser.add_argument(
        "--k_fold",
        type=int,
        default=1,
        help="Number of folds for K-fold cross validation. Use <=1 for original train/val/test split."
    )
    parser.add_argument(
        "--stratify_by_swm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to stratify folds by swm label if supported by dataset_demo_kfold. Use --no-stratify_by_swm to disable."
    )
    parser.add_argument(
        "--group_by_subject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to split folds by subject if supported by dataset_demo_kfold. Use --no-group_by_subject to disable."
    )

    # ===== training =====
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=64)

    # ===== device =====
    parser.add_argument(
        "--gpu",
        type=int,
        default=2,
        help="GPU id to use, set to -1 for CPU"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="dgcnn",
        choices=["pointnet", "pointnet++", "pointmlp", "dgcnn"],
        help="Backbone model type"
    )

    # ===== intermediate (mid) layer =====
    parser.add_argument(
        "--mid_layer",
        type=str,
        default="lobe",
        choices=["lobe", "yeo"],
        help=(
            "Which coarse-level label supervises the model's intermediate "
            "(mid) layer that conditions all downstream atlas heads. "
            "'lobe' uses lobe_start/end (14 classes) from lobe_targets; "
            "'yeo' uses yeo_start/end (7 classes) from atlas_targets. "
            "When 'yeo', yeo also stays as a leaf atlas head (double "
            "appearance) for fair cross-experiment comparison."
        ),
    )

    # ===== atlas loss control =====
    parser.add_argument(
        "--train_atlases",
        type=str,
        default="all",
        help=(
            "Atlases used for loss/backprop. "
            "Use 'all' or comma-separated names, e.g. 'yeo,DK,AAL'. "
            "Metrics are still calculated for all atlases."
        )
    )

    # ===== save =====
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save best model. For K-fold, _fold{i} is automatically appended."
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

    invalid_atlases = [a for a in train_atlases if a not in ATLAS_LIST]
    if len(invalid_atlases) > 0:
        raise ValueError(
            f"Invalid atlas names: {invalid_atlases}. Available atlases: {ATLAS_LIST}"
        )

    if len(train_atlases) == 0:
        raise ValueError("train_atlases is empty. Use 'all' or comma-separated atlas names.")

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
        labels,
        preds,
        average=average,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_model(args, device):
    atlas_roi_dims = {
        "yeo": 7,
        "DK": 70,
        "Brainnetome": 246,
        "AAL": 116,
        "schaefer_100": 100,
        "Destrieux": 75,
    }
    mid_dim = 14 if args.mid_layer == "lobe" else 7  # lobe=14, yeo=7

    model = UnifiedSWMNet(
        atlas_roi_dims=atlas_roi_dims,
        mid_dim=mid_dim,
        backbone=args.model_type,
    ).to(device)

    return model


def make_save_path(args, fold=None):
    """Generate save path. For K-fold, append fold id to avoid overwriting.
    Always includes args.mid_layer in the filename so runs with different
    mid layers (lobe vs yeo) do not overwrite each other.
    """
    if args.save_path is None:
        save_dir = "/data/hyf/swm_identification/models/all"
        filename = f"best_unified_{args.model_type}_{args.mid_layer}"
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


def print_metrics_summary(metrics, mid_label=None, label="Test"):
    """Print SWM binary metrics and per-atlas mean(start, end) accuracy.

    mid_label:
        None -> no mid layer (used by the no-lobe baseline)
        str  -> display this name for the mid_start/mid_end pair, e.g.
                'mid_lobe' or 'mid_yeo'.
    """
    swm = metrics.get("swm")
    if swm is not None:
        print(
            f"  {label} SWM (binary): acc={swm['accuracy']:.4f}, "
            f"f1={swm['f1']:.4f}, precision={swm['precision']:.4f}, "
            f"recall={swm['recall']:.4f}"
        )

    # (display_name, metric_key_prefix) pairs
    rows = [(atlas, atlas) for atlas in ATLAS_LIST]
    if mid_label is not None:
        rows.append((mid_label, "mid"))

    print(f"  {label} per-atlas mean(start, end) accuracy:")
    for display, prefix in rows:
        start_key = f"{prefix}_start"
        end_key = f"{prefix}_end"
        if start_key not in metrics or end_key not in metrics:
            continue
        s_acc = metrics[start_key]["accuracy"]
        e_acc = metrics[end_key]["accuracy"]
        mean_acc = 0.5 * (s_acc + e_acc)
        print(
            f"    {display}: mean_acc={mean_acc:.4f} "
            f"(start={s_acc:.4f}, end={e_acc:.4f})"
        )


def metrics_to_csv_rows(metrics, mid_label=None, fold=None):
    """Convert a metrics dict into long-format rows for CSV writing.

    mid_label:
        None -> no mid layer row
        str  -> a 'mid' row is emitted with this name in the 'atlas' column
                (e.g. 'mid_lobe' / 'mid_yeo').
    """
    fold_str = "single" if fold is None else f"fold{fold}"
    # (atlas_column_value, metric_key_prefix)
    pairs = [(atlas, atlas) for atlas in ATLAS_LIST]
    if mid_label is not None:
        pairs.append((mid_label, "mid"))

    rows = []
    swm = metrics.get("swm")
    if swm is not None:
        rows.append({
            "fold": fold_str,
            "atlas": "swm",
            "position": "binary",
            "accuracy": swm["accuracy"],
            "f1": swm["f1"],
            "precision": swm["precision"],
            "recall": swm["recall"],
        })

    for atlas_name, prefix in pairs:
        for pos in ["start", "end"]:
            key = f"{prefix}_{pos}"
            if key not in metrics:
                continue
            m = metrics[key]
            rows.append({
                "fold": fold_str,
                "atlas": atlas_name,
                "position": pos,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
            })
    return rows


def write_metrics_csv(rows, csv_path):
    """Write rows (list of dicts) to csv_path, overwriting any existing file."""
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
    print(f"Saved best-model test metrics to: {csv_path}")


def mean_std_ignore_nan(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    valid = tensor[~torch.isnan(tensor)]
    if valid.numel() == 0:
        return float("nan"), float("nan")
    return valid.mean().item(), valid.std(unbiased=False).item()


# =====================================================
# One split / one fold
# =====================================================
def run_one_split(args, train_set, val_set, test_set, device, save_path, train_atlases, mid_layer, fold=None):
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("Dataloaders ready")

    model = build_model(args, device)
    print("Model initialized")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_atlases=train_atlases,
            mid_layer=mid_layer,
        )
        print(f"Train Loss: {train_loss:.4f}")
        print_metrics_summary(train_metrics, mid_label=f"mid_{mid_layer}", label="Train")

        val_loss, val_metrics = eval_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            train_atlases=train_atlases,
            mid_layer=mid_layer,
        )
        print(f"Val Loss: {val_loss:.4f}")
        print_metrics_summary(val_metrics, mid_label=f"mid_{mid_layer}", label="Val")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model at epoch {epoch} (Val Loss: {val_loss:.4f})")

    print("\nTesting best model")
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss, test_metrics = eval_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        train_atlases=train_atlases,
        mid_layer=mid_layer,
    )
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics_summary(test_metrics, mid_label=f"mid_{mid_layer}", label="Test")

    return {
        "fold": fold,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "save_path": save_path,
    }


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
            mean_value, std_value = mean_std_ignore_nan(
                [r["test_metrics"][key][metric_name] for r in fold_results]
            )
            parts.append(f"{metric_name}={mean_value:.4f}±{std_value:.4f}")
        print(f"{key}: " + ", ".join(parts))


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

    train_atlases = parse_train_atlases(args)
    print("Atlases used for training loss:", train_atlases)
    print("Metrics will be calculated for all atlases:", ATLAS_LIST)
    print(
        f"Intermediate (mid) layer supervised by: {args.mid_layer} "
        f"({'14 classes from lobe_targets' if args.mid_layer == 'lobe' else '7 classes from atlas_targets.yeo'})"
    )
    mid_layer = args.mid_layer
    mid_label = f"mid_{mid_layer}"

    # k_fold <= 1: keep original train/val/test split.
    if args.k_fold <= 1:
        print("k_fold <= 1, using original train/val/test split")
        train_set, val_set, test_set = build_demo_datasets(
            h5_path=args.h5_path,
            csv_path=args.csv_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            stratify_by_swm=args.stratify_by_swm,
            group_by_subject=args.group_by_subject,
        )
        print("Dataset built")

        save_path = make_save_path(args, fold=None)
        result = run_one_split(
            args=args,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            device=device,
            save_path=save_path,
            train_atlases=train_atlases,
            mid_layer=mid_layer,
            fold=None,
        )

        csv_path = os.path.join(
            os.path.dirname(save_path),
            f"test_metrics_{args.model_type}_{mid_layer}.csv",
        )
        rows = metrics_to_csv_rows(result["test_metrics"], mid_label=mid_label, fold=None)
        write_metrics_csv(rows, csv_path)
        return

    print(f"Using {args.k_fold}-fold cross validation")
    folds = build_demo_kfold_datasets(
        h5_path=args.h5_path,
        csv_path=args.csv_path,
        k_fold=args.k_fold,
        val_ratio=args.val_ratio,
        seed=args.seed,
        shuffle=True,
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
            device=device,
            save_path=save_path,
            train_atlases=train_atlases,
            mid_layer=mid_layer,
            fold=fold,
        )
        fold_results.append(result)

    summarize_kfold_results(fold_results)

    csv_path = os.path.join(
        os.path.dirname(make_save_path(args, fold=fold_results[0]["fold"])),
        f"test_metrics_{args.model_type}_{mid_layer}.csv",
    )
    all_rows = []
    for r in fold_results:
        all_rows.extend(
            metrics_to_csv_rows(r["test_metrics"], mid_label=mid_label, fold=r["fold"])
        )
    write_metrics_csv(all_rows, csv_path)


# =====================================================
# Train / Eval
# =====================================================
def _mid_label_source(mid_layer, pos, atlas_targets, lobe_targets):
    """Look up the supervision label tensor for the mid layer.

    mid_layer == "lobe" -> lobe_targets["lobe_{pos}"]    (14 classes)
    mid_layer == "yeo"  -> atlas_targets["yeo_{pos}"]    (7 classes)
    """
    if mid_layer == "lobe":
        return lobe_targets[f"lobe_{pos}"]
    if mid_layer == "yeo":
        return atlas_targets[f"yeo_{pos}"]
    raise ValueError(f"Unknown mid_layer: {mid_layer}")


def train_one_epoch(model, loader, optimizer, criterion, device, train_atlases, mid_layer):
    model.train()
    total_loss = 0.0
    total_samples = 0

    all_preds = {f"{atlas}_{pos}": [] for atlas in ATLAS_LIST for pos in ["start", "end"]}
    all_labels = {f"{atlas}_{pos}": [] for atlas in ATLAS_LIST for pos in ["start", "end"]}
    all_swm_preds = []
    all_swm_labels = []

    all_mid_preds = {"start": [], "end": []}
    all_mid_labels = {"start": [], "end": []}

    for X, atlas_targets, lobe_targets, subject_ids in loader:
        X = X.to(device, non_blocking=True)
        atlas_targets = {
            k: v.to(device, non_blocking=True)
            for k, v in atlas_targets.items()
        }
        lobe_targets = {
            k: v.to(device, non_blocking=True)
            for k, v in lobe_targets.items()
        }

        # Teacher forcing: feed ground-truth mid labels into atlas conditioning
        # during training. DWM samples (IGNORE_INDEX) get zero mid embeddings
        # inside the model, matching the no-lobe baseline for those rows.
        mid_start_label = _mid_label_source(mid_layer, "start", atlas_targets, lobe_targets)
        mid_end_label = _mid_label_source(mid_layer, "end", atlas_targets, lobe_targets)

        optimizer.zero_grad()
        outputs = model(
            X,
            mid_start_target=mid_start_label,
            mid_end_target=mid_end_label,
        )
        swm_mask = atlas_targets["swm"] == 1

        # All samples participate in SWM/DWM binary classification.
        batch_loss = criterion(outputs["swm"], atlas_targets["swm"])

        swm_pred = outputs["swm"].argmax(dim=1)
        all_swm_preds.append(swm_pred.cpu())
        all_swm_labels.append(atlas_targets["swm"].cpu())

        if swm_mask.sum().item() > 0:
            # Mid-layer loss/metrics. Label source depends on --mid_layer.
            mid_labels = {"start": mid_start_label, "end": mid_end_label}
            for pos in ["start", "end"]:
                out_key = f"mid_{pos}"
                mid_label = mid_labels[pos]

                batch_loss += criterion(
                    outputs[out_key][swm_mask],
                    mid_label[swm_mask],
                )

                preds = outputs[out_key][swm_mask].argmax(dim=1)
                all_mid_preds[pos].append(preds.cpu())
                all_mid_labels[pos].append(mid_label[swm_mask].cpu())

            # Atlas loss: only selected atlases are used for backprop.
            for atlas in train_atlases:
                for pos in ["start", "end"]:
                    key = f"{atlas}_{pos}"
                    batch_loss += criterion(
                        outputs[key][swm_mask],
                        atlas_targets[key][swm_mask],
                    )

            # Atlas metrics: still report all atlases.
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
                all_preds[key],
                all_labels[key],
                average="macro",
            )

    for pos in ["start", "end"]:
        metrics[f"mid_{pos}"] = compute_classification_metrics(
            all_mid_preds[pos],
            all_mid_labels[pos],
            average="macro",
        )

    metrics["swm"] = compute_classification_metrics(
        all_swm_preds,
        all_swm_labels,
        average="binary",
    )

    return avg_loss, metrics


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, train_atlases, mid_layer):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = {f"{atlas}_{pos}": [] for atlas in ATLAS_LIST for pos in ["start", "end"]}
    all_labels = {f"{atlas}_{pos}": [] for atlas in ATLAS_LIST for pos in ["start", "end"]}
    all_swm_preds = []
    all_swm_labels = []

    all_mid_preds = {"start": [], "end": []}
    all_mid_labels = {"start": [], "end": []}

    for X, atlas_targets, lobe_targets, subject_ids in loader:
        X = X.to(device, non_blocking=True)
        atlas_targets = {
            k: v.to(device, non_blocking=True)
            for k, v in atlas_targets.items()
        }
        lobe_targets = {
            k: v.to(device, non_blocking=True)
            for k, v in lobe_targets.items()
        }

        outputs = model(X)
        swm_mask = atlas_targets["swm"] == 1

        # Unified early-stop criterion across all variants:
        # val_loss = SWM_CE + sum_atlas atlas_CE (mid CE is intentionally
        # excluded so the checkpoint selected by best val_loss is the best
        # atlas checkpoint, comparable to train_demo_no_lobe.py). Mid metrics
        # are still computed below for reporting.
        batch_loss = criterion(outputs["swm"], atlas_targets["swm"])

        swm_pred = outputs["swm"].argmax(dim=1)
        all_swm_preds.append(swm_pred.cpu())
        all_swm_labels.append(atlas_targets["swm"].cpu())

        if swm_mask.sum().item() > 0:
            for pos in ["start", "end"]:
                out_key = f"mid_{pos}"
                mid_label = _mid_label_source(mid_layer, pos, atlas_targets, lobe_targets)

                preds = outputs[out_key][swm_mask].argmax(dim=1)
                all_mid_preds[pos].append(preds.cpu())
                all_mid_labels[pos].append(mid_label[swm_mask].cpu())

            # Atlas loss: only selected atlases.
            for atlas in train_atlases:
                for pos in ["start", "end"]:
                    key = f"{atlas}_{pos}"
                    batch_loss += criterion(
                        outputs[key][swm_mask],
                        atlas_targets[key][swm_mask],
                    )

            # Atlas metrics: all atlases.
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
                all_preds[key],
                all_labels[key],
                average="macro",
            )

    for pos in ["start", "end"]:
        metrics[f"mid_{pos}"] = compute_classification_metrics(
            all_mid_preds[pos],
            all_mid_labels[pos],
            average="macro",
        )

    metrics["swm"] = compute_classification_metrics(
        all_swm_preds,
        all_swm_labels,
        average="binary",
    )

    return avg_loss, metrics


if __name__ == "__main__":
    main()
