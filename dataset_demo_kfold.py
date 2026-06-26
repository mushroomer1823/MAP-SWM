# dataset_demo.py

import os
import h5py
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    GroupShuffleSplit,
    train_test_split,
)


ATLAS_LIST = [
    "yeo",
    "DK",
    "Brainnetome",
    "AAL",
    "schaefer_100",
    "Destrieux",
]

# DWM samples have no atlas / lobe endpoint labels, so we fill them with -100
# and rely on CrossEntropyLoss(ignore_index=-100) to skip them.
IGNORE_INDEX = -100


class SWMDemoFiberDataset(Dataset):
    def __init__(
        self,
        h5_path="/data/hyf/swm_identification/data/demo/demo_swm_streamlines.h5",
        csv_path="/data/hyf/swm_identification/data/demo/demo_atlas_start_end_selected.csv",
    ):
        """
        Demo dataset for SWM/DWM fiber classification.

        H5:
            streamlines: [N, 3, 30]
            subject_id: optional, [N]

        CSV:
            rows must be in the same order as H5 streamlines.

            Required columns:
                ifSWM

            For ifSWM == 1:
                atlas_start / atlas_end labels should be valid.
                lobe_start / lobe_end labels should be valid.

            For ifSWM == 0:
                atlas_start / atlas_end can be empty.
                lobe_start / lobe_end can be empty.
                These labels will be set to IGNORE_INDEX.
        """

        self.h5_path = h5_path
        self.csv_path = csv_path

        self.X, self.atlas_targets, self.lobe_targets, self.subject_ids = self._load_all()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        fiber = self.X[idx]

        atlas_target = {
            k: v[idx]
            for k, v in self.atlas_targets.items()
        }

        lobe_target = {
            k: v[idx]
            for k, v in self.lobe_targets.items()
        }

        if self.subject_ids is None:
            subject_id = "unknown"
        else:
            subject_id = self.subject_ids[idx]

        return fiber, atlas_target, lobe_target, subject_id

    @staticmethod
    def _build_label_tensor(col_values, col_name, swm_mask):
        """Vectorized 1-based -> 0-based label conversion with masking.

        For rows where swm_mask is True, the column value must be a positive
        integer; it is decremented by 1. For rows where swm_mask is False the
        result is IGNORE_INDEX, and the column value is not inspected.

        Returns a 1D torch.long tensor of length len(col_values).
        """
        n = len(col_values)
        if n == 0:
            return torch.zeros(0, dtype=torch.long)

        floats = pd.to_numeric(col_values, errors="coerce").to_numpy(dtype=np.float64)

        # NaN check only applies to SWM rows; DWM rows are allowed to be missing.
        nan_in_swm = np.isnan(floats) & swm_mask
        if nan_in_swm.any():
            bad = int(np.argmax(nan_in_swm))
            raise ValueError(f"Missing label in column: {col_name} (row {bad})")

        # Non-integer / non-numeric check on SWM rows.
        non_int_in_swm = (floats != np.floor(floats)) & swm_mask
        if non_int_in_swm.any():
            bad = int(np.argmax(non_int_in_swm))
            raise ValueError(
                f"Invalid label in column {col_name}: {col_values.iloc[bad]} (row {bad})"
            )

        # Replace NaNs in DWM rows with 0 so the cast to int64 is safe; those
        # entries are overwritten by IGNORE_INDEX below.
        safe = np.where(np.isnan(floats), 0.0, floats).astype(np.int64)

        non_positive_in_swm = (safe <= 0) & swm_mask
        if non_positive_in_swm.any():
            bad = int(np.argmax(non_positive_in_swm))
            raise ValueError(
                f"Label in column {col_name} should be 1-based positive integer, "
                f"but got {int(safe[bad])} (row {bad})"
            )

        zero_based = safe - 1
        out = np.where(swm_mask, zero_based, IGNORE_INDEX).astype(np.int64)
        return torch.from_numpy(out)

    def _load_all(self):
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # =====================================================
        # 1. Read H5
        # =====================================================
        with h5py.File(self.h5_path, "r") as f:
            X = f["streamlines"][:]  # [N, 3, 30]

            if "subject_id" in f:
                subject_ids = f["subject_id"][:]
                subject_ids = np.array([
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in subject_ids
                ])
            else:
                subject_ids = None

        # =====================================================
        # 2. Read CSV
        # =====================================================
        df = pd.read_csv(self.csv_path)

        if len(df) != X.shape[0]:
            raise ValueError(
                f"Length mismatch: H5 has {X.shape[0]} streamlines, "
                f"CSV has {len(df)} rows."
            )

        # If both H5 and CSV carry subject_id, verify the ordering matches.
        if subject_ids is not None and "subject_id" in df.columns:
            csv_subject_ids = df["subject_id"].astype(str).values

            if not np.all(csv_subject_ids == subject_ids):
                raise ValueError(
                    "subject_id mismatch between H5 and CSV. "
                    "Please check whether the order is consistent."
                )

        # =====================================================
        # 3. Check required columns
        # =====================================================
        if "ifSWM" not in df.columns:
            raise KeyError("Missing required column in CSV: ifSWM")

        for atlas in ATLAS_LIST:
            start_col = f"{atlas}_start"
            end_col = f"{atlas}_end"

            if start_col not in df.columns:
                raise KeyError(f"Missing column in CSV: {start_col}")

            if end_col not in df.columns:
                raise KeyError(f"Missing column in CSV: {end_col}")

        if "lobe_start" not in df.columns:
            raise KeyError("Missing column in CSV: lobe_start")

        if "lobe_end" not in df.columns:
            raise KeyError("Missing column in CSV: lobe_end")

        # =====================================================
        # 4. SWM / DWM binary label
        # =====================================================
        if_swm_arr = df["ifSWM"].to_numpy()
        if_swm_float = pd.to_numeric(df["ifSWM"], errors="coerce").to_numpy(dtype=np.float64)
        invalid = ~np.isin(if_swm_float, [0.0, 1.0])
        if invalid.any():
            bad = int(np.argmax(invalid))
            raise ValueError(
                f"Invalid ifSWM value at row {bad}: {if_swm_arr[bad]}. "
                f"Expected 0 or 1."
            )
        if_swm = if_swm_float.astype(np.int64)
        swm_mask = if_swm == 1

        atlas_target_tensors = {"swm": torch.from_numpy(if_swm.copy())}

        # =====================================================
        # 5. Atlas endpoint labels (vectorized)
        # =====================================================
        for atlas in ATLAS_LIST:
            for pos in ("start", "end"):
                col = f"{atlas}_{pos}"
                atlas_target_tensors[col] = self._build_label_tensor(
                    df[col], col, swm_mask,
                )

        # =====================================================
        # 6. Lobe endpoint labels (vectorized)
        # =====================================================
        lobe_target_tensors = {}
        for col in ("lobe_start", "lobe_end"):
            lobe_target_tensors[col] = self._build_label_tensor(
                df[col], col, swm_mask,
            )

        # =====================================================
        # 7. Streamline tensor
        # =====================================================
        X = torch.tensor(X, dtype=torch.float32)

        # =====================================================
        # 8. Print summary
        # =====================================================
        n_swm = int(swm_mask.sum())
        n_dwm = int((~swm_mask).sum())

        print("Loaded demo dataset")
        print(f"  H5: {self.h5_path}")
        print(f"  CSV: {self.csv_path}")
        print(f"  Total fibers: {X.shape[0]}")
        print(f"  SWM fibers: {n_swm}")
        print(f"  DWM fibers: {n_dwm}")
        print(f"  Fiber shape: {tuple(X.shape[1:])}")

        if subject_ids is not None:
            print(f"  Subjects in demo: {sorted(set(subject_ids.tolist()))}")

        return X, atlas_target_tensors, lobe_target_tensors, subject_ids


# =====================================================
# Full-dataset builder
# =====================================================
def build_demo_dataset(
    h5_path="/data/hyf/swm_identification/data/demo/demo_swm_streamlines.h5",
    csv_path="/data/hyf/swm_identification/data/demo/demo_atlas_start_end_selected.csv",
):
    """Load the full dataset with no train/val/test split.

    For K-fold training, call this first to get the full dataset, then pass
    it through build_demo_kfold_datasets to materialize per-fold splits.
    """
    return SWMDemoFiberDataset(
        h5_path=h5_path,
        csv_path=csv_path,
    )


# =====================================================
# Single train / val / test split (kept for compatibility)
# =====================================================
def build_demo_datasets(
    h5_path="/data/hyf/swm_identification/data/demo/demo_swm_streamlines.h5",
    csv_path="/data/hyf/swm_identification/data/demo/demo_atlas_start_end_selected.csv",
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    stratify_by_swm=False,
    group_by_subject=False,
):
    """
    Unified train / val / test split with optional SWM stratification and
    subject-level grouping.

    Priority:
        group_by_subject  >  stratify_by_swm  >  random shuffle.

    When group_by_subject=True, stratify_by_swm is ignored (subjects are split
    as whole groups; sklearn.GroupShuffleSplit does not support stratification).
    """

    full_dataset = build_demo_dataset(
        h5_path=h5_path,
        csv_path=csv_path,
    )

    N = len(full_dataset)

    if N == 0:
        raise ValueError("Dataset is empty.")

    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError(
            f"Invalid ratios: train_ratio={train_ratio}, val_ratio={val_ratio}"
        )

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0, got "
            f"{train_ratio + val_ratio}"
        )

    indices = np.arange(N)
    swm_labels = full_dataset.atlas_targets["swm"].cpu().numpy()

    if group_by_subject:
        if full_dataset.subject_ids is None:
            raise ValueError(
                "group_by_subject=True requires subject_id in H5 file, "
                "but this dataset has no subject_id."
            )
        if stratify_by_swm:
            print(
                "Note: stratify_by_swm is ignored because group_by_subject "
                "is enabled (subjects are split as whole groups)."
            )

        groups = np.asarray(full_dataset.subject_ids)

        # Outer split: (train+val) vs test, by subject.
        gss_outer = GroupShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
        train_val_idx, test_idx = next(
            gss_outer.split(indices, swm_labels, groups=groups)
        )

        # Inner split: train vs val from (train+val), by subject.
        rel_val = val_ratio / (train_ratio + val_ratio)
        gss_inner = GroupShuffleSplit(
            n_splits=1, test_size=rel_val, random_state=seed + 1
        )
        tr_rel, val_rel = next(
            gss_inner.split(
                train_val_idx,
                swm_labels[train_val_idx],
                groups=groups[train_val_idx],
            )
        )
        train_idx = train_val_idx[tr_rel]
        val_idx = train_val_idx[val_rel]
        split_mode = "subject-level GroupShuffleSplit"

    elif stratify_by_swm:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            shuffle=True,
            stratify=swm_labels,
        )
        rel_test = test_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test,
            random_state=seed + 1,
            shuffle=True,
            stratify=swm_labels[temp_idx],
        )
        split_mode = "fiber-level stratified split by ifSWM"

    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        split_mode = "fiber-level random split"

    train_set = Subset(full_dataset, np.asarray(train_idx).tolist())
    val_set = Subset(full_dataset, np.asarray(val_idx).tolist())
    test_set = Subset(full_dataset, np.asarray(test_idx).tolist())

    print(f"Split mode: {split_mode}")
    print(f"  Total fibers: {N}")
    print(f"  Train fibers: {len(train_set)}")
    print(f"  Val fibers  : {len(val_set)}")
    print(f"  Test fibers : {len(test_set)}")

    if full_dataset.subject_ids is not None:
        subj = np.asarray(full_dataset.subject_ids)
        train_subs = set(subj[np.asarray(train_idx)].tolist())
        val_subs = set(subj[np.asarray(val_idx)].tolist())
        test_subs = set(subj[np.asarray(test_idx)].tolist())
        print(
            f"  Subjects -> train: {len(train_subs)}, "
            f"val: {len(val_subs)}, test: {len(test_subs)}"
        )
        print(
            f"  Subject overlap -> train∩val: {len(train_subs & val_subs)}, "
            f"train∩test: {len(train_subs & test_subs)}, "
            f"val∩test: {len(val_subs & test_subs)}"
        )

    return train_set, val_set, test_set


# =====================================================
# K-fold helpers
# =====================================================
def _can_stratify(labels, n_splits):
    """Return True if every class has at least n_splits samples.

    StratifiedKFold requires at least n_splits samples per class.
    """
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    return np.all(counts >= n_splits)


def _safe_train_val_split(
    train_val_idx,
    labels,
    val_ratio=0.15,
    seed=42,
    stratify=True,
):
    """Split a fold's train_val_idx into train / val.

    If stratify=True and every class has enough samples, the split is
    stratified by ifSWM. Otherwise it falls back to a plain random split.
    """
    train_val_idx = np.asarray(train_val_idx)

    if val_ratio <= 0:
        return train_val_idx, np.array([], dtype=int)

    if val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be < 1.0, got {val_ratio}")

    y_train_val = labels[train_val_idx]
    stratify_labels = None

    if stratify and _can_stratify(y_train_val, n_splits=2):
        stratify_labels = y_train_val

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels,
    )

    return np.asarray(train_idx), np.asarray(val_idx)


def build_demo_kfold_datasets(
    h5_path="/data/hyf/swm_identification/data/demo/demo_swm_streamlines.h5",
    csv_path="/data/hyf/swm_identification/data/demo/demo_atlas_start_end_selected.csv",
    k_fold=5,
    val_ratio=0.15,
    seed=42,
    shuffle=True,
    stratify_by_swm=True,
    group_by_subject=False,
):
    """Build K-fold cross-validation splits.

    Returns:
        folds: list[dict]

    Each fold is a dict:
        {
            "fold": fold index, 1-based,
            "train_set": Subset,
            "val_set": Subset,
            "test_set": Subset,
            "train_indices": list[int],
            "val_indices": list[int],
            "test_indices": list[int],
        }

    Args:
        k_fold:
            Number of folds.

        val_ratio:
            Fraction of each fold's train_val portion used as validation.
            (test is the held-out fold; val is carved out of the remaining
            train_val pool.)

        stratify_by_swm:
            Whether to stratify the outer KFold on ifSWM. Recommended True when
            both SWM and DWM samples are present.

        group_by_subject:
            Whether to do subject-level GroupKFold. When True, all fibers of a
            given subject stay in the same fold. Requires subject_id in the H5.
    """

    full_dataset = build_demo_dataset(
        h5_path=h5_path,
        csv_path=csv_path,
    )

    N = len(full_dataset)

    if N == 0:
        raise ValueError("Dataset is empty.")

    if k_fold < 2:
        raise ValueError(f"k_fold must be >= 2 for K-fold CV, got {k_fold}")

    if k_fold > N:
        raise ValueError(f"k_fold cannot be larger than dataset size: {k_fold} > {N}")

    indices = np.arange(N)
    swm_labels = full_dataset.atlas_targets["swm"].cpu().numpy()

    # -----------------------------------------------------
    # 1. Build outer train_val / test folds
    # -----------------------------------------------------
    if group_by_subject:
        if full_dataset.subject_ids is None:
            raise ValueError(
                "group_by_subject=True requires subject_id in H5 file, "
                "but this dataset has no subject_id."
            )

        groups = np.asarray(full_dataset.subject_ids)
        splitter = GroupKFold(n_splits=k_fold)
        split_iter = splitter.split(indices, swm_labels, groups=groups)
        split_mode = "subject-level GroupKFold"

    elif stratify_by_swm and _can_stratify(swm_labels, k_fold):
        splitter = StratifiedKFold(
            n_splits=k_fold,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )
        split_iter = splitter.split(indices, swm_labels)
        split_mode = "fiber-level StratifiedKFold by ifSWM"

    else:
        splitter = KFold(
            n_splits=k_fold,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )
        split_iter = splitter.split(indices)
        split_mode = "fiber-level KFold"

        if stratify_by_swm:
            print(
                "Warning: cannot use StratifiedKFold because at least one class "
                f"has fewer than k_fold={k_fold} samples. Falling back to KFold."
            )

    # -----------------------------------------------------
    # 2. Within each fold, split train_val into train / val
    # -----------------------------------------------------
    folds = []

    print("K-fold split:")
    print(f"  Split mode: {split_mode}")
    print(f"  Total fibers: {N}")
    print(f"  K: {k_fold}")
    print(f"  Val ratio within train_val: {val_ratio}")

    for fold_id, (train_val_idx, test_idx) in enumerate(split_iter, start=1):
        train_idx, val_idx = _safe_train_val_split(
            train_val_idx=train_val_idx,
            labels=swm_labels,
            val_ratio=val_ratio,
            seed=seed + fold_id,
            stratify=stratify_by_swm and not group_by_subject,
        )

        train_set = Subset(full_dataset, train_idx.tolist())
        val_set = Subset(full_dataset, val_idx.tolist())
        test_set = Subset(full_dataset, np.asarray(test_idx).tolist())

        print(
            f"  Fold {fold_id}: "
            f"train={len(train_set)}, "
            f"val={len(val_set)}, "
            f"test={len(test_set)}"
        )

        folds.append({
            "fold": fold_id,
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": np.asarray(test_idx).tolist(),
        })

    return folds


if __name__ == "__main__":
    # =====================================================
    # 1. Single split smoke test
    # =====================================================
    train_set, val_set, test_set = build_demo_datasets(
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    )

    print()
    print("Normal split dataset sizes:")
    print("  train:", len(train_set))
    print("  val  :", len(val_set))
    print("  test :", len(test_set))

    fiber, atlas_target, lobe_target, subject_id = train_set[0]

    print()
    print("One sample:")
    print("  fiber shape:", fiber.shape)
    print("  subject_id:", subject_id)

    print()
    print("Atlas target keys:", atlas_target.keys())
    print("  swm:", atlas_target["swm"])
    print("  yeo_start:", atlas_target["yeo_start"])
    print("  yeo_end:", atlas_target["yeo_end"])

    print()
    print("Lobe target keys:", lobe_target.keys())
    print("  lobe_start:", lobe_target["lobe_start"])
    print("  lobe_end:", lobe_target["lobe_end"])

    # =====================================================
    # 2. K-fold split smoke test
    # =====================================================
    print()
    print("Testing K-fold split:")
    folds = build_demo_kfold_datasets(
        k_fold=5,
        val_ratio=0.15,
        seed=42,
        shuffle=True,
        stratify_by_swm=True,
        group_by_subject=False,
    )

    first_fold = folds[0]
    print()
    print("First fold sizes:")
    print("  train:", len(first_fold["train_set"]))
    print("  val  :", len(first_fold["val_set"]))
    print("  test :", len(first_fold["test_set"]))
