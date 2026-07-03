# compute_overlap.py
#
# Compute overlap matrices M_{source}_to_{atlas}.npy where
#   M[i, j] = P(atlas ROI == j+1 | source class == i+1)
# estimated by voxel counts on a common cortical region shared by all atlases
# (intersection of mask AND every atlas's labelled voxels).
#
# Output (per --source):
#   atlas_overlap/{source}/M_{source}_to_{atlas}.npy
#   atlas_overlap/{source}/common_region.npy  (uint8 mask)
#   atlas_overlap/{source}/summary.txt

import argparse
import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

ATLAS_DIR = "data/atlases"
OUT_DIR = "./atlas_overlap"
os.makedirs(OUT_DIR, exist_ok=True)

ATLAS_FILES = {
    "yeo":          "yeo/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
    "lobe":         "lobe/lobe.nii.gz",
    "DK":           "DK/Desikan_space-MNI152NLin6_res-1x1x1.nii.gz",
    "Brainnetome":  "Brainnetome/BN_Atlas_246_1mm.nii.gz",
    "AAL":          "AAL/AAL_space-MNI152NLin6_res-1x1x1.nii.gz",
    "schaefer_100": "schaefer_100/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz",
    "Destrieux":    "Destrieux/Destrieux_space-MNI152NLin6_res-1x1x1.nii.gz",
}

ATLAS_DIMS = {
    "yeo": 7, "lobe": 14, "DK": 70, "Brainnetome": 246,
    "AAL": 116, "schaefer_100": 100, "Destrieux": 75,
}

MASK_PATH = "mask.nii"

# Reference grid: use any 182x218x182 atlas. Picking DK because all five
# non-yeo atlases already share its grid (verified by affine inspection).
REF_NAME = "DK"


def load_as_int(path):
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 4:
        # yeo has a trailing singleton axis
        data = data[..., 0]
        img = nib.Nifti1Image(data, img.affine, img.header)
    return img, data.astype(np.int32)


def resample_to_ref(img, ref_shape, ref_affine):
    """Nearest-neighbour resampling so discrete labels are preserved."""
    if img.shape == ref_shape and np.allclose(img.affine, ref_affine):
        return img.get_fdata().astype(np.int32)
    resampled = resample_from_to(img, (ref_shape, ref_affine), order=0)
    return resampled.get_fdata().astype(np.int32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute overlap matrices M_{source}_to_{atlas}."
    )
    parser.add_argument(
        "--source", type=str, default="yeo",
        choices=sorted(ATLAS_FILES.keys()),
        help="Source atlas whose classes condition the overlap rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source
    if source not in ATLAS_FILES:
        raise ValueError(f"Unknown source '{source}'. Available: {list(ATLAS_FILES)}")

    out_dir = os.path.join(OUT_DIR, source)
    os.makedirs(out_dir, exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"Source atlas: {source}  (dim={ATLAS_DIMS[source]})")
    log(f"Output dir:   {out_dir}")

    # --- 1. reference grid ----------------------------------------------
    ref_img = nib.load(os.path.join(ATLAS_DIR, ATLAS_FILES[REF_NAME]))
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine
    log(f"Reference grid from '{REF_NAME}': shape={ref_shape}")
    log(f"Reference affine:\n{ref_affine}")

    # --- 2. load + resample every atlas to reference grid ---------------
    atlas_data = {}
    for name, rel_path in ATLAS_FILES.items():
        path = os.path.join(ATLAS_DIR, rel_path)
        img, _ = load_as_int(path)
        data = resample_to_ref(img, ref_shape, ref_affine)

        u = np.unique(data)
        n_nonzero_labels = int((u > 0).sum())
        max_label = int(u.max())
        log(
            f"[{name:12s}] shape={data.shape} "
            f"nonzero_labels={n_nonzero_labels} max_label={max_label} "
            f"expected_dim={ATLAS_DIMS[name]}"
        )
        if max_label > ATLAS_DIMS[name]:
            log(
                f"  WARNING: max_label {max_label} > expected_dim {ATLAS_DIMS[name]}. "
                f"Out-of-range labels will be ignored when building M."
            )
        atlas_data[name] = data

    # --- 3. load + resample mask ----------------------------------------
    mask_img = nib.load(os.path.join(ATLAS_DIR, MASK_PATH))
    mask_resampled = resample_to_ref(mask_img, ref_shape, ref_affine)
    mask_bool = mask_resampled > 0
    log(f"[mask]       shape={mask_bool.shape} positive_voxels={int(mask_bool.sum())}")

    # --- 4. common region = mask AND every atlas labelled ---------------
    common = mask_bool.copy()
    log("\nBuilding common region:")
    log(f"  start (mask): {int(common.sum())}")
    for name, data in atlas_data.items():
        before = int(common.sum())
        common &= (data > 0)
        after = int(common.sum())
        pct = 100.0 * after / max(before, 1)
        log(f"  ∩ {name:12s}: {after} ({pct:.2f}% of previous)")
    log(f"\nFinal common region size: {int(common.sum())} voxels")

    np.save(os.path.join(out_dir, "common_region.npy"), common.astype(np.uint8))
    log(f"Saved common region mask -> {os.path.join(out_dir, 'common_region.npy')}")

    # --- 5. build M for each atlas --------------------------------------
    src_vals = atlas_data[source][common]
    src_dim = ATLAS_DIMS[source]

    log("\nComputing overlap matrices:")
    for name, data in atlas_data.items():
        n_roi = ATLAS_DIMS[name]
        atlas_vals = data[common]

        # Joint histogram on (source_class, atlas_class); both are >=1 inside
        # `common` by construction, but clamp defensively in case the atlas
        # file has labels outside [1, n_roi].
        valid = (
            (src_vals >= 1) & (src_vals <= src_dim) &
            (atlas_vals >= 1) & (atlas_vals <= n_roi)
        )
        sv = src_vals[valid] - 1
        av = atlas_vals[valid] - 1
        flat = sv.astype(np.int64) * n_roi + av.astype(np.int64)
        joint = np.bincount(flat, minlength=src_dim * n_roi).reshape(src_dim, n_roi)

        src_counts = joint.sum(axis=1, keepdims=True)
        M = joint / np.maximum(src_counts, 1)
        M = M.astype(np.float32)

        # Check empty source classes (a row with no voxels in the common
        # region will produce an all-zero row of M).
        empty_rows = np.where(src_counts.squeeze() == 0)[0]
        if len(empty_rows) > 0:
            log(f"  [{name:12s}] WARN: empty {source} classes (no voxels): {empty_rows.tolist()}")

        row_sums = M.sum(axis=1)
        log(
            f"  [{name:12s}] M shape={M.shape}  "
            f"row_sum_min={row_sums.min():.6f}  row_sum_max={row_sums.max():.6f}"
        )

        out_path = os.path.join(out_dir, f"M_{source}_to_{name}.npy")
        np.save(out_path, M)
        log(f"    saved -> {out_path}")

    # --- 6. write summary -----------------------------------------------
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nSummary written to {os.path.join(out_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()
