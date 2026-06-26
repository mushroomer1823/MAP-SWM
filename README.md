# Multi-Atlas Superficial White Matter (SWM) Identification

Identify superficial white matter (SWM) vs. deep white matter (DWM) streamlines and predict their anatomical endpoint regions across multiple brain atlases simultaneously, using a deep learning model with a gated anatomical prior.

## Overview

The project takes brain white matter streamlines (3D fiber tracts reconstructed from diffusion MRI) and performs:

1. **Binary classification** — SWM vs. DWM
2. **Multi-atlas endpoint mapping** — predict which anatomical ROI each fiber endpoint belongs to, across 6 atlases (Yeo 7 networks, DK, Brainnetome, AAL, Schaefer 100, Destrieux)
3. **Mid-layer labeling** — predict a coarse anatomical label (7 Yeo networks or 14 hemisphere-lobes) for each endpoint

The key idea is a **gated residual anatomical prior**: a coarse "mid layer" prediction (e.g., Yeo network) is converted into an atlas-ROI prior via precomputed overlap matrices, then blended with the direct prediction through a per-head learnable gate. The model can learn to ignore the prior when it is unhelpful for a particular atlas.

## Project Structure

```
multi-atlas-swm/
├── dataset_demo_kfold.py          # Dataset class and data splitting utilities
├── models/
│   └── model_anatomical_prior.py  # UnifiedSWMNet model definition
├── networks/
│   ├── pointnet.py                # PointNet backbone
│   ├── pointnetpp.py              # PointNet++ backbone
│   ├── DGCNN.py                   # DGCNN backbone
│   └── pointMLP.py                # PointMLP backbone
├── train/
│   └── train_anatomical_prior.py  # Training script (main entry point)
└── atlas_overlap/
    ├── compute_overlap.py         # Precompute overlap matrices from atlas NIfTI files
    ├── yeo/                       # Ouput: M_yeo_to_{atlas}.npy matrices
    └── lobe/                      # Output: M_lobe_to_{atlas}.npy matrices
```

## Workflow

### Step 1: Precompute Overlap Matrices (one-time preprocessing)

Run `atlas_overlap/compute_overlap.py` to compute conditional probability matrices from atlas NIfTI files:

```
python atlas_overlap/compute_overlap.py --source yeo
python atlas_overlap/compute_overlap.py --source lobe
```

This loads multiple brain atlas parcellations, resamples them to a common reference grid (182×218×182 at 1mm³ isotropic), finds the intersection of labeled voxels across all atlases ("common cortical region"), and computes:

**M\[i, j\] = P(atlas_ROI = j | mid_class = i)**

estimated by voxel counts, saved as `.npy` files. Results are stored under `atlas_overlap/{yeo,lobe}/`.

### Step 2: Prepare Data

Data consists of two files:

- **H5 file**: streamlines array of shape `[N, 3, 30]` (N fibers, each with 30 sample points in 3D space), plus optional `subject_id`
- **CSV file**: Same ordering as H5, containing:
  - `ifSWM` — binary label (1=SWM, 0=DWM)
  - `{atlas}_start` / `{atlas}_end` — 1-based ROI labels for each of 6 atlases (valid only for SWM fibers)
  - `lobe_start` / `lobe_end` — 1-based lobe labels (valid only for SWM fibers)

The dataset class `SWMDemoFiberDataset` handles:
- Converting 1-based labels to 0-based
- Setting DWM fiber labels to `-100` (ignored in loss computation)
- Data splitting with support for random, stratified-by-SWM, and subject-level grouping strategies
- K-fold cross validation

### Step 3: Train the Model

Run `train/train_anatomical_prior.py`:

```
python train/train_anatomical_prior.py \
    --model_type dgcnn \
    --mid_layer yeo \
    --global_feat_dim 1024 \
    --batch_size 2048 \
    --epochs 50 \
    --k_fold 5 \
    --group_by_subject \
    --gpu 2
```

Key parameters:

| Parameter | Description |
|---|---|
| `--model_type` | Backbone: `pointnet`, `pointnet++`, `dgcnn`, `pointmlp` |
| `--mid_layer` | Mid-layer source: `yeo` (7 classes) or `lobe` (14 classes) |
| `--k_fold` | K for cross-validation (≤1 uses single split) |
| `--group_by_subject` | Keep all fibers of a subject in the same split |
| `--no_prior` | Ablation: freeze gate to 0, disabling the anatomical prior |
| `--gate_init` | Initial raw gate value (default -6 → sigmoid≈0.0025) |
| `--lambda_mid` | Weight of mid-layer CE loss |
| `--temperature` | Softmax temperature for prior softening (<1 sharpens, >1 softens) |

### Step 4: Outputs

Each run creates a timestamped directory under `--result_root` containing:

- `config.json` — all arguments
- `best_model.pth` / `final_model.pth` — model checkpoints
- `epoch_metrics_*.csv` — per-epoch train/val metrics for every head
- `test_metrics.csv` — per-head test metrics (accuracy, precision, recall, F1)
- `test_metrics_per_subject_*.csv` — subject-level aggregated metrics
- `test_metrics_pair_*.csv` — pair-level metrics (start×end composite class)
- `convergence_*.png` — loss, F1, and accuracy curves
- `prior_gate_*.csv` — learned gate values per epoch
- `result_summary.json` / `kfold_result_summary.json`
- `runtime.json` — training wall-clock time

## Model Architecture

```
Fiber (B, 3, 30)
    │
    ├─► Backbone (PointNet/DGCNN/etc.) ─► global_feat (B, 1024)
    │
    ├─► Endpoint MLP (optional) ─► start_feat, end_feat (B, 256 each)
    │
    └─► [global_feat | start_feat | end_feat] = z (B, fused_dim)
            │
            ├─► SWM Head ─► swm_logits (B, 2)
            │
            ├─► Mid Head (start/end) ─► mid_logits (B, 7 or 14)
            │       │
            │       └─► softmax(mid_logits/τ) @ M ─► log_prior (anatomical prior)
            │
            └─► Atlas Heads (start/end × 6 atlases) ─► raw_logits
                    │
                    └─► raw_logits + sigmoid(gate) × log_prior ─► final_logits
```

The gated residual formulation:

```
atlas_logits = base_head(z) + sigmoid(gate) × log( softmax(mid_logits/τ) @ M )
```

- **gate**: per-(atlas, position) learnable scalar, initialized at -6 (≈0.0025 effective weight)
- **gradient isolation**: mid_logits are detached when forming the prior, so atlas CE does not backpropagate into the mid head through the prior path
- The model can learn to open the gate (positive drift) when the anatomical prior helps, or keep it closed (negative drift) when it doesn't

## Atlases

| Atlas | # ROIs | Description |
|---|---|---|
| Yeo | 7 | Resting-state functional networks |
| DK (Desikan-Killiany) | 70 | Gyral-based anatomical parcellation |
| Brainnetome | 246 | Connectivity-based parcellation |
| AAL | 116 | Automated Anatomical Labeling |
| Schaefer 100 | 100 | Functional parcellation (100 parcels) |
| Destrieux | 75 | Sulcal/gyral anatomical parcellation |

## Light Anatomical Bottleneck version

This version adds a deployable lightweight hierarchical model for multi-atlas SWM classification.
The recommended configuration is:

```bash
python train/train_anatomical_prior.py \
  --model_type pointnet \
  --model_scale light \
  --endpoint_usage mid_only \
  --mid_layer lobe \
  --prior_mode adapter \
  --classifier_head prototype \
  --lambda_mid 0.3 \
  --gate_init 0 \
  --temperature 1.5
```

Key switches:

- `--endpoint_usage all`: original-style setting; endpoint features enter final atlas heads directly.
- `--endpoint_usage mid_only`: endpoint features only predict the intermediate anatomical layer; final atlas heads receive streamline global feature + compact mid-layer bottleneck.
- `--endpoint_usage none`: no endpoint encoder.
- `--classifier_head prototype`: replaces large linear final atlas classifiers with compact cosine prototype classifiers.
- `--prior_mode adapter`: uses a learnable adapter from mid probability + overlap prior to residual atlas logits.
- `--model_scale full/light/tiny`: presets for feature dimensions.

Recommended ablations:

```bash
# Strong full baseline
python train/train_anatomical_prior.py --model_scale full --endpoint_usage all --classifier_head linear --no_prior

# Lightweight bottleneck with prior adapter and prototype heads
python train/train_anatomical_prior.py --model_scale light --endpoint_usage mid_only --classifier_head prototype --prior_mode adapter

# Clean bottleneck ablation: endpoint cannot affect final heads because the prior/mid bottleneck is disabled
python train/train_anatomical_prior.py --model_scale light --endpoint_usage mid_only --classifier_head prototype --no_prior

# Tiny deployable version
python train/train_anatomical_prior.py --model_scale tiny --endpoint_usage mid_only --classifier_head prototype --prior_mode adapter
```
