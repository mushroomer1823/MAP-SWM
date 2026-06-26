# Light Multi-Atlas SWM

Deep learning model for identifying superficial white matter (SWM) streamlines and predicting their endpoint regions across multiple brain atlases simultaneously, with a lightweight anatomical bottleneck design.

## What the model does

For each input streamline (a 3D fiber tract, sampled as `[3, 30]` points), the model jointly predicts:

1. **SWM vs DWM** — binary classification.
2. **Mid-layer anatomy** — a coarse anatomical label per endpoint (Yeo 7 networks or 14 hemisphere lobes).
3. **Per-atlas endpoint ROIs** — the start/end region of the streamline in 6 atlases (Yeo, DK, Brainnetome, AAL, Schaefer-100, Destrieux).

## Design idea

Rather than training six independent atlas heads from raw features, the model uses the mid-layer prediction as an **anatomical bottleneck** that injects prior knowledge into every atlas head:

```
streamline ─► backbone ─► global feature
                              │
endpoint  ──► endpoint MLP ──┤
                              ├─► mid head ─► P(yeo / lobe)
                              │         │
                              │         └─► anatomical prior (via overlap matrix M)
                              │                       │
                              └─► atlas heads ◄───────┘  (residual, gated)
```

Two ingredients make this work:

- **Precomputed overlap matrices `M`**: `M[i, j] = P(atlas_ROI = j | mid_class = i)`, estimated offline from atlas NIfTI files on a shared cortical mask (`atlas_overlap/compute_overlap.py`). Each `M` becomes a registered buffer on the model — it ships with the weights, no path leakage at inference.
- **Gated residual prior**: each atlas head outputs `atlas_logits = base_head(z) + sigmoid(gate) · prior_delta`. The `gate` is a per-(atlas, position) learnable scalar; the model can open it where the prior helps and close it where it doesn't. Gradient from the prior path is isolated from the mid head, so atlas-CE never corrupts the mid supervision.

The **lightweight variant** (recommended) routes endpoint features only through the mid head, so the final atlas classifiers see just `[global_feature, mid_bottleneck]`. Combined with prototype classifiers (cosine prototypes in a small projected space) instead of giant `Linear(d, n_roi)` heads, the model stays small enough to deploy.

Configurable knobs:

- `--endpoint_usage {all, mid_only, none}` — how endpoint features feed downstream heads.
- `--classifier_head {linear, cosine, prototype}` — atlas head type.
- `--prior_mode {none, overlap_log, adapter, hybrid}` — how the prior delta is computed.
- `--model_scale {full, light, tiny, custom}` — preset dimension bundles.
- `--no_prior` — ablation: freezes gate to 0 and skips mid CE.

## Project layout

```
dataset_demo_kfold.py              # Dataset, single split, K-fold (subject/stratified/random)
models/model_anatomical_prior.py   # UnifiedSWMNet + prior adapters + prototype heads
networks/                          # Point-cloud backbones (PointNet, PointNet++, DGCNN, PointMLP)
train/train_anatomical_prior.py    # Full trainer with metrics, k-fold, logging
atlas_overlap/
  compute_overlap.py               # Build M_{mid}_to_{atlas}.npy from atlas NIfTIs
  {yeo,lobe}/                      # Precomputed overlap matrices
```

## Usage

```bash
# 1. Precompute overlap matrices (one-time, needs atlas NIfTI files)
python atlas_overlap/compute_overlap.py --source yeo
python atlas_overlap/compute_overlap.py --source lobe

# 2. Train (recommended lightweight configuration)
python train/train_anatomical_prior.py \
    --model_type pointnet \
    --model_scale light \
    --endpoint_usage mid_only \
    --mid_layer lobe \
    --prior_mode adapter \
    --classifier_head prototype \
    --lambda_mid 0.3 \
    --gate_init 0 \
    --temperature 1.5 \
    --k_fold 5 \
    --group_by_subject
```

Each run writes to a timestamped folder under `--result_root` with the config, checkpoints, per-epoch metrics, per-subject and pair-level test metrics, convergence curves, and learned gate trajectories.

Common ablations:

```bash
# Full baseline, no prior
python train/train_anatomical_prior.py --model_scale full --endpoint_usage all --classifier_head linear --no_prior

# Lightweight bottleneck with adapter prior
python train/train_anatomical_prior.py --model_scale light --endpoint_usage mid_only --classifier_head prototype --prior_mode adapter

# Tiny deployable
python train/train_anatomical_prior.py --model_scale tiny --endpoint_usage mid_only --classifier_head prototype --prior_mode adapter
```

## Atlases

| Atlas         | # ROIs | Description                                |
|---------------|--------|--------------------------------------------|
| Yeo           | 7      | Resting-state functional networks          |
| DK            | 70     | Desikan-Killiany gyral parcellation        |
| Brainnetome   | 246    | Connectivity-based parcellation            |
| AAL           | 116    | Automated Anatomical Labeling              |
| Schaefer-100  | 100    | Functional parcellation (100 parcels)      |
| Destrieux     | 75     | Sulcal/gyral anatomical parcellation       |
