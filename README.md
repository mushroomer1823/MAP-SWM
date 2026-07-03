# MAP-SWM 

MAP-SWM (Multi-Atlas Prior-guided Superficial White Matter extraction frameword) provides a lightweight deep learning framework for superficial white matter (SWM) streamline identification and multi-atlas endpoint classification.

Given an input streamline represented as a 3D point sequence, the model jointly predicts:

1. whether the streamline belongs to SWM or non-SWM fibers;
2. coarse mid-level anatomical labels for the streamline endpoints;
3. atlas-specific endpoint regions across multiple cortical parcellations, including Yeo, DK, Brainnetome, AAL, Schaefer-100, and Destrieux.

The model uses a hierarchical multi-task design. A mid-level anatomical bottleneck is introduced to provide anatomical prior information for atlas-specific endpoint prediction. In the recommended lightweight setting, endpoint features are used only for mid-level prediction, while the final atlas classifiers combine the global streamline representation with the learned anatomical bottleneck. This design reduces model complexity while preserving multi-atlas prediction capability.

## Project structure

```text
dataset_demo_kfold.py              # Dataset loading and split utilities
models/model_anatomical_prior.py   # Main model implementation
networks/                          # Point-cloud backbone networks
train/train_anatomical_prior.py    # Training and evaluation script
atlas_overlap/                     # Anatomical overlap matrix computation
```

## Recommended training command

```bash
python train/train_anatomical_prior.py \
    --model_type dgcnn \
    --model_scale light \
    --endpoint_usage mid_only \
    --mid_layer lobe \
    --prior_mode adapter \
    --classifier_head prototype \
    --lambda_mid 0.3 \
    --gate_init 0 \
    --temperature 1.5 \
    --group_by_subject
```

The training results, including configuration files, checkpoints, metrics, convergence curves, and test summaries, will be saved under the specified result directory.
