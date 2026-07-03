# inference_benchmark.py
#
# Run a trained MAP-SWM model on whole-brain .tck file(s), measure FLOPs and
# per-subject inference time, and save the resulting SWM mask + per-atlas /
# per-class .tck files.
#
# Usage examples:
#   python inference_benchmark.py \
#       --ckpt path/to/best_model.pth \
#       --input path/to/subject01.tck \
#       --output_dir ./inference_out
#
#   python inference_benchmark.py \
#       --ckpt path/to/best_model.pth \
#       --input path/to/tck_dir \
#       --output_dir ./inference_out \
#       --batch_size 4096
#
# The script auto-detects whether --input is a single .tck file or a directory
# containing multiple .tck files.

import argparse
import json
import os
import sys
import time
from glob import glob

import numpy as np
import nibabel as nib
import nibabel.streamlines as nbs
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from dataset_demo_kfold import ATLAS_LIST  # noqa: E402
from models.model_anatomical_prior import UnifiedSWMNet  # noqa: E402


MID_DIMS = {"yeo": 7, "lobe": 14}
ATLAS_ROI_DIMS = {
    "yeo": 7, "DK": 70, "Brainnetome": 246,
    "AAL": 116, "schaefer_100": 100, "Destrieux": 75,
}


# =====================================================
# CLI
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference + FLOPs/timing benchmark for MAP-SWM."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to the trained model checkpoint (.pth) saved by "
             "train_anatomical_prior.py. Must contain both 'model_state_dict' "
             "and 'args'.",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to a single whole-brain .tck file OR a directory "
             "containing multiple .tck files. The script auto-detects the "
             "mode.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where per-subject outputs (swm.tck + per-atlas tcks) "
             "and a benchmark.json summary will be written.",
    )
    parser.add_argument(
        "--overlap_dir", type=str, default=None,
        help="Override the overlap-matrix directory recorded in the "
             "checkpoint's args. Useful when the model was trained on a "
             "different machine. The path should be the parent containing "
             "{yeo,lobe}/ subfolders.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--batch_size", type=int, default=4096,
        help="Batch size during inference. Larger values amortize per-batch "
             "Python overhead and give the most representative throughput "
             "numbers on GPU.",
    )
    parser.add_argument(
        "--num_points", type=int, default=30,
        help="Number of equidistant points to resample each streamline to. "
             "Must match the value used during training (default 30).",
    )
    parser.add_argument(
        "--warmup_batches", type=int, default=2,
        help="Number of warmup forward passes before timing starts. Excludes "
             "first-batch CUDA kernel compilation / cuBLAS autotune overhead.",
    )
    parser.add_argument(
        "--skip_per_class_tck", action="store_true",
        help="Skip writing per-(atlas, class) .tck files. Only swm.tck is "
             "saved. Useful when you only need timing and FLOPs.",
    )
    parser.add_argument(
        "--measure_flops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to measure FLOPs via torch.utils.flop_counter. "
             "Disable with --no-measure_flops if it is unavailable.",
    )
    return parser.parse_args()


# =====================================================
# I/O
# =====================================================
def is_tck_file(p):
    return os.path.isfile(p) and p.lower().endswith(".tck")


def collect_tck_inputs(input_path):
    """Return a list of (subject_name, tck_path) tuples."""
    if is_tck_file(input_path):
        subj = os.path.splitext(os.path.basename(input_path))[0]
        return [(subj, input_path)]

    if os.path.isdir(input_path):
        tcks = sorted(glob(os.path.join(input_path, "*.tck")))
        if not tcks:
            raise FileNotFoundError(
                f"No .tck files found in directory: {input_path}"
            )
        return [
            (os.path.splitext(os.path.basename(p))[0], p) for p in tcks
        ]

    raise FileNotFoundError(
        f"--input does not exist or is not a .tck file / directory: "
        f"{input_path}"
    )


def load_tck(tck_path):
    """Load a .tck file and return (streamlines_list, header).

    Each streamline is a numpy array of shape [n_i, 3] with raw (variable)
    point counts. The header is preserved so we can write output tcks in the
    same reference space.
    """
    tractogram_file = nbs.load(tck_path)
    if not isinstance(tractogram_file, nbs.TckFile):
        raise ValueError(f"Not a .tck file: {tck_path}")
    streamlines = list(tractogram_file.streamlines)
    header = dict(tractogram_file.header)
    return streamlines, header


def save_tck(streamlines, header, out_path):
    """Save a list of [n_i, 3] streamlines to a .tck file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tractogram = nbs.Tractogram(
        streamlines=streamlines,
        affine_to_rasmm=np.eye(4),
    )
    tck = nbs.TckFile(tractogram, header=header)
    tck.save(out_path)


# =====================================================
# Preprocessing: resample to N equidistant points
# =====================================================
def resample_streamline(points, n_target):
    """Resample one streamline to `n_target` equidistant points along
    normalized arc length using linear interpolation.

    Args:
        points: np.ndarray of shape [n, 3]
        n_target: int

    Returns:
        np.ndarray of shape [n_target, 3]
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        raise ValueError("Empty streamline.")
    if n == 1:
        return np.repeat(pts, n_target, axis=0).astype(np.float32)

    # Cumulative arc-length parameterization in [0, 1].
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        # Degenerate: all points coincide.
        return np.repeat(pts[:1], n_target, axis=0).astype(np.float32)
    cum = cum / total

    targets = np.linspace(0.0, 1.0, n_target)
    out = np.empty((n_target, 3), dtype=np.float32)
    for d in range(3):
        out[:, d] = np.interp(targets, cum, pts[:, d])
    return out


def preprocess_streamlines(streamlines, n_target):
    """Resample every streamline and stack into a [B, 3, N] float32 array.

    Returns:
        tensor_input: np.ndarray of shape [B, 3, N]
        valid_indices: np.ndarray of indices into the input list that produced
            a valid resampled streamline (degenerate ones are skipped).
    """
    out = []
    valid = []
    for i, sl in enumerate(streamlines):
        try:
            r = resample_streamline(sl, n_target)
        except ValueError:
            continue
        # Transpose to (3, N) to match the model's expected layout.
        out.append(r.T.astype(np.float32))
        valid.append(i)
    if not out:
        raise ValueError("No valid streamlines after resampling.")
    return np.stack(out, axis=0), np.asarray(valid, dtype=np.int64)


# =====================================================
# Model loading
# =====================================================
def _resolve_overlap_dir(args_dict, override):
    if override is not None:
        parent = override
    else:
        parent = args_dict.get("overlap_dir")
        if parent is None:
            raise KeyError(
                "Checkpoint args has no 'overlap_dir' and --overlap_dir was "
                "not provided."
            )
    mid_layer = args_dict["mid_layer"]
    return os.path.join(parent, mid_layer)


def build_model_from_ckpt(ckpt_path, device, overlap_override=None):
    """Reconstruct UnifiedSWMNet from a training checkpoint and load weights.

    Returns:
        model (in eval mode), args_dict, mid_layer
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            "Checkpoint does not contain 'model_state_dict'. Expecting the "
            "training-time format produced by train_anatomical_prior.py."
        )
    if "args" not in ckpt:
        raise ValueError(
            "Checkpoint does not contain training 'args'. Cannot reconstruct "
            "model configuration."
        )

    a = dict(ckpt["args"])
    mid_layer = a["mid_layer"]
    overlap_dir = _resolve_overlap_dir(a, overlap_override)

    model = UnifiedSWMNet(
        atlas_roi_dims=ATLAS_ROI_DIMS,
        backbone=a["model_type"],
        mid_dim=MID_DIMS[mid_layer],
        mid_source=mid_layer,
        overlap_dir=overlap_dir,
        temperature=a.get("temperature", 1.0),
        gate_init=a.get("gate_init", 0.0),
        global_feat_dim=a["global_feat_dim"],
        endpoint_dim=a["endpoint_dim"],
        swm_hidden_dim=a["swm_hidden_dim"],
        endpoint_usage=a.get("endpoint_usage", "mid_only"),
        mid_embed_dim=a.get("mid_embed_dim", 64),
        classifier_head=a.get("classifier_head", "prototype"),
        proto_dim=a.get("proto_dim", 128),
        head_dropout=a.get("head_dropout", 0.1),
        prior_mode=a.get("prior_mode", "adapter"),
        prior_hidden_dim=a.get("prior_hidden_dim", 64),
        prior_dropout=a.get("prior_dropout", 0.1),
        detach_prior=a.get("detach_prior", False),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"  Model: backbone={a['model_type']} scale={a.get('model_scale','?')} "
        f"endpoint_usage={a.get('endpoint_usage')} "
        f"prior_mode={a.get('prior_mode')} "
        f"classifier_head={a.get('classifier_head')} "
        f"mid_layer={mid_layer}"
    )
    print(
        f"  Params: total={sum(p.numel() for p in model.parameters()):,}"
    )
    return model, a, mid_layer


# =====================================================
# FLOPs measurement
# =====================================================
def measure_flops(model, sample_input):
    """Run one forward pass under FlopCounterMode and return total FLOPs.

    sample_input: tensor [B, 3, N] on the same device as the model.
    Returns total FLOPs for that batch (int); divide by B to get per-fiber.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as e:
        print(f"  FlopCounterMode unavailable ({e}); skipping FLOPs.")
        return None

    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            _ = model(sample_input)
        total = int(fc.get_total_flops())
    return total


# =====================================================
# Inference
# =====================================================
@torch.no_grad()
def run_inference_with_timing(
    model, fiber_array_np, device, batch_size, warmup_batches,
):
    """Run model forward pass over all streamlines in batches.

    Returns:
        outputs: dict[str, np.ndarray]  # predicted class index per streamline
            "swm": [B] in {0, 1}
            f"{atlas}_start": [B] int
            f"{atlas}_end": [B] int
        timing: dict with 'total_seconds', 'fibers_per_second', 'num_batches'
    """
    n_total = fiber_array_np.shape[0]
    fiber_tensor = torch.from_numpy(fiber_array_np)  # [B, 3, N]

    # ----- Warmup -----
    warm_n = min(batch_size, n_total)
    warm_batch = fiber_tensor[:warm_n].to(device, non_blocking=True)
    for _ in range(max(0, warmup_batches)):
        _ = model(warm_batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # ----- Collect outputs and time the model-only forward passes -----
    swm_preds = np.empty(n_total, dtype=np.int64)
    atlas_preds = {
        f"{a}_{p}": np.empty(n_total, dtype=np.int64)
        for a in ATLAS_LIST for p in ("start", "end")
    }

    use_cuda_event = device.type == "cuda"
    if use_cuda_event:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

    n_batches = 0
    elapsed_total = 0.0

    for i0 in range(0, n_total, batch_size):
        i1 = min(i0 + batch_size, n_total)
        batch = fiber_tensor[i0:i1].to(device, non_blocking=True)

        if use_cuda_event:
            torch.cuda.synchronize(device)
            start_evt.record()
        else:
            t0 = time.perf_counter()

        out = model(batch)

        if use_cuda_event:
            end_evt.record()
            torch.cuda.synchronize(device)
            elapsed_total += start_evt.elapsed_time(end_evt) / 1000.0
        else:
            elapsed_total += time.perf_counter() - t0

        n_batches += 1

        swm_preds[i0:i1] = out["swm"].argmax(dim=1).cpu().numpy()
        for atlas in ATLAS_LIST:
            for pos in ("start", "end"):
                key = f"{atlas}_{pos}"
                atlas_preds[key][i0:i1] = out[key].argmax(dim=1).cpu().numpy()

    timing = {
        "total_seconds": elapsed_total,
        "fibers_per_second": (n_total / elapsed_total) if elapsed_total > 0 else float("nan"),
        "num_batches": n_batches,
        "num_fibers": n_total,
    }

    outputs = {"swm": swm_preds}
    outputs.update(atlas_preds)
    return outputs, timing


# =====================================================
# Output writing
# =====================================================
def write_outputs(
    subject_name,
    original_streamlines,
    valid_indices,
    predictions,
    header,
    out_root,
    skip_per_class_tck,
):
    """Write swm.tck and per-(atlas, class) tcks for one subject."""
    subj_dir = os.path.join(out_root, subject_name)
    os.makedirs(subj_dir, exist_ok=True)

    # Map valid-index -> original streamline list index for fast lookup.
    swm_pred = predictions["swm"]
    swm_streamlines = [
        original_streamlines[orig_i]
        for vi, orig_i in enumerate(valid_indices)
        if swm_pred[vi] == 1
    ]
    swm_path = os.path.join(subj_dir, "swm.tck")
    if swm_streamlines:
        save_tck(swm_streamlines, header, swm_path)
        print(f"  Saved {len(swm_streamlines):>7d} SWM fibers -> {swm_path}")
    else:
        print(f"  No SWM fibers predicted; skipped {swm_path}")

    if skip_per_class_tck:
        return

    # Per-atlas, per-class: only include fibers that were classified as SWM,
    # and group by (start_pred OR end_pred) per class index.
    swm_mask = swm_pred == 1
    for atlas in ATLAS_LIST:
        atlas_dir = os.path.join(subj_dir, atlas)
        s_pred = predictions[f"{atlas}_start"]
        e_pred = predictions[f"{atlas}_end"]

        # Restrict to SWM-classified fibers only (others have no meaningful
        # cortical endpoint label).
        n_classes = ATLAS_ROI_DIMS[atlas]
        for k in range(n_classes):
            in_class = swm_mask & ((s_pred == k) | (e_pred == k))
            if not in_class.any():
                continue
            sel_idx = np.where(in_class)[0]
            streams = [
                original_streamlines[valid_indices[vi]] for vi in sel_idx
            ]
            # k+1 in the filename to match the 1-based label convention used
            # during training (see _build_label_tensor in dataset_demo_kfold).
            out_path = os.path.join(atlas_dir, f"class-{k + 1}.tck")
            save_tck(streams, header, out_path)


# =====================================================
# Main per-subject pipeline
# =====================================================
def process_one_subject(
    subject_name, tck_path, model, device, args, flops_per_fiber_cache,
):
    """Load -> resample -> infer -> save for one subject. Returns a summary dict."""
    print(f"\n=== Subject: {subject_name} ===")
    print(f"  Loading {tck_path}")
    t_load_0 = time.perf_counter()
    streamlines, header = load_tck(tck_path)
    t_load = time.perf_counter() - t_load_0
    print(f"  Loaded {len(streamlines):,} streamlines in {t_load:.2f}s")

    print(f"  Resampling to {args.num_points} points per streamline...")
    t_pre_0 = time.perf_counter()
    fiber_array, valid_idx = preprocess_streamlines(streamlines, args.num_points)
    t_pre = time.perf_counter() - t_pre_0
    print(
        f"  Resampled {fiber_array.shape[0]:,} valid streamlines "
        f"(skipped {len(streamlines) - len(valid_idx)} degenerate) in "
        f"{t_pre:.2f}s"
    )

    # ----- FLOPs (measure once on a single-batch sample, cache per-fiber) -----
    flops_per_fiber = flops_per_fiber_cache.get("value")
    if args.measure_flops and flops_per_fiber is None:
        sample_n = min(args.batch_size, fiber_array.shape[0])
        sample = torch.from_numpy(fiber_array[:sample_n]).to(device)
        total = measure_flops(model, sample)
        if total is not None:
            flops_per_fiber = total / sample_n
            flops_per_fiber_cache["value"] = flops_per_fiber
            print(
                f"  FLOPs: {total:,} per batch of {sample_n} "
                f"-> {flops_per_fiber:,.0f} per fiber"
            )

    # ----- Inference -----
    print(f"  Running inference (batch_size={args.batch_size})...")
    preds, timing = run_inference_with_timing(
        model=model,
        fiber_array_np=fiber_array,
        device=device,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
    )
    print(
        f"  Inference: {timing['total_seconds']:.3f}s for "
        f"{timing['num_fibers']:,} fibers "
        f"({timing['fibers_per_second']:,.0f} fibers/s, "
        f"{timing['num_batches']} batches)"
    )

    # ----- Save outputs -----
    write_outputs(
        subject_name=subject_name,
        original_streamlines=streamlines,
        valid_indices=valid_idx,
        predictions=preds,
        header=header,
        out_root=args.output_dir,
        skip_per_class_tck=args.skip_per_class_tck,
    )

    n_swm_pred = int((preds["swm"] == 1).sum())
    summary = {
        "subject": subject_name,
        "input_tck": tck_path,
        "num_streamlines_loaded": len(streamlines),
        "num_streamlines_inferred": int(fiber_array.shape[0]),
        "num_streamlines_predicted_swm": n_swm_pred,
        "tck_load_seconds": t_load,
        "resample_seconds": t_pre,
        "inference_seconds": timing["total_seconds"],
        "fibers_per_second": timing["fibers_per_second"],
        "num_batches": timing["num_batches"],
        "flops_per_fiber": flops_per_fiber,
        "flops_total_inference": (
            flops_per_fiber * fiber_array.shape[0]
            if flops_per_fiber is not None else None
        ),
    }
    return summary


# =====================================================
# Main
# =====================================================
def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    inputs = collect_tck_inputs(args.input)
    print(f"Found {len(inputs)} subject(s) to process.")

    os.makedirs(args.output_dir, exist_ok=True)

    model, _, _ = build_model_from_ckpt(
        ckpt_path=args.ckpt,
        device=device,
        overlap_override=args.overlap_dir,
    )

    flops_per_fiber_cache = {}
    summaries = []
    total_wall_0 = time.perf_counter()
    for subj_name, tck_path in inputs:
        try:
            s = process_one_subject(
                subject_name=subj_name,
                tck_path=tck_path,
                model=model,
                device=device,
                args=args,
                flops_per_fiber_cache=flops_per_fiber_cache,
            )
            summaries.append(s)
        except Exception as e:
            print(f"  ERROR on {subj_name}: {e}")
            summaries.append({
                "subject": subj_name,
                "input_tck": tck_path,
                "error": repr(e),
            })

    total_wall = time.perf_counter() - total_wall_0

    # ----- Aggregate -----
    ok = [s for s in summaries if "error" not in s]
    if ok:
        inf_times = [s["inference_seconds"] for s in ok]
        fibers = [s["num_streamlines_inferred"] for s in ok]
        agg = {
            "num_subjects": len(ok),
            "total_wall_seconds": total_wall,
            "mean_inference_seconds_per_subject": float(np.mean(inf_times)),
            "std_inference_seconds_per_subject": float(np.std(inf_times)),
            "mean_fibers_per_subject": float(np.mean(fibers)),
            "mean_fibers_per_second": float(np.mean(
                [s["fibers_per_second"] for s in ok]
            )),
            "flops_per_fiber": flops_per_fiber_cache.get("value"),
        }
    else:
        agg = {"num_subjects": 0, "total_wall_seconds": total_wall}

    print("\n" + "=" * 60)
    print("Benchmark summary")
    print("=" * 60)
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_json = os.path.join(args.output_dir, "benchmark.json")
    with open(out_json, "w") as f:
        json.dump(
            {"args": vars(args), "aggregate": agg, "per_subject": summaries},
            f, indent=2, default=str,
        )
    print(f"\nSaved benchmark JSON -> {out_json}")


if __name__ == "__main__":
    main()
