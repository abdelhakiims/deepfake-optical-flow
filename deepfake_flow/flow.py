"""Optical flow: RAFT, and the transforms applied to the flow before the network.

RAFT
    Sintel weights, 32 refinement iterations, on 256 x 256 face crops. RAFT receives
    pixel values in 0..255, as it was trained (it rescales them to -1..1 itself).
    The flow is resized to 128 x 128 and stored as float16; values stay in pixels
    of the 256 x 256 crop.

Transforms (applied at training and test time)
    compensate_motion   removes the head's rigid motion from each flow map
    flip_flow           horizontal mirror: mirror the map and negate dx
    estimate_flow_scale one scale for the whole dataset, from training clips only
"""

import os
import sys

import numpy as np

from . import paths


def load_raft(device, weights=None, raft_code_dir=None):
    """RAFT with the Sintel weights, from a clone of the official repository.

    The RAFT code is not part of this project: clone it from paths.RAFT_URL into
    third_party/RAFT (or point --raft_dir / DEEPFAKE_RAFT_DIR to it) and download
    its pretrained models into its models/ folder.
    """
    import torch
    from argparse import Namespace
    code_dir = paths.raft_dir(raft_code_dir)
    weights = paths.raft_weights(weights, raft_code_dir)
    if not os.path.isdir(os.path.join(code_dir, "core")):
        raise SystemExit(f"RAFT code not found in {code_dir}. Clone {paths.RAFT_URL} there, "
                         "or pass --raft_dir.")
    if not os.path.isfile(weights):
        raise SystemExit(f"RAFT weights not found: {weights}. Run RAFT's download_models.sh "
                         "(or download raft-sintel.pth by hand), or pass --raft_weights.")
    # the official RAFT code imports its modules both as core.x and as x
    for folder in (os.path.join(code_dir, "core"), code_dir):
        if folder not in sys.path:
            sys.path.insert(0, folder)
    from core.raft import RAFT

    model = RAFT(Namespace(small=False, alternate_corr=False,
                           mixed_precision=(device.type == "cuda")))
    state = torch.load(weights, map_location=device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return model.to(device).eval()


def compute_flows(crops, raft, device, iters=32, batch=10, save_size=128):
    """Optical flow between consecutive crops of one clip.

    crops: uint8 RGB (N, H, W, 3), H and W multiples of 8.
    Returns float16 (N-1, 2, save_size, save_size); channel 0 = dx, 1 = dy,
    in pixels of the H x W crop.
    """
    import torch
    import torch.nn.functional as F
    frames = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()  # uint8, CPU
    n_pairs = frames.shape[0] - 1
    out = []
    with torch.no_grad():
        for s in range(0, n_pairs, batch):
            e = min(s + batch, n_pairs)
            # RAFT expects float images in the range 0..255
            img1 = frames[s:e].to(device).float()
            img2 = frames[s + 1:e + 1].to(device).float()
            _, flow = raft(img1, img2, iters=iters, test_mode=True)
            flow = flow.float()
            if flow.shape[-1] != save_size or flow.shape[-2] != save_size:
                flow = F.interpolate(flow, size=(save_size, save_size), mode="bilinear", align_corners=False)
            out.append(flow.cpu().numpy().astype(np.float16))
    return np.concatenate(out, axis=0)



# ----------------------------------------------------------------------------- #
# Transforms
# ----------------------------------------------------------------------------- #
_DESIGN_CACHE = {}


def _affine_design(h, w):
    """Design matrix [x, y, 1] on a normalised grid and its pseudo-inverse."""
    key = (h, w)
    if key not in _DESIGN_CACHE:
        ys, xs = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
        X = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)], axis=1).astype(np.float32)
        _DESIGN_CACHE[key] = (X, np.linalg.pinv(X).astype(np.float32))
    return _DESIGN_CACHE[key]


def compensate_motion(flow, mode="affine"):
    """Remove the global (head / camera) motion from each flow map.

    flow: float array (L, 2, H, W), channel 0 = dx, channel 1 = dy.
      'none'   : flow unchanged
      'mean'   : subtract the mean displacement of the map (pure translation)
      'affine' : fit dx and dy as affine functions of the pixel position by least
                 squares and subtract the fit. This removes translation, in-plane
                 rotation, scaling and shear of the crop, and keeps the residual,
                 non-rigid motion of the face (mouth, eyes, skin, blending border).
    """
    flow = np.asarray(flow, dtype=np.float32)
    if mode == "none":
        return flow
    if mode == "mean":
        return flow - flow.mean(axis=(2, 3), keepdims=True)
    if mode == "affine":
        L, C, H, W = flow.shape
        X, X_pinv = _affine_design(H, W)
        flat = flow.reshape(L, C, H * W)
        coef = flat @ X_pinv.T            # (L, 2, 3): one affine model per map and channel
        return (flat - coef @ X.T).reshape(L, C, H, W)
    raise ValueError(f"Unknown motion mode: {mode}")


def flip_flow(flow):
    """Horizontal flip of a flow clip (L, 2, H, W): mirror the map and negate dx."""
    out = flow[..., ::-1].copy()
    out[:, 0] *= -1.0
    return out


def estimate_flow_scale(samples, motion="affine", max_clips=200, seed=0):
    """Root mean square of the (compensated) flow over a sample of training clips.

    One number for the whole dataset: dividing by it keeps the difference
    between still and moving faces, which per-map standardisation erased.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))[:max_clips]
    total, count = 0.0, 0
    for i in order:
        flow_path, _, k = samples[i]
        flow = compensate_motion(np.load(flow_path, mmap_mode="r")[k], motion)
        total += float((flow.astype(np.float64) ** 2).sum())
        count += flow.size
    scale = float(np.sqrt(total / max(count, 1)))
    return scale if scale > 1e-6 else 1.0

