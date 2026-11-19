"""Check that this machine can run the project: Python, packages, GPU, RAFT, face detector.

It needs no dataset (it uses random images). For a test of the code itself, run
    python -m pytest tests

Example
    python -m deepfake_flow.check_env
"""

import argparse
import importlib
import os
import sys
import traceback

from . import paths

results = []


def step(name, fn, *args):
    try:
        detail = fn(*args)
        results.append((name, True))
        print(f"[ OK ] {name}  {detail or ''}")
    except BaseException as exc:          # SystemExit from a missing file is a failed check too
        results.append((name, False))
        print(f"[FAIL] {name}: {exc}")
        if not isinstance(exc, SystemExit):
            traceback.print_exc(limit=3)


def python_version():
    v = sys.version_info
    if not ((3, 10) <= (v.major, v.minor) <= (3, 14)):
        raise RuntimeError(f"Python {v.major}.{v.minor} found; this project is tested with 3.10 to 3.14")
    return f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"


def packages():
    found = []
    for module, label in [("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"),
                          ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("cv2", "opencv-python")]:
        mod = importlib.import_module(module)
        found.append(f"{label} {getattr(mod, '__version__', '?')}")
    return ", ".join(found)


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch does not see a GPU (CPU-only build or driver problem); "
                           "preprocessing and training would be very slow")
    props = torch.cuda.get_device_properties(0)
    return f"{props.name}, {props.total_memory / 1e9:.1f} GB, CUDA {torch.version.cuda}"


def raft_flow(raft_dir, raft_weights):
    import numpy as np
    import torch
    from .flow import compute_flows, load_raft
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raft = load_raft(device, raft_weights, raft_dir)
    crops = np.random.default_rng(0).integers(0, 256, size=(3, 256, 256, 3), dtype=np.uint8)
    flows = compute_flows(crops, raft, device, iters=12, batch=2, save_size=128)
    if flows.shape != (2, 2, 128, 128) or str(flows.dtype) != "float16":
        raise RuntimeError(f"unexpected flow output {flows.shape} {flows.dtype}")
    if not np.isfinite(flows.astype("float32")).all():
        raise RuntimeError("flow contains NaN or infinite values")
    return f"RAFT ran on {device}, output {flows.shape}"


def face_detector(face_model):
    import numpy as np
    from .faces import FaceDetector
    path = paths.face_model(face_model)
    if os.path.isfile(path) and os.path.getsize(path) < 100_000:
        raise RuntimeError(f"{path} is only {os.path.getsize(path)} bytes: not the real model "
                           "(a web page saved instead?). Download it again.")
    FaceDetector(path).faces(np.zeros((360, 640, 3), dtype=np.uint8))
    return f"YuNet loads and runs ({path})"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the environment of the project")
    ap.add_argument("--face_model", default=None)
    ap.add_argument("--raft_dir", default=None)
    ap.add_argument("--raft_weights", default=None)
    args = ap.parse_args(argv)
    step("Python version", python_version)
    step("Packages", packages)
    step("GPU", gpu)
    step("RAFT optical flow (random images)", raft_flow, args.raft_dir, args.raft_weights)
    step("Face detector", face_detector, args.face_model)
    failed = [name for name, ok in results if not ok]
    print()
    if failed:
        print("NOT READY. Failed checks: " + "; ".join(failed))
        sys.exit(1)
    print("READY. Every check passed.")


if __name__ == "__main__":
    main()
