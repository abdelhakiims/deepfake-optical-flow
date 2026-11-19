"""Tests of the code on small random data (CPU, no dataset, no RAFT, no internet).

    python -m pytest tests
"""

import json
import os

import numpy as np
import pytest
import torch

from deepfake_flow import data, evaluate, metrics, summarize, train
from deepfake_flow.flow import compensate_motion, estimate_flow_scale, flip_flow
from deepfake_flow.model import MODES, TEMPORAL, DeepfakeNet

FFPP_SPLITS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "ffpp")


# ----------------------------------------------------------------------------- #
# Flow transforms
# ----------------------------------------------------------------------------- #
def test_affine_motion_is_removed_exactly():
    h = w = 16
    ys, xs = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    flow = np.stack([0.3 * xs - 0.2 * ys + 1.5, 0.1 * xs + 0.4 * ys - 2.0])[None].astype(np.float32)
    assert np.abs(compensate_motion(flow, "affine")).max() < 1e-4
    assert np.abs(compensate_motion(flow, "mean").mean(axis=(2, 3))).max() < 1e-5
    assert np.array_equal(compensate_motion(flow, "none"), flow)


def test_residual_motion_is_kept():
    rng = np.random.default_rng(0)
    local = compensate_motion(rng.normal(size=(3, 2, 16, 16)).astype(np.float32), "affine")
    # a flow with no affine part is left unchanged
    assert np.allclose(compensate_motion(local, "affine"), local, atol=1e-4)


def test_flip_mirrors_and_negates_dx():
    flow = np.random.default_rng(1).normal(size=(2, 2, 4, 5)).astype(np.float32)
    out = flip_flow(flow)
    assert np.allclose(out[:, 0], -flow[:, 0, :, ::-1])
    assert np.allclose(out[:, 1], flow[:, 1, :, ::-1])
    assert np.allclose(flip_flow(out), flow)


# ----------------------------------------------------------------------------- #
# Names, splits
# ----------------------------------------------------------------------------- #
def test_group_keys():
    assert data.group_key("id0_id1_0000") == "id0"
    assert data.group_key("01_02__walking") == "01"
    assert data.group_key("00170") == "00170"
    assert data.group_key("DF-000_003", "ffpp") == "000"
    assert data.video_ids("NT-012_026") == ["012", "026"]
    assert data.strip_method_prefix("id0_0000") == "id0_0000"


def fake_videos(n_ids=14):
    vids = []
    for i in range(n_ids):
        for k in range(2):
            vids.append({"name": f"id{i}_{k:04d}", "label": 0, "n_clips": 3})
            vids.append({"name": f"id{i}_id{(i + 1) % n_ids}_{k:04d}", "label": 1, "n_clips": 3})
    return vids


def test_kfold_tests_every_video_once_without_leak():
    vids = fake_videos()
    seen = []
    for fold in range(7):
        split = data.split_videos_kfold(vids, 7, fold, seed=42)
        data.assert_no_leak(vids, split)
        seen += split["test"]
    assert sorted(seen) == list(range(len(vids)))


def test_leak_is_detected():
    vids = fake_videos()
    split = {"train": [0], "val": [], "test": [2], "dropped": []}   # same identity id0
    with pytest.raises(AssertionError):
        data.assert_no_leak(vids, split)


def test_official_ffpp_split():
    pairs = {p: json.load(open(os.path.join(FFPP_SPLITS, p + ".json"))) for p in ("train", "val", "test")}
    vids = []
    for part, plist in pairs.items():
        for a, b in plist[:3]:
            vids += [{"name": a, "label": 0, "n_clips": 1}, {"name": f"DF-{a}_{b}", "label": 1, "n_clips": 1},
                     {"name": f"FS-{b}_{a}", "label": 1, "n_clips": 1}]
    split = data.split_videos_ffpp(vids, FFPP_SPLITS)
    assert [len(split[p]) for p in ("train", "val", "test")] == [9, 9, 9] and not split["dropped"]
    assert len(data.ffpp_val_test_fakes(FFPP_SPLITS)) == 280


# ----------------------------------------------------------------------------- #
# Metrics
# ----------------------------------------------------------------------------- #
def test_metrics_and_threshold():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.55, 0.6, 0.7, 0.8, 0.9]
    m = metrics.compute_metrics(y, p)
    assert m["auc"] == 1.0 and m["confusion_matrix"]["FP_real_as_fake"] == 2
    thr = metrics.choose_threshold(y, p)
    assert metrics.compute_metrics(y, p, thr)["accuracy"] == 1.0
    v_true, v_prob, _ = metrics.aggregate_by_video([1, 1, 0], [0.2, 0.6, 0.1], [7, 7, 3])
    assert v_true == [0, 1] and np.allclose(v_prob, [0.1, 0.4])


# ----------------------------------------------------------------------------- #
# Network and end-to-end run
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("temporal", TEMPORAL)
def test_network_variants(mode, temporal):
    net = DeepfakeNet(mode=mode, temporal=temporal, pretrained=False).eval()
    flow = torch.randn(2, 5, 2, 32, 32)
    rgb = torch.randint(0, 256, (2, 5, 3, 32, 32), dtype=torch.uint8)
    with torch.no_grad():
        assert tuple(net(flow, rgb).shape) == (2, 2)


def write_dataset(root, n_ids=14):
    rng = np.random.default_rng(0)
    for label in ("Real", "Fake"):
        os.makedirs(os.path.join(root, label))
    for v in fake_videos(n_ids):
        folder = "Fake" if v["label"] else "Real"
        np.save(os.path.join(root, folder, v["name"] + ".flow.npy"),
                rng.normal(size=(2, 4, 2, 32, 32)).astype(np.float16))
        np.save(os.path.join(root, folder, v["name"] + ".rgb.npy"),
                rng.integers(0, 256, size=(2, 5, 32, 32, 3), dtype=np.uint8))


def test_train_evaluate_summarize(tmp_path):
    flows = str(tmp_path / "flows")
    write_dataset(flows)
    common = ["--flow_root", flows, "--num_workers", "0", "--batch_size", "2"]
    train.main(common + ["--out_dir", str(tmp_path / "runs" / "a"), "--folds", "7", "--fold", "0",
                         "--epochs", "1", "--no_pretrained"])
    ckpt = torch.load(tmp_path / "runs" / "a" / "best.pth", weights_only=False)
    assert 0.0 <= ckpt["threshold"] <= 1.0 and ckpt["config"]["flow_scale"] > 0
    evaluate.main(common + ["--out_dir", str(tmp_path / "runs" / "b"),
                            "--checkpoint", str(tmp_path / "runs" / "a" / "best.pth")])
    for run in ("a", "b"):
        report = json.load(open(tmp_path / "runs" / run / "results.json"))
        assert report["video_level"]["n"] > 0
    summarize.main(["--runs_dir", str(tmp_path / "runs")])
    assert (tmp_path / "runs" / "summary.csv").is_file()


def test_flow_scale_is_positive(tmp_path):
    flows = str(tmp_path / "flows")
    write_dataset(flows, n_ids=2)
    videos = data.list_videos(flows)
    samples, _, _ = data.make_clips(videos, range(len(videos)))
    assert estimate_flow_scale(samples, "affine") > 0
