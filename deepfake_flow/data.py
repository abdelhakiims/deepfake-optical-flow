"""Data protocol: preprocessed videos, identity-safe splits, clips, PyTorch dataset.

On disk, one video = two files written by deepfake_flow.preprocess:
    <root>/<Real|Fake>/<video>.flow.npy   float16  (K, L, 2, S, S)   K clips of L flow maps
    <root>/<Real|Fake>/<video>.rgb.npy    uint8    (K, L+1, S, S, 3) the face crops of each clip

Rules
  * Videos are split, never clips: all clips of a video are on the same side.
  * No identity is on two sides: videos are grouped by identity (DFD actor,
    Celeb-DF celebrity) or by the official FaceForensics++ pairs, and every split
    is checked by assert_no_leak.
  * Clips are built after the split.
"""

import json
import os
import random
import re

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .flow import compensate_motion, flip_flow

LABEL_MAP = {"Real": 0, "Fake": 1}  # positive class = Fake, everywhere
FLOW_SUFFIX = ".flow.npy"
RGB_SUFFIX = ".rgb.npy"
CONFIG_NAME = "preprocess_config.json"


# ----------------------------------------------------------------------------- #
# Videos and names
# ----------------------------------------------------------------------------- #
def list_videos(flow_root):
    """One record per video: name, label, flow path, rgb path, clips, clip length, size."""
    videos = []
    for label_name, label in LABEL_MAP.items():
        label_dir = os.path.join(flow_root, label_name)
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Missing folder: {label_dir}")
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(FLOW_SUFFIX):
                continue
            name = fname[:-len(FLOW_SUFFIX)]
            flow_path = os.path.join(label_dir, fname)
            rgb_path = os.path.join(label_dir, name + RGB_SUFFIX)
            shape = np.load(flow_path, mmap_mode="r").shape
            if len(shape) != 5 or shape[2] != 2:
                raise ValueError(f"Unexpected flow shape {shape} in {flow_path}")
            videos.append({"name": name, "label": label, "flow": flow_path,
                           "rgb": rgb_path if os.path.isfile(rgb_path) else None,
                           "n_clips": int(shape[0]), "clip_len": int(shape[1]),
                           "size": int(shape[-1])})
    if not videos:
        raise FileNotFoundError(f"No '*{FLOW_SUFFIX}' file under {flow_root}. Run python -m deepfake_flow.preprocess first.")
    return videos


def load_preprocess_config(flow_root):
    """Settings used by the preprocessing for this folder (None if absent)."""
    path = os.path.join(flow_root, CONFIG_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_method_prefix(video_name):
    """'DF-000_003' -> '000_003'. FaceForensics++ fakes carry a method prefix
    (DF-, F2F-, FS-, NT-) because the four methods share file names."""
    head, sep, tail = video_name.partition("-")
    return tail if sep and "_" not in head else video_name


def video_ids(video_name):
    """FaceForensics++ source-video ids in a name: '000' -> ['000'],
    'DF-000_003' -> ['000', '003'] (target 000, face taken from 003)."""
    return strip_method_prefix(video_name).split("_")


def group_key(video_name, mode="prefix"):
    """Group used to keep related videos on the same side of the split.

    mode='prefix': text before the first underscore.
        Celeb-DF  'id0_0000'            -> 'id0'   (real video of identity 0)
                  'id0_id1_0000'        -> 'id0'   (fake built on that same real video)
                  '00170'               -> '00170' (YouTube-real, its own group)
        DFD       '01__walking_...'     -> '01'    (actor 01)
                  '01_02__walking_...'  -> '01'    (fake built on actor 01's video)
    mode='ffpp': FaceForensics++ target video, method prefix removed
                  '000' -> '000',  'DF-000_003' -> '000'. Use it with the
                  official split (split_videos_ffpp), which keeps the source video
                  003 on the same side as well.
    mode='video': each video is its own group (weaker).
    """
    if mode == "video":
        return video_name
    if mode == "ffpp":
        return video_ids(video_name)[0]
    return re.split(r"_", video_name, maxsplit=1)[0]


def read_name_list(path):
    """Video names (no folder, no extension) from a list file, one per line.

    Accepts the Celeb-DF test list ("<label> <folder>/<name>.mp4") as well as plain names."""
    names = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                token = line.split()[-1].replace("\\", "/")
                names.add(os.path.splitext(os.path.basename(token))[0])
    return names


# ----------------------------------------------------------------------------- #
# Splits
# ----------------------------------------------------------------------------- #
def _holdout(indices, labels, groups, fraction, seed):
    """Hold out about `fraction` of the videos, keeping groups intact."""
    indices = np.asarray(indices)
    y = np.asarray([labels[i] for i in indices])
    g = np.asarray([groups[i] for i in indices])
    n_splits = max(2, int(round(1.0 / fraction)))
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        keep, held = next(sgkf.split(indices, y, g))
    except ValueError:
        gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        keep, held = next(gss.split(indices, y, g))
    return indices[keep].tolist(), indices[held].tolist()


def split_videos(videos, seed=42, test_names=None, val_fraction=0.15,
                 test_fraction=0.15, group_by="prefix"):
    """Split video indices into train / val / test with no shared group.

    With test_names (official list), exactly those videos form the test set and
    any other video sharing a group with a test video is dropped from training.
    """
    labels = [v["label"] for v in videos]
    groups = [group_key(v["name"], group_by) for v in videos]
    all_idx = list(range(len(videos)))
    dropped = []

    if test_names:
        test_idx = [i for i in all_idx if videos[i]["name"] in test_names]
        if not test_idx:
            raise ValueError("No video of the dataset matches the test list. "
                             "Check that file names equal the video names.")
        test_groups = {groups[i] for i in test_idx}
        rest = [i for i in all_idx if videos[i]["name"] not in test_names]
        dropped = [i for i in rest if groups[i] in test_groups]
        rest = [i for i in rest if groups[i] not in test_groups]
        if not rest:
            raise ValueError("Every training video shares a group with a test video. "
                             "Use --group_by video with this test list.")
    else:
        rest, test_idx = _holdout(all_idx, labels, groups, test_fraction, seed)

    rel_val = val_fraction / (1.0 - (0.0 if test_names else test_fraction))
    train_idx, val_idx = _holdout(rest, labels, groups, rel_val, seed + 1)

    split = {"train": sorted(train_idx), "val": sorted(val_idx),
             "test": sorted(test_idx), "dropped": sorted(dropped)}
    assert_no_leak(videos, split, group_by)
    return split


def split_videos_kfold(videos, n_folds, fold, seed=42, val_fraction=0.15, group_by="prefix"):
    """Fold `fold` of a K-fold split by identity group.

    The groups are divided into n_folds folds (stratified by class, fixed by
    `seed`); fold `fold` is the test set, and a validation set is held out from
    the other folds. Running every fold once tests every identity exactly once.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be between 0 and {n_folds - 1}")
    labels = [v["label"] for v in videos]
    groups = [group_key(v["name"], group_by) for v in videos]
    idx = np.arange(len(videos))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rest, test = list(sgkf.split(idx, labels, groups))[fold]
    rel_val = val_fraction / (1.0 - 1.0 / n_folds)
    train_idx, val_idx = _holdout(rest.tolist(), labels, groups, rel_val, seed + 1)
    split = {"train": sorted(train_idx), "val": sorted(val_idx),
             "test": sorted(test.tolist()), "dropped": []}
    assert_no_leak(videos, split, group_by)
    return split


def read_ffpp_splits(split_dir):
    """Official FaceForensics++ split: {video id: 'train' | 'val' | 'test'}.

    split_dir holds train.json, val.json, test.json from the FaceForensics
    repository (dataset/splits): lists of pairs of ids, 360 / 70 / 70 pairs,
    i.e. 720 / 140 / 140 source videos. The fakes of pair [a, b] are a_b and b_a.
    """
    part_of = {}
    for part in ("train", "val", "test"):
        path = os.path.join(split_dir, part + ".json")
        with open(path, "r", encoding="utf-8") as f:
            for pair in json.load(f):
                for vid in pair:
                    if vid in part_of and part_of[vid] != part:
                        raise ValueError(f"Video id {vid} is in {part_of[vid]} and {part} of {split_dir}")
                    part_of[vid] = part
    return part_of


def split_videos_ffpp(videos, split_dir):
    """Official FaceForensics++ train / val / test split.

    A real video goes to the part of its id; a fake a_b goes to the part of
    the pair [a, b]. A video whose ids are missing from the lists, or fall in
    two parts, is dropped (that does not happen with the official files).
    """
    part_of = read_ffpp_splits(split_dir)
    split = {"train": [], "val": [], "test": [], "dropped": []}
    for i, v in enumerate(videos):
        parts = {part_of.get(vid) for vid in video_ids(v["name"])}
        if len(parts) == 1 and None not in parts:
            split[parts.pop()].append(i)
        else:
            split["dropped"].append(i)
    for part in ("train", "val", "test"):
        if not split[part]:
            raise ValueError(f"No video of the dataset falls in the official '{part}' part. "
                             "Check the names (000.flow.npy, DF-000_003.flow.npy, ...).")
    assert_no_leak(videos, split, "ffpp")
    seen = {}
    for part in ("train", "val", "test"):      # every source id stays on one side
        for i in split[part]:
            for vid in video_ids(videos[i]["name"]):
                if seen.setdefault(vid, part) != part:
                    raise AssertionError(f"Video id {vid} is in {seen[vid]} and {part}")
    return split


def assert_no_leak(videos, split, group_by="prefix"):
    """Fail loudly if a video or a group appears in two splits."""
    seen_videos, seen_groups = {}, {}
    for part in ("train", "val", "test"):
        for i in split[part]:
            key = (videos[i]["label"], videos[i]["name"])
            if key in seen_videos and seen_videos[key] != part:
                raise AssertionError(f"Video {key} is in {seen_videos[key]} and {part}")
            seen_videos[key] = part
            g = group_key(videos[i]["name"], group_by)
            if g in seen_groups and seen_groups[g] != part:
                raise AssertionError(f"Group '{g}' is in {seen_groups[g]} and {part}")
            seen_groups[g] = part


def describe_split(videos, split):
    """Videos and clips per class in each split, for the log and the paper."""
    out = {}
    for part in ("train", "val", "test", "dropped"):
        idx = split.get(part, [])
        real = [i for i in idx if videos[i]["label"] == 0]
        fake = [i for i in idx if videos[i]["label"] == 1]
        out[part] = {"videos": len(idx), "real": len(real), "fake": len(fake),
                     "clips_real": sum(videos[i]["n_clips"] for i in real),
                     "clips_fake": sum(videos[i]["n_clips"] for i in fake)}
    return out


# ----------------------------------------------------------------------------- #
# Clips, built after the split
# ----------------------------------------------------------------------------- #
def make_clips(videos, indices):
    """Samples (flow_path, rgb_path, clip_index), labels, and source video ids."""
    samples, labels, video_ids = [], [], []
    for i in indices:
        for k in range(videos[i]["n_clips"]):
            samples.append((videos[i]["flow"], videos[i]["rgb"], k))
            labels.append(videos[i]["label"])
            video_ids.append(i)
    return samples, labels, video_ids


def ffpp_val_test_fakes(split_dir):
    """Names of the FaceForensics++ fakes of the official validation and test parts
    (pair [a, b] gives fakes a_b and b_a), so that preprocessing always keeps them."""
    names = []
    for part in ("val", "test"):
        with open(os.path.join(split_dir, part + ".json"), "r", encoding="utf-8") as f:
            for a, b in json.load(f):
                names += [f"{a}_{b}", f"{b}_{a}"]
    return set(names)


def make_split(videos, seed=42, test_names=None, ffpp_splits=None, folds=0, fold=0,
               val_fraction=0.15, test_fraction=0.15, group_by="prefix"):
    """The split used for one run, chosen from the arguments:
    official FaceForensics++ split > K-fold by identity > official test list or own split."""
    if ffpp_splits:
        if folds or test_names:
            raise ValueError("The FaceForensics++ split cannot be combined with folds or a test list")
        return split_videos_ffpp(videos, ffpp_splits), "ffpp"
    if folds and not test_names:
        return split_videos_kfold(videos, folds, fold, seed=seed, val_fraction=val_fraction,
                                  group_by=group_by), group_by
    return split_videos(videos, seed=seed, test_names=test_names, val_fraction=val_fraction,
                        test_fraction=test_fraction, group_by=group_by), group_by


# ----------------------------------------------------------------------------- #
# PyTorch dataset
# ----------------------------------------------------------------------------- #
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ClipDataset(torch.utils.data.Dataset):
    """One sample = one clip: L residual flow maps and the L face crops they start from."""

    def __init__(self, samples, labels, mode, motion, flow_scale, train=False, flip=True):
        self.samples = samples
        self.labels = labels
        self.use_flow = mode in ("two_stream", "flow")
        self.use_rgb = mode in ("two_stream", "rgb")
        self.motion = motion
        self.flow_scale = float(flow_scale)
        self.train = train
        self.flip = flip

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flow_path, rgb_path, k = self.samples[idx]
        do_flip = self.train and self.flip and random.random() < 0.5
        flow_t = torch.zeros(0)
        rgb_t = torch.zeros(0, dtype=torch.uint8)

        if self.use_flow:
            flow = compensate_motion(np.load(flow_path, mmap_mode="r")[k], self.motion) / self.flow_scale
            if do_flip:
                flow = flip_flow(flow)
            flow_t = torch.from_numpy(np.ascontiguousarray(flow, dtype=np.float32))

        if self.use_rgb:
            if rgb_path is None:
                raise FileNotFoundError(f"No .rgb.npy next to {flow_path}")
            rgb = np.load(rgb_path, mmap_mode="r")[k]
            rgb = np.asarray(rgb[:rgb.shape[0] - 1])        # frame t pairs with flow t -> t+1
            if do_flip:
                rgb = rgb[:, :, ::-1]
            rgb_t = torch.from_numpy(np.array(rgb, copy=True)).permute(0, 3, 1, 2)

        return flow_t, rgb_t, torch.tensor(self.labels[idx], dtype=torch.long)


def make_loader(samples, labels, cfg, train, seed, batch_size, num_workers):
    dataset = ClipDataset(samples, labels, cfg["mode"], cfg["motion"], cfg["flow_scale"],
                          train=train, flip=cfg["flip"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                       num_workers=num_workers, pin_memory=False,
                                       worker_init_fn=seed_worker, generator=generator)

