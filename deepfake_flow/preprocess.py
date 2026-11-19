"""Video -> clips of consecutive face crops -> RAFT optical flow.

For each video it writes two files in <out_root>/<label>/:
    <video>.flow.npy   float16  (K, L, 2, S, S)    K clips, L flow maps each (dx, dy)
    <video>.rgb.npy    uint8    (K, L+1, S, S, 3)  the face crops the flows were computed from

Steps (settings in DEFAULT_CONFIG)
  1. Face scan and tracking of the main subject (see faces.py).
  2. Up to K non-overlapping clips of L+1 CONSECUTIVE frames, at the video's own
     frame rate, where the main subject is usable all along.
  3. Face crop following the subject: box interpolated between scan frames,
     smoothed over 5 frames, square, enlarged by `margin`, black padding outside
     the frame, resized to 256 x 256.
  4. RAFT optical flow between each crop and the next (see flow.py).
  5. Flow and crops stored at 128 x 128.

Every video of one out_root must use the same settings (preprocess_config.json);
the script refuses otherwise. A run can be stopped and restarted: finished videos
are skipped. Each video gets one line in a CSV log (usable or not, and why) and a
preview image (top row: first frame of each clip, bottom row: last frame).

Example
    python -m deepfake_flow.preprocess --videos_dir <dataset>/Celeb-real --label Real --out_root <flows>/CelebDF
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np

from . import paths
from .data import CONFIG_NAME, ffpp_val_test_fakes, read_name_list
from .faces import (FaceDetector, choose_windows, interpolate_boxes, link_tracks, main_track,
                    smooth_boxes, valid_scan_positions)
from .flow import compute_flows, load_raft

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")

DEFAULT_CONFIG = {
    "fps": 0,              # 0 = consecutive frames at the video's own frame rate
    "clip_len": 20,        # L: flow maps per clip (L+1 frames)
    "clips": 6,            # K: clips per video (at most)
    "crop_size": 256,      # size of the face crop given to RAFT
    "save_size": 128,      # size at which flow and crops are stored
    "margin": 1.3,         # enlargement of the face box
    "iters": 32,           # RAFT refinement iterations
    "face_crop": True,     # False = whole frame squashed (comparison only)
    "crop": "track",       # crop follows the tracked main subject
    "scan_fps": 8,         # face detections per second in the scan
    "min_face": 80,        # minimum face width in pixels of the original video
    "min_frontal": 0.25,   # minimum eye distance / face width for the main subject
    "rival_frontal": 0.30, # another face this frontal ...
    "rival_area": 0.5,     # ... and at least this fraction of the subject's area makes a frame ambiguous
    "smooth": 5,           # frames of the moving average applied to the crop box
}


def frame_step(native_fps, target_fps):
    """Keep one frame every `step` frames. target_fps = 0 means every frame."""
    if target_fps <= 0 or native_fps <= 0:
        return 1
    return max(1, int(round(native_fps / target_fps)))


def clip_starts(n_frames, clip_len, step, n_clips):
    """First frame of each clip at regular positions (whole-frame mode only)."""
    span = clip_len * step + 1
    if n_frames < span:
        return []
    k = min(n_clips, n_frames // span)
    if k == 1:
        return [(n_frames - span) // 2]
    return [int(round(s)) for s in np.linspace(0, n_frames - span, k)]


def select_videos(all_names, max_videos=0, must_include=None, seed=42):
    """Choose which videos to process, reproducibly.

    Videos in must_include (e.g. the official test list) are always kept. If
    max_videos > 0, the rest is a random sample so that the total is max_videos.
    """
    all_names = sorted(all_names)
    must = [n for n in all_names if must_include and n in must_include]
    rest = [n for n in all_names if not (must_include and n in must_include)]
    if max_videos and max_videos > 0:
        k = max(0, max_videos - len(must))
        rest = sorted(random.Random(seed).sample(rest, min(k, len(rest))))
    return sorted(must + rest)


def check_or_write_config(out_root, config):
    """All videos of one out_root must share the same preprocessing settings."""
    path = os.path.join(out_root, CONFIG_NAME)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        keys = set(existing) | set(config)
        diff = {k: (existing.get(k), config.get(k)) for k in keys if existing.get(k) != config.get(k)}
        if diff:
            raise SystemExit(
                f"{out_root} was already preprocessed with other settings.\n"
                f"Differences (existing, requested): {diff}\n"
                "Use another --out_root for this configuration.")
    else:
        os.makedirs(out_root, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


def video_info(video_path):
    """Number of frames and native frame rate of a video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:                                  # some containers do not store the count
        while cap.grab():
            n += 1
    cap.release()
    return n, fps


def scan_faces(video_path, detector, scan_step):
    """Detect faces on every scan_step-th frame. Returns (scan, frames_read, frame_size)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    scan, index, size = [], 0, None
    try:
        while cap.grab():
            if index % scan_step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    size = (frame.shape[1], frame.shape[0])
                    scan.append((index, detector.faces(frame)))
            index += 1
    finally:
        cap.release()
    return scan, index, size


def crop_square(frame, cx, cy, side, out_size):
    """Square crop centred on (cx, cy), padded with black outside the frame."""
    import cv2
    h, w = frame.shape[:2]
    side = max(int(round(side)), 2)
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side
    pl, pt, pr, pb = max(0, -x0), max(0, -y0), max(0, x1 - w), max(0, y1 - h)
    patch = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if pl or pt or pr or pb:
        patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)


def read_frames(video_path, wanted):
    """Frames at the requested indices, as {index: BGR frame}; reads the video once."""
    import cv2
    wanted = set(wanted)
    last = max(wanted)
    out, index = {}, 0
    cap = cv2.VideoCapture(video_path)
    try:
        while index <= last and cap.grab():
            if index in wanted:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    out[index] = frame
            index += 1
    finally:
        cap.release()
    return out


def extract_clips(video_path, detector, config):
    """Return (clips, info); each clip is uint8 RGB (L+1, crop_size, crop_size, 3)."""
    import cv2
    n_frames, native_fps = video_info(video_path)
    step = frame_step(native_fps, config["fps"])
    L = config["clip_len"]
    span = L * step + 1
    size = config["crop_size"]
    info = {"frames_total": n_frames, "native_fps": round(float(native_fps), 3), "step": step,
            "clips": 0, "max_faces": 0, "valid_share": None, "face_px": None, "rejected": None}
    if n_frames < span:
        raise RuntimeError(f"video too short: {n_frames} frames, {span} needed")

    if detector is None:                         # whole frame, regular positions
        starts = clip_starts(n_frames, L, step, config["clips"])
        windows = [[s + j * step for j in range(L + 1)] for s in starts]
        frames = read_frames(video_path, [f for w in windows for f in w])
        clips = []
        for w in windows:
            if all(f in frames for f in w):
                clips.append(np.stack([cv2.cvtColor(cv2.resize(frames[f], (size, size), interpolation=cv2.INTER_AREA),
                                                    cv2.COLOR_BGR2RGB) for f in w]))
        info["clips"] = len(clips)
        if not clips:
            raise RuntimeError("no frame could be read")
        return clips, info

    scan_step = frame_step(native_fps, config["scan_fps"])
    scan, n_read, frame_size = scan_faces(video_path, detector, scan_step)
    n_frames = min(n_frames, n_read) if n_read else n_frames
    info["frames_total"] = n_frames
    info["max_faces"] = max((len(f) for _, f in scan), default=0)
    tracks = link_tracks(scan)
    main = main_track(tracks, config["min_frontal"])
    if main is None:
        raise RuntimeError("no frontal face found")
    valid, reasons = valid_scan_positions(scan, tracks, main, config, frame_size)
    info["valid_share"] = round(float(np.mean(valid)), 3) if valid else 0.0
    info["rejected"] = reasons
    scan_frames = [f for f, _ in scan]
    starts = choose_windows(scan_frames, valid, n_frames, span, config["clips"])
    if not starts:
        raise RuntimeError(f"no usable clip: main face usable on {info['valid_share']:.0%} of scanned frames; "
                           f"rejected frames by reason {reasons}")

    windows = [[s + j * step for j in range(L + 1)] for s in starts]
    frames = read_frames(video_path, [f for w in windows for f in w])
    clips, sizes = [], []
    for w in windows:
        if not all(f in frames for f in w):
            continue
        boxes = smooth_boxes(interpolate_boxes(scan_frames, tracks[main], w), config["smooth"])
        crops = [cv2.cvtColor(crop_square(frames[f], b[0], b[1], b[2] * config["margin"], size), cv2.COLOR_BGR2RGB)
                 for f, b in zip(w, boxes)]
        clips.append(np.stack(crops))
        sizes.append(float(boxes[:, 2].mean()))
    if not clips:
        raise RuntimeError("frames of the chosen clips could not be read")
    info["clips"] = len(clips)
    info["face_px"] = int(round(np.mean(sizes)))
    return clips, info


def process_video(video_path, raft, detector, device, config, batch=10):
    """Full preprocessing of one video.

    Returns (flows, rgb, info):
        flows  float16 (K, L, 2, S, S)
        rgb    uint8   (K, L+1, S, S, 3)
    """
    import cv2
    clips, info = extract_clips(video_path, detector, config)
    S = config["save_size"]
    flows, rgb = [], []
    for crops in clips:
        flows.append(compute_flows(crops, raft, device, config["iters"], batch, S))
        if crops.shape[1] != S:
            crops = np.stack([cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA) for c in crops])
        rgb.append(crops)
    return np.stack(flows), np.stack(rgb), info


def preview_image(rgb):
    """First (top row) and last (bottom row) crop of every clip."""
    return np.vstack([np.hstack(list(rgb[:, 0])), np.hstack(list(rgb[:, -1]))])


# ----------------------------------------------------------------------------- #
# Command line
# ----------------------------------------------------------------------------- #
def build_parser():
    d = DEFAULT_CONFIG
    ap = argparse.ArgumentParser(description="Face clips + RAFT optical flow, two .npy files per video")
    ap.add_argument("--videos_dir", required=True, help="folder with the video files of one class")
    ap.add_argument("--label", required=True, choices=["Real", "Fake"])
    ap.add_argument("--out_root", required=True, help="output folder of the dataset")
    ap.add_argument("--fps", type=float, default=d["fps"],
                    help="0 = consecutive frames (default); e.g. 5 = one frame every 0.2 s")
    ap.add_argument("--clip_len", type=int, default=d["clip_len"], help="flow maps per clip")
    ap.add_argument("--clips", type=int, default=d["clips"], help="clips per video (at most)")
    ap.add_argument("--crop_size", type=int, default=d["crop_size"], help="face crop size given to RAFT")
    ap.add_argument("--save_size", type=int, default=d["save_size"], help="size at which flow and crops are stored")
    ap.add_argument("--margin", type=float, default=d["margin"], help="enlargement of the face box")
    ap.add_argument("--iters", type=int, default=d["iters"], help="RAFT refinement iterations")
    ap.add_argument("--min_face", type=int, default=d["min_face"], help="minimum face width in pixels")
    ap.add_argument("--no_face_crop", action="store_true", help="whole frame squashed (comparison only)")
    ap.add_argument("--batch", type=int, default=10, help="frame pairs per RAFT batch")
    ap.add_argument("--max_videos", type=int, default=0, help="random subset size (0 = all)")
    ap.add_argument("--must_include", default=None,
                    help="list file of videos always kept (e.g. the official Celeb-DF test list)")
    ap.add_argument("--must_include_ffpp", default=None,
                    help="folder of the official FaceForensics++ split: always keep its validation "
                         "and test fakes")
    ap.add_argument("--seed", type=int, default=42, help="seed of the random video subset")
    ap.add_argument("--name_prefix", default="",
                    help="text put before every output name, e.g. DF- for FaceForensics++ Deepfakes "
                         "(the four methods share file names); must not contain '_'")
    ap.add_argument("--face_model", default=None, help="YuNet .onnx file (see paths.py)")
    ap.add_argument("--raft_dir", default=None, help="clone of the RAFT repository (see paths.py)")
    ap.add_argument("--raft_weights", default=None, help="RAFT weights file (see paths.py)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if "_" in args.name_prefix:
        raise SystemExit("--name_prefix must not contain '_' (it would change the identity groups)")

    import cv2
    import torch

    config = dict(DEFAULT_CONFIG)
    config.update({"fps": args.fps, "clip_len": args.clip_len, "clips": args.clips,
                   "crop_size": args.crop_size, "save_size": args.save_size, "margin": args.margin,
                   "iters": args.iters, "min_face": args.min_face, "face_crop": not args.no_face_crop})
    if args.no_face_crop:
        config["crop"] = "whole_frame"
    check_or_write_config(args.out_root, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Settings: {config}")

    files = {os.path.splitext(f)[0]: os.path.join(args.videos_dir, f)
             for f in os.listdir(args.videos_dir) if f.lower().endswith(VIDEO_EXT)}
    must = set()
    if args.must_include:
        must |= read_name_list(args.must_include)
    if args.must_include_ffpp:
        must |= ffpp_val_test_fakes(args.must_include_ffpp)
    names = select_videos(files.keys(), args.max_videos, must or None, args.seed)
    print(f"{len(files)} videos found, {len(names)} selected")

    out_dir = os.path.join(args.out_root, args.label)
    preview_dir = os.path.join(args.out_root, "_previews", args.label)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    tag = f"{args.label}_{args.name_prefix}{os.path.basename(os.path.normpath(args.videos_dir))}"
    with open(os.path.join(args.out_root, f"selection_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"videos_dir": os.path.basename(os.path.normpath(args.videos_dir)),
                   "name_prefix": args.name_prefix, "max_videos": args.max_videos,
                   "must_include": sorted(must), "seed": args.seed, "selected": names}, f, indent=2)
    log_path = os.path.join(args.out_root, f"log_{tag}.csv")
    new_log = not os.path.exists(log_path)

    detector = FaceDetector(args.face_model) if config["face_crop"] else None
    raft = load_raft(device, args.raft_weights, args.raft_dir)

    done = failed = skipped = 0
    t_start = time.time()
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(["video", "status", "frames_total", "native_fps", "step", "clips",
                          "max_faces", "valid_share", "face_px", "seconds", "message"])
        for n, name in enumerate(names, 1):
            out_name = args.name_prefix + name
            flow_path = os.path.join(out_dir, out_name + ".flow.npy")
            rgb_path = os.path.join(out_dir, out_name + ".rgb.npy")
            if os.path.exists(flow_path) and os.path.exists(rgb_path):
                skipped += 1
                continue
            t0 = time.time()
            try:
                flows, rgb, info = process_video(files[name], raft, detector, device, config, args.batch)
                np.save(rgb_path, rgb)
                tmp = flow_path + ".tmp.npy"      # the flow file is written last: its
                np.save(tmp, flows)               # presence marks the video as complete
                os.replace(tmp, flow_path)
                cv2.imwrite(os.path.join(preview_dir, out_name + ".jpg"),
                            cv2.cvtColor(preview_image(rgb), cv2.COLOR_RGB2BGR))
                log.writerow([out_name, "ok", info["frames_total"], info["native_fps"], info["step"],
                              info["clips"], info["max_faces"], info["valid_share"], info["face_px"],
                              round(time.time() - t0, 1), ""])
                done += 1
            except Exception as exc:              # one bad video must not stop the run
                log.writerow([out_name, "failed", "", "", "", "", "", "", "",
                              round(time.time() - t0, 1), str(exc)])
                failed += 1
                print(f"  FAILED {out_name}: {exc}")
            log_file.flush()
            if n % 20 == 0 or n == len(names):
                print(f"[{n}/{len(names)}] ok {done}, failed {failed}, already done {skipped}, "
                      f"{(time.time() - t_start) / 60:.1f} min")

    print(f"Finished. Output in {out_dir}. Log: {log_path}.")
    print(f"Check the face crops in {preview_dir} (top row: first frame of each clip, "
          "bottom row: last frame).")


if __name__ == "__main__":
    main()
