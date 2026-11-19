"""Training of the detector, then test on the held-out videos.

One run = one split (official FaceForensics++ split, one fold of a K-fold split by
identity, an official test list, or an own split by identity), one configuration
(mode, motion removal, temporal head) and one seed. Outputs in --out_dir:

    config.json            arguments and model configuration
    split.json             the videos of each part
    history.json           one line per epoch
    best.pth               model of the best validation epoch, with its configuration,
                           flow scale and decision threshold
    results.json           test metrics (clip and video level, threshold 0.5 and the
                           validation threshold, bootstrap interval of the video AUC)
    test_predictions.csv   one line per test video

Examples
    python -m deepfake_flow.train --flow_root <flows>/DFD --folds 7 --fold 0 --out_dir runs/dfd_f0
    python -m deepfake_flow.train --flow_root <flows>/FFPP --ffpp_splits splits/ffpp --out_dir runs/ffpp
"""

import argparse
import csv
import gc
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import (describe_split, list_videos, load_preprocess_config, make_clips, make_loader,
                   make_split, read_name_list)
from .flow import estimate_flow_scale
from .metrics import aggregate_by_video, bootstrap_auc_ci, choose_threshold, compute_metrics
from .model import MODES, TEMPORAL, DeepfakeNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def autocast(device):
    return torch.autocast(device_type=device.type, enabled=(device.type == "cuda"))


def make_scaler(device):
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_model(cfg, pretrained):
    return DeepfakeNet(mode=cfg["mode"], temporal=cfg["temporal"], pretrained=pretrained)


@torch.no_grad()
def predict(model, loader, device):
    """True labels, probability of Fake, and mean loss, in loader order."""
    model.eval()
    y_true, p_fake, loss_sum, n = [], [], 0.0, 0
    for flow, rgb, y in loader:
        flow, rgb, y = flow.to(device), rgb.to(device), y.to(device)
        with autocast(device):
            logits = model(flow, rgb)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
        p_fake.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
    cleanup_memory()
    return y_true, p_fake, loss_sum / max(n, 1)


def full_report(y_true, p_fake, video_ids):
    v_true, v_prob, v_ids = aggregate_by_video(y_true, p_fake, video_ids)
    report = {"clip_level": compute_metrics(y_true, p_fake),
              "video_level": compute_metrics(v_true, v_prob)}
    report["video_level"]["auc_ci95"] = bootstrap_auc_ci(v_true, v_prob)
    return report, (v_true, v_prob, v_ids)


def print_report(title, report):
    print(f"\n===== {title} =====")
    for level in ("clip_level", "video_level"):
        m = report[level]
        auc = "n/a" if m["auc"] is None else f"{m['auc']:.4f}"
        far = "n/a" if m["false_alarm_rate"] is None else f"{m['false_alarm_rate']:.4f}"
        print(f"{level:12s} n={m['n']:5d} (real {m['n_real']}, fake {m['n_fake']})  AUC={auc}  "
              f"acc={m['accuracy']:.4f}  prec={m['precision_fake']:.4f}  rec={m['recall_fake']:.4f}  "
              f"F1={m['f1_fake']:.4f}  false-alarm={far}")
        print(f"{'':12s} {m['confusion_matrix']}")
    ci = report["video_level"].get("auc_ci95")
    if ci:
        print(f"{'':12s} video-level AUC 95% interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    m = report.get("video_level_at_val_threshold")
    if m:
        far = "n/a" if m["false_alarm_rate"] is None else f"{m['false_alarm_rate']:.4f}"
        bal = "n/a" if m["balanced_accuracy"] is None else f"{m['balanced_accuracy']:.4f}"
        print(f"video level at the validation threshold {report['threshold']:.3f}: acc={m['accuracy']:.4f}  "
              f"balanced acc={bal}  false-alarm={far}")
        print(f"{'':12s} {m['confusion_matrix']}")


def train_model(model, train_loader, val_loader, val_video_ids, class_weights, device, args, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(device)
    acc_steps = args.accumulation_steps
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    history, best_score, bad_epochs = [], -1.0, 0
    best_path = os.path.join(args.out_dir, "best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, correct, n = 0.0, 0, 0
        t0 = time.time()
        n_batches = len(train_loader)

        for batch_idx, (flow, rgb, y) in enumerate(train_loader):
            flow, rgb, y = flow.to(device), rgb.to(device), y.to(device)
            with autocast(device):
                logits = model(flow, rgb)
                loss = F.cross_entropy(logits.float(), y, weight=weights)
            scaler.scale(loss / acc_steps).backward()

            if (batch_idx + 1) % acc_steps == 0 or (batch_idx + 1) == n_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            n += y.numel()
            if batch_idx % 200 == 0:
                print(f"  epoch {epoch} batch {batch_idx}/{n_batches} loss {loss.item():.4f}")

        cleanup_memory()
        y_val, p_val, val_loss = predict(model, val_loader, device)
        val_report, _ = full_report(y_val, p_val, val_video_ids)
        score = val_report["video_level"]["auc"]
        if score is None:                       # one class only in validation
            score = val_report["video_level"]["accuracy"]

        row = {"epoch": epoch,
               "train_loss": loss_sum / n, "train_accuracy": correct / n,
               "val_loss": val_loss,
               "val_clip_auc": val_report["clip_level"]["auc"],
               "val_clip_accuracy": val_report["clip_level"]["accuracy"],
               "val_video_auc": val_report["video_level"]["auc"],
               "val_video_accuracy": val_report["video_level"]["accuracy"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}  train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} | "
              f"val loss {val_loss:.4f} video AUC {row['val_video_auc']} video acc {row['val_video_accuracy']:.4f}")
        with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if score > best_score:
            best_score, bad_epochs = score, 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                        "val_video_auc": best_score}, best_path)
            print(f"  saved best model (validation video-level score {best_score:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stopping at epoch {epoch}")
                break
    return best_path


def save_predictions(path, videos, v_true, v_prob, v_ids, threshold=0.5):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "true_label", "p_fake", "predicted_at_0.5", "predicted_at_val_threshold"])
        for y, p, vid in zip(v_true, v_prob, v_ids):
            writer.writerow([videos[vid]["name"], "Fake" if y else "Real", f"{p:.6f}",
                             "Fake" if p >= 0.5 else "Real", "Fake" if p >= threshold else "Real"])


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_split(path, videos, split, summary):
    write_json(path, {"summary": summary,
                      **{part: [[videos[i]["name"], "Fake" if videos[i]["label"] else "Real"]
                                for i in split[part]] for part in ("train", "val", "test", "dropped")}})


def test_model(model, checkpoint, cfg, videos, test_idx, device, args, extra):
    """Test a model on the given videos, print and save the report."""
    model.load_state_dict(checkpoint["model"])
    threshold = float(checkpoint.get("threshold", 0.5))
    te_s, te_y, te_v = make_clips(videos, test_idx)
    test_loader = make_loader(te_s, te_y, cfg, False, args.seed, args.batch_size, args.num_workers)
    y_te, p_te, te_loss = predict(model, test_loader, device)
    report, (v_true, v_prob, v_ids) = full_report(y_te, p_te, te_v)
    report["threshold"] = threshold
    report["video_level_at_val_threshold"] = compute_metrics(v_true, v_prob, threshold)
    report.update({"test_loss": te_loss, "checkpoint_epoch": checkpoint.get("epoch"),
                   "seed": args.seed, "mode": cfg["mode"], "temporal": cfg["temporal"],
                   "motion": cfg["motion"], "train_flow_root": cfg.get("train_flow_root"),
                   "test_flow_root": args.flow_root, **extra})
    print_report("TEST", report)
    write_json(os.path.join(args.out_dir, "results.json"), report)
    save_predictions(os.path.join(args.out_dir, "test_predictions.csv"), videos, v_true, v_prob, v_ids,
                     threshold)
    return report


def add_runtime_args(ap):
    ap.add_argument("--seed", type=int, default=42, help="seed of the split, the sampling and the weights")
    ap.add_argument("--batch_size", type=int, default=4, help="clips per batch")
    ap.add_argument("--num_workers", type=int, default=2, help="data-loading processes (0 = main process)")


def build_parser():
    ap = argparse.ArgumentParser(description="Train the deepfake detector, then test it")
    ap.add_argument("--flow_root", required=True, help="folder written by deepfake_flow.preprocess")
    ap.add_argument("--out_dir", required=True, help="where the results of this run are written")
    ap.add_argument("--mode", choices=MODES, default="two_stream")
    ap.add_argument("--temporal", choices=TEMPORAL, default="lstm")
    ap.add_argument("--motion", choices=["affine", "mean", "none"], default="affine",
                    help="head motion removed from the flow")
    ap.add_argument("--test_list", default=None, help="official test list (e.g. Celeb-DF)")
    ap.add_argument("--ffpp_splits", default=None,
                    help="folder with the official FaceForensics++ train.json / val.json / test.json")
    ap.add_argument("--folds", type=int, default=0, help="K-fold split by identity (e.g. 7); 0 = one split")
    ap.add_argument("--fold", type=int, default=0, help="which fold is the test set (0 .. folds-1)")
    ap.add_argument("--group_by", choices=["prefix", "video"], default="prefix",
                    help="identity = text before the first '_' (default) or each video alone")
    ap.add_argument("--val_fraction", type=float, default=0.15, help="share of identities held out for validation")
    ap.add_argument("--test_fraction", type=float, default=0.15, help="share of identities held out for test (own split only)")
    add_runtime_args(ap)
    ap.add_argument("--accumulation_steps", type=int, default=4, help="batches per optimiser step (effective batch 16)")
    ap.add_argument("--epochs", type=int, default=30, help="maximum number of epochs")
    ap.add_argument("--patience", type=int, default=5, help="stop after this many epochs without a better validation AUC")
    ap.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--no_flip", action="store_true", help="disable horizontal-flip augmentation")
    ap.add_argument("--no_pretrained", action="store_true", help="do not load ImageNet weights")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    videos = list_videos(args.flow_root)
    prep = load_preprocess_config(args.flow_root)
    print(f"{len(videos)} videos, clip length {videos[0]['clip_len']}, size {videos[0]['size']}")
    test_names = read_name_list(args.test_list) if args.test_list else None

    split, group_by = make_split(videos, seed=args.seed, test_names=test_names,
                                 ffpp_splits=args.ffpp_splits, folds=args.folds, fold=args.fold,
                                 val_fraction=args.val_fraction, test_fraction=args.test_fraction,
                                 group_by=args.group_by)
    args.group_by = group_by
    tr_s, tr_y, _ = make_clips(videos, split["train"])
    va_s, va_y, va_v = make_clips(videos, split["val"])
    n_real, n_fake = tr_y.count(0), tr_y.count(1)
    if n_real == 0 or n_fake == 0:
        raise SystemExit("The training split has only one class.")
    cfg = {"mode": args.mode, "temporal": args.temporal, "motion": args.motion,
           "flip": not args.no_flip, "seed": args.seed,
           "flow_scale": estimate_flow_scale(tr_s, args.motion, seed=args.seed),
           "clip_len": videos[0]["clip_len"], "size": videos[0]["size"],
           "preprocess": prep, "train_flow_root": args.flow_root,
           # weight of class c = n / (2 n_c): both classes count equally in the loss
           "class_weights": [len(tr_y) / (2.0 * n_real), len(tr_y) / (2.0 * n_fake)]}
    model = build_model(cfg, pretrained=not args.no_pretrained).to(device)

    summary = describe_split(videos, split)
    print("Split:", json.dumps(summary))
    print("Config:", json.dumps({k: v for k, v in cfg.items() if k != "preprocess"}))
    write_json(os.path.join(args.out_dir, "config.json"), {"args": vars(args), "config": cfg})
    write_split(os.path.join(args.out_dir, "split.json"), videos, split, summary)

    train_loader = make_loader(tr_s, tr_y, cfg, True, args.seed, args.batch_size, args.num_workers)
    val_loader = make_loader(va_s, va_y, cfg, False, args.seed, args.batch_size, args.num_workers)
    best_path = train_model(model, train_loader, val_loader, va_v, cfg["class_weights"], device, args, cfg)

    # decision threshold chosen on the validation videos, stored with the model
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    y_va, p_va, _ = predict(model, val_loader, device)
    v_va, pv_va, _ = aggregate_by_video(y_va, p_va, va_v)
    checkpoint["threshold"] = choose_threshold(v_va, pv_va)
    torch.save(checkpoint, best_path)

    protocol = ("official FaceForensics++ split" if args.ffpp_splits else
                "official test list" if test_names else "own split by group")
    test_model(model, checkpoint, cfg, videos, split["test"], device, args,
               {"folds": args.folds or None, "fold": args.fold if args.folds else None,
                "checkpoint": best_path, "protocol": protocol})
    print(f"\nSaved config.json, split.json, history.json, best.pth, results.json, "
          f"test_predictions.csv in {args.out_dir}")


if __name__ == "__main__":
    main()
