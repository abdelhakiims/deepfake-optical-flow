"""One table of all results.

Reads every <runs_dir>/*/results.json, prints one line per run and the mean and
standard deviation over runs (seeds and folds) of each configuration, and writes
summary.csv and summary_by_config.csv in <runs_dir>.

Example
    python -m deepfake_flow.summarize --runs_dir runs
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean, pstdev


def data_name(path):
    """Last folder name of a dataset path, whichever slash it uses."""
    return (path or "").replace("\\", "/").rstrip("/").split("/")[-1]


def load_runs(runs_dir):
    rows = []
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name, "results.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)
        v, c = r["video_level"], r["clip_level"]
        ci = v.get("auc_ci95") or [None, None]
        rows.append({
            "run": name,
            "train_data": data_name(r.get("train_flow_root")),
            "test_data": data_name(r.get("test_flow_root")),
            "protocol": r.get("protocol"), "mode": r.get("mode"), "motion": r.get("motion"),
            "temporal": r.get("temporal"), "seed": r.get("seed"), "fold": r.get("fold"),
            "videos": v["n"], "video_auc": v["auc"], "auc_ci_low": ci[0], "auc_ci_high": ci[1],
            "video_acc": v["accuracy"], "video_bal_acc": v["balanced_accuracy"],
            "video_f1": v["f1_fake"], "false_alarm": v["false_alarm_rate"],
            "clip_auc": c["auc"], "clip_acc": c["accuracy"],
            "threshold": r.get("threshold"),
            "bal_acc_val_thr": (r.get("video_level_at_val_threshold") or {}).get("balanced_accuracy"),
            "false_alarm_val_thr": (r.get("video_level_at_val_threshold") or {}).get("false_alarm_rate"),
        })
    return rows


def by_config(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r["train_data"], r["test_data"], r["protocol"], r["mode"], r["motion"], r["temporal"])
        groups[key].append(r)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = dict(zip(["train_data", "test_data", "protocol", "mode", "motion", "temporal"], key))
        row["runs"] = len(items)
        for metric in ("video_auc", "video_acc", "video_bal_acc", "false_alarm", "clip_auc",
                       "bal_acc_val_thr", "false_alarm_val_thr"):
            values = [r[metric] for r in items if r[metric] is not None]
            row[metric + "_mean"] = round(mean(values), 4) if values else None
            row[metric + "_std"] = round(pstdev(values), 4) if len(values) > 1 else None
        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x):
    return "  n/a " if x is None else f"{x:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Collect all results into one table")
    ap.add_argument("--runs_dir", default="runs")
    args = ap.parse_args(argv)

    rows = load_runs(args.runs_dir)
    if not rows:
        raise SystemExit(f"No results.json found under {args.runs_dir}")

    print(f"{'run':34s} {'train->test':22s} {'mode':11s} {'motion':7s} {'temp':5s} seed/fold  "
          f"vidAUC  [95% interval]    vidAcc  falseAl clipAUC")
    for r in rows:
        pair = f"{r['train_data']}->{r['test_data']}"
        print(f"{r['run'][:34]:34s} {pair[:22]:22s} {str(r['mode']):11s} {str(r['motion']):7s} "
              f"{str(r['temporal']):5s} {str(r['seed']) + '/' + str(r['fold']):9s}  {fmt(r['video_auc'])}  "
              f"[{fmt(r['auc_ci_low'])}, {fmt(r['auc_ci_high'])}]  {fmt(r['video_acc'])}  "
              f"{fmt(r['false_alarm'])}  {fmt(r['clip_auc'])}")

    grouped = by_config(rows)
    print("\nMean over runs (seeds and folds; std in brackets):")
    for g in grouped:
        std = "" if g["video_auc_std"] is None else f" ({g['video_auc_std']:.4f})"
        print(f"  {g['train_data']}->{g['test_data']}  {g['mode']}/{g['motion']}/{g['temporal']}  "
              f"runs={g['runs']}  video AUC {fmt(g['video_auc_mean'])}{std}  "
              f"video acc {fmt(g['video_acc_mean'])}")

    write_csv(os.path.join(args.runs_dir, "summary.csv"), rows)
    write_csv(os.path.join(args.runs_dir, "summary_by_config.csv"), grouped)
    print(f"\nWrote summary.csv and summary_by_config.csv in {args.runs_dir}")


if __name__ == "__main__":
    main()
