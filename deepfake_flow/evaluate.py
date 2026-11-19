"""Test a trained model, unchanged, on a preprocessed dataset (e.g. cross-dataset).

The model keeps its own flow scale and decision threshold. All videos of
--flow_root are tested, or only those of --test_list. Outputs in --out_dir:
config.json, split.json, results.json, test_predictions.csv.

Example (model trained on FaceForensics++, tested on the official Celeb-DF test videos)
    python -m deepfake_flow.evaluate --checkpoint runs/ffpp/best.pth --flow_root <flows>/CelebDF \
        --test_list <Celeb-DF>/List_of_testing_videos.txt --out_dir runs/ffpp_to_celeb
"""

import argparse
import json
import os

import torch

from .data import describe_split, list_videos, load_preprocess_config, read_name_list
from .train import add_runtime_args, build_model, set_seed, test_model, write_json, write_split


def build_parser():
    ap = argparse.ArgumentParser(description="Test a trained deepfake detector on a dataset")
    ap.add_argument("--checkpoint", required=True, help="best.pth written by deepfake_flow.train")
    ap.add_argument("--flow_root", required=True, help="folder written by deepfake_flow.preprocess")
    ap.add_argument("--out_dir", required=True, help="where the results of this run are written")
    ap.add_argument("--test_list", default=None, help="test only these videos (e.g. official list)")
    add_runtime_args(ap)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    videos = list_videos(args.flow_root)
    prep = load_preprocess_config(args.flow_root)
    test_names = read_name_list(args.test_list) if args.test_list else None
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    if prep is not None and cfg.get("preprocess") is not None and prep != cfg["preprocess"]:
        diff = {k: (cfg["preprocess"].get(k), prep.get(k)) for k in set(prep) | set(cfg["preprocess"])
                if cfg["preprocess"].get(k) != prep.get(k)}
        print(f"Note: preprocessing settings differ from the training data (training, here): {diff}")

    test_idx = [i for i, v in enumerate(videos) if not test_names or v["name"] in test_names]
    split = {"train": [], "val": [], "test": test_idx, "dropped": []}
    summary = describe_split(videos, split)
    print(f"{len(test_idx)} test videos:", json.dumps(summary["test"]))
    write_json(os.path.join(args.out_dir, "config.json"), {"args": vars(args), "config": cfg})
    write_split(os.path.join(args.out_dir, "split.json"), videos, split, summary)

    model = build_model(cfg, pretrained=False).to(device)
    test_model(model, checkpoint, cfg, videos, test_idx, device, args,
               {"folds": None, "fold": None, "checkpoint": args.checkpoint, "protocol": "cross-dataset"})
    print(f"\nSaved config.json, split.json, results.json, test_predictions.csv in {args.out_dir}")


if __name__ == "__main__":
    main()
