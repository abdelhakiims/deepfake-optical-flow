"""Every experiment of the project, in order, with one command per stage.

Each step is skipped when its output already exists, so a stage can be stopped and
started again. Success is judged by the output files (results.json, .flow.npy),
not by exit codes. Datasets are expected in their official layout:

    <dfd_dir>      original_sequences/actors/c23/videos, manipulated_sequences/DeepFakeDetection/c23/videos
    <celebdf_dir>  Celeb-real, YouTube-real, Celeb-synthesis, List_of_testing_videos.txt
    <ffpp_dir>     original_sequences/youtube/c23/videos, manipulated_sequences/<method>/c23/videos

Stages
    preprocess-dfd      363 real + 1,000 fakes                                    (~3 h)
    preprocess-celebdf  890 real + 1,500 fakes, all official test fakes included  (~3 h)
    preprocess-ffpp     1,000 real + 500 fakes per method, all val/test fakes     (~5 h)
    dfd-cv              7-fold cross-validation by actor on DFD                   (~5 h)
    dfd-to-celebdf      the 7 DFD models tested on the official Celeb-DF test set (minutes)
    ffpp                6 configurations trained on FaceForensics++, each tested
                        on FaceForensics++, Celeb-DF and DFD                      (~1 h each)
    summary             one table of all results

Example
    python -m deepfake_flow.experiments ffpp summary --flows_dir <flows> --celebdf_dir <Celeb-DF-v2>
"""

import argparse
import os
import subprocess
import sys

from .paths import REPO_ROOT

FFPP_METHODS = [("Deepfakes", "DF-"), ("Face2Face", "F2F-"), ("FaceSwap", "FS-"),
                ("NeuralTextures", "NT-")]
FFPP_SPLITS = os.path.join(REPO_ROOT, "splits", "ffpp")
# name, options: each changes one thing compared with the full model
FFPP_CONFIGS = [
    ("ffpp_two_stream_s42", ["--mode", "two_stream"]),   # full model
    ("ffpp_rgb_s42", ["--mode", "rgb"]),                 # face images only
    ("ffpp_flow_s42", ["--mode", "flow"]),               # optical flow only
    ("ffpp_nomotion_s42", ["--motion", "none"]),         # no head-motion removal
    ("ffpp_meanmotion_s42", ["--motion", "mean"]),       # translation removed only
    ("ffpp_meantemp_s42", ["--temporal", "mean"]),       # mean over time instead of the LSTM
]


def run(module, *args):
    """Run one command-line module of the package in a separate process."""
    cmd = [sys.executable, "-m", f"deepfake_flow.{module}", *map(str, args)]
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT)


def done(out_dir):
    return os.path.isfile(os.path.join(out_dir, "results.json"))


def need(value, flag):
    if not value:
        raise SystemExit(f"This stage needs {flag}")
    return value


def preprocess(a, videos_dir, label, out_root, *extra):
    run("preprocess", "--videos_dir", videos_dir, "--label", label, "--out_root", out_root, *extra,
        *a.tool_args)


def stage_preprocess_dfd(a):
    root, out = need(a.dfd_dir, "--dfd_dir"), os.path.join(a.flows_dir, "DFD")
    preprocess(a, os.path.join(root, "original_sequences", "actors", "c23", "videos"), "Real", out)
    preprocess(a, os.path.join(root, "manipulated_sequences", "DeepFakeDetection", "c23", "videos"),
               "Fake", out, "--max_videos", 1000)


def stage_preprocess_celebdf(a):
    root, out = need(a.celebdf_dir, "--celebdf_dir"), os.path.join(a.flows_dir, "CelebDF")
    test_list = os.path.join(root, "List_of_testing_videos.txt")
    preprocess(a, os.path.join(root, "Celeb-real"), "Real", out)
    preprocess(a, os.path.join(root, "YouTube-real"), "Real", out)
    preprocess(a, os.path.join(root, "Celeb-synthesis"), "Fake", out,
               "--max_videos", 1500, "--must_include", test_list)


def stage_preprocess_ffpp(a):
    root, out = need(a.ffpp_dir, "--ffpp_dir"), os.path.join(a.flows_dir, "FFPP")
    # FaceForensics++ videos are often low resolution: faces of 50 px and more are kept
    preprocess(a, os.path.join(root, "original_sequences", "youtube", "c23", "videos"), "Real", out,
               "--min_face", 50)
    for method, prefix in FFPP_METHODS:
        preprocess(a, os.path.join(root, "manipulated_sequences", method, "c23", "videos"), "Fake", out,
                   "--min_face", 50, "--name_prefix", prefix, "--max_videos", a.ffpp_per_method,
                   "--must_include_ffpp", FFPP_SPLITS)


def stage_dfd_cv(a):
    flows = os.path.join(a.flows_dir, "DFD")
    for k in range(7):
        out = os.path.join(a.runs_dir, f"dfd_two_stream_f{k}")
        if not done(out):
            run("train", "--flow_root", flows, "--folds", 7, "--fold", k, "--out_dir", out,
                "--seed", 42, *a.extra)


def evaluate(a, name, flows, *extra):
    out = os.path.join(a.runs_dir, f"{name}_to_{os.path.basename(flows).lower()}")
    if done(os.path.join(a.runs_dir, name)) and not done(out):
        run("evaluate", "--checkpoint", os.path.join(a.runs_dir, name, "best.pth"),
            "--flow_root", flows, "--out_dir", out, *extra)


def celebdf_test(a):
    flows = os.path.join(a.flows_dir, "CelebDF")
    return flows, ["--test_list", os.path.join(need(a.celebdf_dir, "--celebdf_dir"),
                                               "List_of_testing_videos.txt")]


def stage_dfd_to_celebdf(a):
    flows, extra = celebdf_test(a)
    for k in range(7):
        evaluate(a, f"dfd_two_stream_f{k}", flows, *extra)


def stage_ffpp(a):
    flows = os.path.join(a.flows_dir, "FFPP")
    celeb, celeb_extra = celebdf_test(a)
    for name, options in FFPP_CONFIGS:
        out = os.path.join(a.runs_dir, name)
        if not done(out):
            run("train", "--flow_root", flows, "--ffpp_splits", FFPP_SPLITS, "--out_dir", out,
                "--seed", 42, *options, *a.extra)
        if not done(out):
            raise SystemExit(f"Training {name} did not finish; see the messages above.")
        evaluate(a, name, celeb, *celeb_extra)
        evaluate(a, name, os.path.join(a.flows_dir, "DFD"))


def stage_summary(a):
    run("summarize", "--runs_dir", a.runs_dir)


STAGES = {"preprocess-dfd": stage_preprocess_dfd, "preprocess-celebdf": stage_preprocess_celebdf,
          "preprocess-ffpp": stage_preprocess_ffpp, "dfd-cv": stage_dfd_cv,
          "dfd-to-celebdf": stage_dfd_to_celebdf, "ffpp": stage_ffpp, "summary": stage_summary}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the experiments of the project",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("stages", nargs="+", choices=list(STAGES), help="stages to run, in this order")
    ap.add_argument("--flows_dir", required=True, help="where the preprocessed datasets are (or go)")
    ap.add_argument("--runs_dir", default=os.path.join(REPO_ROOT, "runs"))
    ap.add_argument("--dfd_dir", default=None)
    ap.add_argument("--celebdf_dir", default=None)
    ap.add_argument("--ffpp_dir", default=None)
    ap.add_argument("--ffpp_per_method", type=int, default=500,
                    help="fakes per FaceForensics++ method (0 = all 1,000)")
    ap.add_argument("--face_model", default=None)
    ap.add_argument("--raft_dir", default=None)
    ap.add_argument("--raft_weights", default=None)
    a, extra = ap.parse_known_args(argv)
    a.extra = extra       # anything else is passed on to training (e.g. --num_workers 0)
    a.tool_args = [x for flag in ("face_model", "raft_dir", "raft_weights")
                   for x in ((f"--{flag}", getattr(a, flag)) if getattr(a, flag) else ())]
    for stage in a.stages:
        print(f"\n===== {stage} =====", flush=True)
        STAGES[stage](a)


if __name__ == "__main__":
    main()
