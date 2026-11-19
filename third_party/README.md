# RAFT

The optical flow is computed with RAFT (Teed and Deng, ECCV 2020), which is not part
of this repository. Clone it here and download its pretrained models:

    git clone https://github.com/princeton-vl/RAFT third_party/RAFT
    cd third_party/RAFT && ./download_models.sh

Only `models/raft-sintel.pth` is used. On Windows, download `models.zip` from the link
in `download_models.sh` and unpack it into `third_party/RAFT/models`.
Another location can be given with `--raft_dir` / `--raft_weights` or the
`DEEPFAKE_RAFT_DIR` / `DEEPFAKE_RAFT_WEIGHTS` environment variables.
