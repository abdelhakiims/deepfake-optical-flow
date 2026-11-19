"""Locations of the third-party components.

Nothing is hard-coded to one machine. Each location can be given on the command
line, through an environment variable, or left to its default inside the repository:

    RAFT code       --raft_dir     DEEPFAKE_RAFT_DIR    third_party/RAFT
    RAFT weights    --raft_weights DEEPFAKE_RAFT_WEIGHTS <RAFT code>/models/raft-sintel.pth
    face detector   --face_model   DEEPFAKE_FACE_MODEL  models/face_detection_yunet_2023mar.onnx
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAFT_URL = "https://github.com/princeton-vl/RAFT"
FACE_MODEL_URL = ("https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet "
                  "(file face_detection_yunet_2023mar.onnx)")


def raft_dir(override=None):
    return override or os.environ.get("DEEPFAKE_RAFT_DIR") or os.path.join(REPO_ROOT, "third_party", "RAFT")


def raft_weights(override=None, raft_code_dir=None):
    return (override or os.environ.get("DEEPFAKE_RAFT_WEIGHTS")
            or os.path.join(raft_dir(raft_code_dir), "models", "raft-sintel.pth"))


def face_model(override=None):
    return (override or os.environ.get("DEEPFAKE_FACE_MODEL")
            or os.path.join(REPO_ROOT, "models", "face_detection_yunet_2023mar.onnx"))
