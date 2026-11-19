"""Deepfake video detection with motion-compensated optical flow.

Modules
    paths       where the RAFT code, RAFT weights and the face-detection model are found
    faces       face detection, tracking and choice of the clips
    flow        RAFT optical flow and the flow transforms (head-motion removal, flip, scale)
    preprocess  video -> face clips -> optical flow, saved as .npy files      (command line)
    data        listing preprocessed videos, identity-safe splits, PyTorch dataset
    metrics     clip and video level metrics, threshold choice, bootstrap interval
    model       the two-stream ResNet18 + LSTM network
    train       training, then test on the held-out videos                    (command line)
    evaluate    test a trained model on any preprocessed dataset              (command line)
    predict     verdict for one video file                                    (command line)
    summarize   one table of all results                                      (command line)
    experiments every experiment of the project, in order                     (command line)
    check_env   environment check on random data                             (command line)

Command-line modules are run from the repository root, e.g.
    python -m deepfake_flow.train --help
"""

__version__ = "2.0.0"
