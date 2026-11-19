"""Metrics. Fake is the positive class everywhere (0 = Real, 1 = Fake).

A video's score is the mean of its clips' probabilities of being fake. The
decision threshold is chosen on validation videos only, never on test videos.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, p_fake, threshold=0.5):
    """Binary metrics with Fake as the positive class (0 = Real, 1 = Fake)."""
    y_true = np.asarray(y_true).astype(int)
    p_fake = np.asarray(p_fake, dtype=float)
    y_pred = (p_fake >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "auc": float(roc_auc_score(y_true, p_fake)) if both_classes else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(0.5 * (tp / (tp + fn) + tn / (tn + fp))) if both_classes else None,
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        # share of real items wrongly flagged as fake
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "confusion_matrix": {"TN_real_as_real": int(tn), "FP_real_as_fake": int(fp),
                             "FN_fake_as_real": int(fn), "TP_fake_as_fake": int(tp)},
    }


def choose_threshold(y_true, p_fake):
    """Decision threshold that maximises balanced accuracy (Youden's J).

    Chosen on the validation videos and then applied unchanged to the test
    videos, so the test set plays no part in it. 0.5 when only one class.
    """
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, np.asarray(p_fake, dtype=float))
    best = int(np.argmax(tpr - fpr))
    return float(min(max(thr[best], 0.0), 1.0))


def aggregate_by_video(y_true, p_fake, video_ids):
    """Average the Fake probability of all clips of a video."""
    sums, counts, truth = defaultdict(float), defaultdict(int), {}
    for y, p, v in zip(y_true, p_fake, video_ids):
        sums[v] += float(p)
        counts[v] += 1
        truth[v] = int(y)
    vids = sorted(sums)
    return ([truth[v] for v in vids], [sums[v] / counts[v] for v in vids], vids)


def bootstrap_auc_ci(y_true, p_fake, n_boot=1000, seed=0):
    """95 % interval of the AUC by resampling the videos with replacement."""
    y_true = np.asarray(y_true).astype(int)
    p_fake = np.asarray(p_fake, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            aucs.append(roc_auc_score(y_true[idx], p_fake[idx]))
    if not aucs:
        return None
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]

