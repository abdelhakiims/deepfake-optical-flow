"""Face detection, tracking of the main subject, and choice of the clips.

How the face of a video is chosen
  1. Faces are detected (OpenCV YuNet) on about 8 frames per second.
  2. Detections are linked into tracks, one per person (nearest box centre).
  3. The MAIN SUBJECT is the track with the largest total area of frontal
     detections. Frontality = distance between the eyes / face width: a person seen
     from the side or from behind has the eyes close together in the image.
  4. A scan frame is valid when the main subject is present, frontal enough, large
     enough, inside the frame, and no other frontal face of similar size is visible
     (otherwise the swapped face cannot be identified automatically).
  5. Clips are placed where every scan frame is valid; the crop box is interpolated
     between scan frames and smoothed over time.
"""

import os

import numpy as np

from . import paths


class FaceDetector:
    """Faces in a frame, with OpenCV's YuNet detector."""

    def __init__(self, model_path=None, score_threshold=0.6, max_side=640):
        import cv2
        model_path = paths.face_model(model_path)
        if not os.path.isfile(model_path):
            raise SystemExit(
                f"Face model not found: {model_path}\n"
                f"Download it from {paths.FACE_MODEL_URL} into the models/ folder, "
                "or pass --face_model <path>.")
        self.cv2 = cv2
        self.max_side = max_side
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)

    def faces(self, frame_bgr):
        """Faces as dicts: x, y, w, h (full-frame pixels), frontal, score."""
        cv2 = self.cv2
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / float(max(h, w)))
        small = frame_bgr
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, found = self.detector.detect(small)
        out = []
        if found is not None:
            for r in found:
                fw = float(r[2]) / scale
                eyes = float(np.hypot(r[4] - r[6], r[5] - r[7])) / scale
                out.append({"x": float(r[0]) / scale, "y": float(r[1]) / scale, "w": fw,
                            "h": float(r[3]) / scale, "frontal": eyes / max(fw, 1e-6),
                            "score": float(r[14])})
        return out


def link_tracks(scan, max_gap=3):
    """Link the detections of the scan frames into tracks, one per person.

    scan: list of (frame_index, faces); a face is a dict with x, y, w, h, frontal.
    A face joins the track whose last box centre is nearest, if that centre is
    within 0.6 face widths and the track was seen in the last `max_gap` scan frames.
    Returns a list of tracks; a track is a dict {scan position: face}.
    """
    tracks, last = [], []          # last[t] = (scan position, face) of track t
    for pos, (_, faces) in enumerate(scan):
        taken = set()
        order = sorted(range(len(faces)), key=lambda i: -faces[i]["w"] * faces[i]["h"])
        for i in order:
            f = faces[i]
            cx, cy = f["x"] + f["w"] / 2.0, f["y"] + f["h"] / 2.0
            best, best_d = None, None
            for t, (p, g) in enumerate(last):
                if t in taken or pos - p > max_gap:
                    continue
                d = np.hypot(cx - (g["x"] + g["w"] / 2.0), cy - (g["y"] + g["h"] / 2.0))
                if d <= 0.6 * max(g["w"], f["w"]) and (best_d is None or d < best_d):
                    best, best_d = t, d
            if best is None:
                tracks.append({pos: f})
                last.append((pos, f))
                taken.add(len(tracks) - 1)
            else:
                tracks[best][pos] = f
                last[best] = (pos, f)
                taken.add(best)
    return tracks


def main_track(tracks, min_frontal):
    """Index of the track with the largest total area of frontal detections."""
    if not tracks:
        return None
    scores = [sum(f["w"] * f["h"] for f in t.values() if f["frontal"] >= min_frontal) for t in tracks]
    best = int(np.argmax(scores))
    return best if scores[best] > 0 else None


def valid_scan_positions(scan, tracks, main, cfg, frame_size=None):
    """Scan positions where the main subject is usable and not ambiguous.

    Returns (valid, reasons): valid is a list of booleans; reasons counts why
    positions were rejected (absent, profile, small, outside, ambiguous).
    """
    valid, reasons = [], {"absent": 0, "profile": 0, "small": 0, "outside": 0, "ambiguous": 0}
    subject = tracks[main]
    for pos, (_, faces) in enumerate(scan):
        f = subject.get(pos)
        why = None
        if f is None:
            why = "absent"
        elif f["frontal"] < cfg["min_frontal"]:
            why = "profile"
        elif min(f["w"], f["h"]) < cfg["min_face"]:
            why = "small"
        elif frame_size is not None and (
                f["x"] < -0.05 * f["w"] or f["y"] < -0.05 * f["h"]
                or f["x"] + f["w"] > frame_size[0] + 0.05 * f["w"]
                or f["y"] + f["h"] > frame_size[1] + 0.05 * f["h"]):
            why = "outside"
        else:
            area = f["w"] * f["h"]
            for g in faces:
                if g is not f and g["frontal"] >= cfg["rival_frontal"] and g["w"] * g["h"] >= cfg["rival_area"] * area:
                    why = "ambiguous"
                    break
        if why:
            reasons[why] += 1
        valid.append(why is None)
    return valid, reasons


def choose_windows(scan_frames, valid, n_frames, span, n_clips):
    """Start frames of up to n_clips non-overlapping windows of `span` frames.

    A window is usable when every scan frame from the last one at or before its
    start to the first one at or after its end is valid, so the box can be
    interpolated on every frame of it. The windows are spread over the video.
    """
    scan_frames = np.asarray(scan_frames)
    ok_starts = []
    for s in range(0, n_frames - span + 1):
        e = s + span - 1
        lo = np.searchsorted(scan_frames, s, side="right") - 1
        hi = np.searchsorted(scan_frames, e, side="left")
        if lo < 0 or hi >= len(scan_frames):
            continue
        if all(valid[lo:hi + 1]):
            ok_starts.append(s)
    if not ok_starts:
        return []
    chosen = []
    for target in np.linspace(ok_starts[0], ok_starts[-1], n_clips):
        free = [s for s in ok_starts if all(abs(s - c) >= span for c in chosen)]
        if not free:
            break
        chosen.append(min(free, key=lambda s: abs(s - target)))
    return sorted(chosen)


def interpolate_boxes(scan_frames, subject, frames):
    """Box (cx, cy, size) of the main subject on each requested frame.

    Linear interpolation between the subject's scan detections; size = the larger
    side of the detected box.
    """
    known = sorted((scan_frames[p], d) for p, d in subject.items())
    xs = np.array([k for k, _ in known], dtype=float)
    cx = np.array([d["x"] + d["w"] / 2.0 for _, d in known])
    cy = np.array([d["y"] + d["h"] / 2.0 for _, d in known])
    sz = np.array([max(d["w"], d["h"]) for _, d in known])
    f = np.asarray(frames, dtype=float)
    return np.stack([np.interp(f, xs, cx), np.interp(f, xs, cy), np.interp(f, xs, sz)], axis=1)


def smooth_boxes(boxes, k):
    """Moving average over k frames, with the edges padded by their own values."""
    if k <= 1 or len(boxes) < 2:
        return boxes
    pad = k // 2
    padded = np.concatenate([np.repeat(boxes[:1], pad, 0), boxes, np.repeat(boxes[-1:], pad, 0)])
    kernel = np.ones(k) / k
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(boxes.shape[1])], axis=1)

