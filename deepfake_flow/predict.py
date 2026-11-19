"""Verdict for one video file.

Uses the trained weights, the preprocessing settings, the flow scale and the
decision threshold stored in the checkpoint, so a new video is treated exactly
like the training videos.

Example
    python -m deepfake_flow.predict --video my_video.mp4 --checkpoint runs/ffpp/best.pth
"""

import argparse

import numpy as np
import torch

from .faces import FaceDetector
from .flow import compensate_motion, load_raft
from .model import DeepfakeNet
from .preprocess import DEFAULT_CONFIG, process_video


class DeepfakeDetector:
    """Loads RAFT, the face detector and the trained model once; then predicts."""

    def __init__(self, checkpoint, face_model=None, raft_dir=None, raft_weights=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.cfg = ckpt["config"]
        self.threshold = float(ckpt.get("threshold", 0.5))   # chosen on validation videos
        self.prep = dict(DEFAULT_CONFIG)
        self.prep.update(self.cfg.get("preprocess") or {})

        self.model = DeepfakeNet(mode=self.cfg["mode"], temporal=self.cfg["temporal"],
                                 pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.raft = load_raft(self.device, raft_weights, raft_dir)
        self.face = FaceDetector(face_model) if self.prep["face_crop"] else None

    @torch.no_grad()
    def predict(self, video_path):
        """Probability that the video is fake, averaged over its clips."""
        flows, rgb, info = process_video(video_path, self.raft, self.face, self.device, self.prep)
        probs = []
        for k in range(flows.shape[0]):
            flow = compensate_motion(flows[k], self.cfg["motion"]) / float(self.cfg["flow_scale"])
            flow_t = torch.from_numpy(np.ascontiguousarray(flow, dtype=np.float32)).unsqueeze(0)
            frames = rgb[k][:rgb.shape[1] - 1]
            rgb_t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).unsqueeze(0)
            logits = self.model(flow_t.to(self.device), rgb_t.to(self.device)).float()
            probs.append(float(torch.softmax(logits, dim=1)[0, 1]))
        p_fake = float(np.mean(probs))
        return {"p_fake": p_fake,
                "verdict": "FAKE" if p_fake >= self.threshold else "REAL",
                "threshold": self.threshold,
                "clips": len(probs),
                "clip_p_fake": [round(p, 4) for p in probs],
                "frames_total": info["frames_total"],
                "people_seen": info["max_faces"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Is this video a deepfake?")
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", required=True, help="best.pth written by deepfake_flow.train")
    ap.add_argument("--face_model", default=None, help="YuNet .onnx file (see paths.py)")
    ap.add_argument("--raft_dir", default=None, help="clone of the RAFT repository (see paths.py)")
    ap.add_argument("--raft_weights", default=None, help="RAFT weights file (see paths.py)")
    args = ap.parse_args(argv)

    detector = DeepfakeDetector(args.checkpoint, args.face_model, args.raft_dir, args.raft_weights)
    result = detector.predict(args.video)
    print(f"Verdict: {result['verdict']}  (probability of fake: {result['p_fake']:.3f}, "
          f"averaged over {result['clips']} clips)")
    print(f"Per clip: {result['clip_p_fake']}")
    if result["people_seen"] > 1:
        print(f"Note: up to {result['people_seen']} faces were visible; the largest one was analysed.")


if __name__ == "__main__":
    main()
