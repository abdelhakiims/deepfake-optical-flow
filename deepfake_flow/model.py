"""The detector network. Shared by training, evaluation and prediction.

Two streams over a clip of T time steps, fused by a recurrent layer:

    motion stream      residual optical flow (2 channels) -> ResNet18 -> 512 values per step
    appearance stream  face crop (3 channels)             -> ResNet18 -> 512 values per step
    fusion             [motion, appearance] per step -> LSTM -> real / fake

Modes (for the ablations):
    'two_stream'  both streams (the full model)
    'flow'        motion stream only
    'rgb'         appearance stream only
Temporal head:
    'lstm'        2-layer LSTM, last hidden state
    'mean'        average of the per-step features (no temporal modelling)
"""

import torch
import torch.nn as nn
from torchvision import models

MODES = ("two_stream", "flow", "rgb")
TEMPORAL = ("lstm", "mean")


def make_encoder(in_channels, pretrained=True):
    """ResNet18 that returns the 512 pooled features.

    For 2 input channels the first layer cannot reuse the ImageNet filters
    directly (they expect 3 channels). Each new filter is the average of the
    pretrained filter over its 3 colour channels, copied to both flow channels,
    so the layer starts from trained edge detectors, not from random values.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    if in_channels != 3:
        old = net.conv1
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        net.conv1 = new
    net.fc = nn.Identity()
    return net


class DeepfakeNet(nn.Module):
    def __init__(self, mode="two_stream", temporal="lstm", hidden_size=128,
                 num_layers=2, dropout=0.3, pretrained=True):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if temporal not in TEMPORAL:
            raise ValueError(f"temporal must be one of {TEMPORAL}")
        self.mode = mode
        self.temporal = temporal
        self.use_flow = mode in ("two_stream", "flow")
        self.use_rgb = mode in ("two_stream", "rgb")

        feat_dim = 0
        if self.use_flow:
            self.flow_encoder = make_encoder(2, pretrained)
            feat_dim += 512
        if self.use_rgb:
            self.rgb_encoder = make_encoder(3, pretrained)
            feat_dim += 512
            # ImageNet statistics, for images scaled to 0..1
            self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if temporal == "lstm":
            self.lstm = nn.LSTM(feat_dim, hidden_size, num_layers, batch_first=True,
                                dropout=0.1 if num_layers > 1 else 0.0)
            head_dim = hidden_size
        else:
            head_dim = feat_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_dim, 2)

    def forward(self, flow, rgb):
        """flow: float (B, T, 2, H, W), already compensated and scaled.
        rgb:  uint8 or float (B, T, 3, H, W) with values 0..255.
        A stream that the mode does not use may be an empty tensor.
        Returns logits (B, 2); index 1 = Fake."""
        feats = []
        if self.use_flow:
            B, T = flow.shape[:2]
            f = self.flow_encoder(flow.reshape(B * T, *flow.shape[2:]).float())
            feats.append(f.reshape(B, T, -1))
        if self.use_rgb:
            B, T = rgb.shape[:2]
            x = rgb.reshape(B * T, *rgb.shape[2:]).float() / 255.0
            x = (x - self.rgb_mean) / self.rgb_std
            feats.append(self.rgb_encoder(x).reshape(B, T, -1))
        x = torch.cat(feats, dim=-1)
        if self.temporal == "lstm":
            _, (h, _) = self.lstm(x)
            z = h[-1]
        else:
            z = x.mean(dim=1)
        return self.classifier(self.dropout(z))
