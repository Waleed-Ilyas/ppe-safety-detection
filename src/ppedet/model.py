"""Faster R-CNN with a MobileNetV3 backbone (torchvision), COCO-pretrained, re-headed for the PPE classes."""

from __future__ import annotations

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import config


def build(kind: str, pretrained: bool = True, score_thresh: float = 0.001):
    builder, short, long = config.MODELS[kind]
    m = getattr(torchvision.models.detection, builder)(
        weights="DEFAULT" if pretrained else None, weights_backbone=None, min_size=short, max_size=long,
        box_score_thresh=score_thresh, box_detections_per_img=100)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,
                                                  config.NUM_LABELS)
    return m


def save_fp16(model, path) -> None:
    """Half-precision weights on disk (halves the file; loaded back to float32)."""
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, path)


def load(kind: str, path=None, score_thresh: float = 0.001):
    m = build(kind, pretrained=False, score_thresh=score_thresh)
    sd = torch.load(path or config.ARTIFACTS / f"{kind}.pt", map_location="cpu", weights_only=True)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return m.eval()
