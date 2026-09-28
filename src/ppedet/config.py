"""Paths, classes and model settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
CKPT = ROOT / "data" / "checkpoints"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
HF_BASE = "https://huggingface.co/datasets/keremberke/protective-equipment-detection/resolve/main/data"

SEED = 42
# COCO category id + 1 = label id (0 is background in torchvision detectors)
CLASSES = ["glove", "goggles", "helmet", "mask", "no_glove", "no_goggles", "no_helmet", "no_mask",
           "no_shoes", "shoes"]
VIOLATIONS = {"no_glove", "no_goggles", "no_helmet", "no_mask", "no_shoes"}
NUM_LABELS = len(CLASSES) + 1

# name -> (torchvision builder, short side, long side). Images are resized to this before the model.
MODELS = {
    "fast": ("fasterrcnn_mobilenet_v3_large_320_fpn", 320, 576),
    "accurate": ("fasterrcnn_mobilenet_v3_large_fpn", 416, 740),
}
N_TRAIN, N_VAL = 1600, 300  # compute-bounded subsets of the train / valid splits (test is used in full)
