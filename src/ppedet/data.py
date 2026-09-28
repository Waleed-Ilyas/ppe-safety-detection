"""Roboflow "PPEs" dataset (CC BY 4.0) via the Hugging Face mirror: download, parse COCO annotations,
class-balanced subsampling, and a torch Dataset that resizes to the model's input scale."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance

from . import config


def download(split: str) -> None:
    """Fetch + unzip one split into data/raw/<split> (no-op if already extracted)."""
    out = config.DATA / split
    if (out / "_annotations.coco.json").exists():
        return
    config.DATA.mkdir(parents=True, exist_ok=True)
    zpath = config.DATA / f"{split}.zip"
    if not zpath.exists():
        urllib.request.urlretrieve(f"{config.HF_BASE}/{split}.zip", zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)


@dataclass
class Record:
    image_id: int
    path: str
    width: int
    height: int
    boxes: np.ndarray  # (n, 4) xyxy, original pixels
    labels: np.ndarray  # (n,) 1..10


def load_split(split: str) -> list[Record]:
    download(split)
    root = config.DATA / split
    d = json.loads((root / "_annotations.coco.json").read_text(encoding="utf-8"))
    by_img: dict[int, list] = {}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        if w >= 2 and h >= 2:  # drop degenerate boxes
            by_img.setdefault(a["image_id"], []).append((x, y, x + w, y + h, a["category_id"] + 1))
    recs = []
    for im in sorted(d["images"], key=lambda i: i["file_name"]):
        rows = np.array(by_img.get(im["id"], []), dtype=np.float32).reshape(-1, 5)
        recs.append(Record(im["id"], str(root / im["file_name"]), im["width"], im["height"],
                           rows[:, :4], rows[:, 4].astype(np.int64)))
    return recs


def source_id(rec: Record) -> str:
    """Roboflow file names are '<source>_jpg.rf.<hash>.jpg'; the source part identifies the original photo."""
    return rec.path.replace("\\", "/").rsplit("/", 1)[-1].split("_jpg.rf.")[0]


def _dhash(path: str) -> int:
    """64-bit difference hash: robust to re-encoding and resizing, so near-duplicate photos collide."""
    with Image.open(path) as im:
        im.draft("L", (72, 72))
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate_groups(recs: list[Record], max_dist: int = 4) -> np.ndarray:
    """Group id per image: images whose 64-bit dHashes differ in <= max_dist bits share a group (union-find).
    Cached in data/raw/dhash.json."""
    cache = config.DATA / "dhash.json"
    hashes = json.loads(cache.read_text()) if cache.exists() else {}
    for r in recs:
        if r.path not in hashes:
            hashes[r.path] = str(_dhash(r.path))
    cache.write_text(json.dumps(hashes))
    h = np.array([int(hashes[r.path]) for r in recs], dtype=np.uint64)
    parent = np.arange(len(h))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(h)):
        d = np.bitwise_count(h[i + 1:] ^ h[i]) if hasattr(np, "bitwise_count") else np.array(
            [bin(int(x)).count("1") for x in h[i + 1:] ^ h[i]])
        for j in np.flatnonzero(d <= max_dist):
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a
    return np.array([find(i) for i in range(len(h))])


def make_splits(fractions=(0.7, 0.1, 0.2), seed: int = config.SEED) -> dict[str, list[Record]]:
    """Pool the three official splits, drop duplicate photos, and re-split by near-duplicate group.

    The official splits are not usable as they are: their class mix differs completely (train has zero helmet /
    mask / no_helmet / no_mask boxes, test is 84% helmet + no_helmet), and the pooled 11,978 files contain only
    ~6.3k distinct images by perceptual hash. So: (1) keep one image per identical-hash group, (2) group the
    rest by near-identical hash (<= 1 bit) so look-alike frames stay on one side, (3) assign whole groups,
    largest first, to whichever split is furthest below its target share."""
    recs = sorted((r for s in ("train", "valid", "test") for r in load_split(s)), key=lambda r: r.path)
    g0 = near_duplicate_groups(recs, 0)
    _, first = np.unique(g0, return_index=True)
    uniq = [recs[i] for i in sorted(first)]
    for i, r in enumerate(uniq):
        r.image_id = i  # ids from the three official files collide; make them unique
    g1 = near_duplicate_groups(uniq, 1)
    rng = np.random.default_rng(seed)
    ids, sizes = np.unique(g1, return_counts=True)
    order = np.lexsort((rng.random(len(ids)), -sizes))  # largest first, ties shuffled
    target = np.array(fractions) * len(uniq)
    filled = np.zeros(3)
    where = {}
    for k in order:
        s = int(np.argmax((target - filled) / target))
        where[ids[k]] = s
        filled[s] += sizes[k]
    out: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
    for r, g in zip(uniq, g1, strict=True):
        out[("train", "val", "test")[where[g]]].append(r)
    return out


def class_counts(recs: list[Record]) -> np.ndarray:
    c = np.zeros(len(config.CLASSES), dtype=int)
    for r in recs:
        c += np.bincount(r.labels - 1, minlength=len(config.CLASSES))
    return c


def balanced_subset(recs: list[Record], n: int, seed: int = config.SEED) -> list[Record]:
    """Sample n images without replacement, weighting each image by the rarest class it contains
    (1/sqrt(class frequency)), so rare classes (gloves, no_shoes...) survive the down-sampling.
    Images with no annotations get the lowest weight."""
    freq = class_counts(recs).astype(float)
    freq = np.maximum(freq, 1) / freq.sum()
    w = np.array([(1 / np.sqrt(freq[r.labels - 1])).max() if len(r.labels) else 0.2 for r in recs])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(recs), size=min(n, len(recs)), replace=False, p=w / w.sum())
    return [recs[i] for i in sorted(idx)]


def scale_for(w: int, h: int, short: int, long: int) -> float:
    """Resize factor so the short side is `short` unless that pushes the long side over `long`."""
    s = short / min(w, h)
    return min(s, long / max(w, h))


class DetDataset(torch.utils.data.Dataset):
    """Images resized to the model's input scale (done here, in parallel workers, so the model's own
    transform is a no-op). `train=True` adds a horizontal flip and mild colour jitter."""

    def __init__(self, recs: list[Record], short: int, long: int, train: bool = False):
        self.recs, self.short, self.long, self.train = recs, short, long, train
        self.rng = np.random.default_rng(config.SEED)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(r.path).convert("RGB")
        s = scale_for(r.width, r.height, self.short, self.long)
        img = img.resize((round(r.width * s), round(r.height * s)), Image.BILINEAR)
        boxes = r.boxes * s
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                boxes = boxes[:, [2, 1, 0, 3]].copy() * np.array([-1, 1, -1, 1]) + np.array([img.width, 0, img.width, 0])
            for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                img = enh(img).enhance(rng.uniform(0.8, 1.2))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        target = {"boxes": torch.from_numpy(boxes.astype(np.float32)).reshape(-1, 4),
                  "labels": torch.from_numpy(r.labels), "image_id": r.image_id, "scale": s}
        return x, target


def collate(batch):
    return tuple(zip(*batch, strict=True))
