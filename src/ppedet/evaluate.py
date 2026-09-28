"""Inference, COCO metrics and the operating-point (threshold) analysis."""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from . import config
from .data import DetDataset, Record, collate

MIN_GT = 30  # a class needs at least this many test boxes for its AP to enter the "reliable" mean


@torch.no_grad()
def predict(model, recs: list[Record], short: int, long: int, batch: int = 8, workers: int = 2):
    """Detections in ORIGINAL image pixels: list of dicts(boxes xyxy, scores, labels), aligned with recs."""
    ds = DetDataset(recs, short, long, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate)
    model.eval()
    out = []
    for imgs, tgts in dl:
        for det, t in zip(model(list(imgs)), tgts, strict=True):
            out.append({"boxes": (det["boxes"] / t["scale"]).numpy(), "scores": det["scores"].numpy(),
                        "labels": det["labels"].numpy()})
    return out


def save_preds(preds: list[dict], path) -> None:
    off = np.cumsum([0] + [len(p["scores"]) for p in preds])
    np.savez_compressed(path, offsets=off, boxes=np.concatenate([p["boxes"] for p in preds]),
                        scores=np.concatenate([p["scores"] for p in preds]),
                        labels=np.concatenate([p["labels"] for p in preds]))


def load_preds(path) -> list[dict]:
    z = np.load(path)
    o = z["offsets"]
    return [{"boxes": z["boxes"][a:b], "scores": z["scores"][a:b], "labels": z["labels"][a:b]}
            for a, b in zip(o[:-1], o[1:], strict=True)]


def _coco_gt(recs: list[Record]) -> COCO:
    images = [{"id": i, "width": r.width, "height": r.height} for i, r in enumerate(recs)]
    anns = []
    for i, r in enumerate(recs):
        for b, lab in zip(r.boxes, r.labels, strict=True):
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(lab),
                         "bbox": [float(b[0]), float(b[1]), w, h], "area": w * h, "iscrowd": 0})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i + 1, "name": n} for i, n in enumerate(config.CLASSES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def coco_metrics(preds: list[dict], recs: list[Record]) -> dict:
    """COCO mAP (IoU .5:.95), AP50, AP75, size buckets and per-class AP50 / AP."""
    gt = _coco_gt(recs)
    rows = []
    for i, p in enumerate(preds):
        for b, s, lab in zip(p["boxes"], p["scores"], p["labels"], strict=True):
            rows.append({"image_id": i, "category_id": int(lab), "score": float(s),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    if not rows:
        return {"map": 0.0, "ap50": 0.0, "ap75": 0.0, "per_class": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(rows)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    prec = ev.eval["precision"]  # [T, R, K, A, M]

    def cls_ap(k, t=None):
        p = prec[:, :, k, 0, 2] if t is None else prec[t, :, k, 0, 2]
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")

    n_gt = np.bincount(np.concatenate([r.labels for r in recs]) - 1, minlength=len(config.CLASSES))
    per_class = {c: {"ap": cls_ap(k), "ap50": cls_ap(k, 0), "n_gt": int(n_gt[k])}
                 for k, c in enumerate(config.CLASSES)}
    solid = [v["ap"] for v in per_class.values() if v["n_gt"] >= MIN_GT]  # classes with enough test boxes to trust
    solid50 = [v["ap50"] for v in per_class.values() if v["n_gt"] >= MIN_GT]
    return {"map": float(s[0]), "ap50": float(s[1]), "ap75": float(s[2]), "ap_small": float(s[3]),
            "ap_medium": float(s[4]), "ap_large": float(s[5]),
            "reliable_map": float(np.mean(solid)) if solid else float("nan"),
            "reliable_ap50": float(np.mean(solid50)) if solid50 else float("nan"),
            "reliable_classes": [c for c, v in per_class.items() if v["n_gt"] >= MIN_GT],
            "per_class": per_class}


def _area(x: np.ndarray) -> np.ndarray:
    return (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    return inter / (_area(a)[:, None] + _area(b)[None, :] - inter + 1e-9)


def _greedy_match(iou_rows: np.ndarray, n_gt: int, iou: float):
    """Detections (rows, already sorted by score) claim their best unmatched ground truth. Returns
    (matched gt index or -1 per detection, used mask)."""
    used = np.zeros(n_gt, bool)
    hit = np.full(len(iou_rows), -1)
    for i, row in enumerate(iou_rows):
        if n_gt == 0:
            break
        j = int(np.argmax(np.where(used, -1, row)))
        if not used[j] and row[j] >= iou:
            used[j] = True
            hit[i] = j
    return hit, used


def match_counts(preds: list[dict], recs: list[Record], thr, iou: float = 0.5) -> dict:
    """Per-class TP/FP/FN at a score threshold (a float, or one threshold per class). A detection is a TP if it
    overlaps an unmatched same-class ground-truth box by IoU >= `iou` (greedy, highest score first)."""
    k = len(config.CLASSES)
    thr = np.full(k, thr) if np.isscalar(thr) else np.asarray(thr)
    tp, fp, fn = np.zeros(k, int), np.zeros(k, int), np.zeros(k, int)
    for p, r in zip(preds, recs, strict=True):
        keep = p["scores"] >= thr[p["labels"] - 1]
        boxes, scores, labels = p["boxes"][keep], p["scores"][keep], p["labels"][keep]
        for c in range(1, k + 1):
            g = r.boxes[r.labels == c]
            sel = labels == c
            d = boxes[sel][np.argsort(-scores[sel])]
            hit, used = _greedy_match(iou_matrix(d, g), len(g), iou)
            tp[c - 1] += int((hit >= 0).sum())
            fp[c - 1] += int((hit < 0).sum())
            fn[c - 1] += int((~used).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def prf(tp, fp, fn):
    p = tp / np.maximum(tp + fp, 1)
    r = tp / np.maximum(tp + fn, 1)
    return p, r, 2 * p * r / np.maximum(p + r, 1e-9)


def tune_threshold(preds, recs, grid=None) -> dict:
    """The single score threshold maximising micro-F1 (all classes pooled) on a validation split."""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if grid is None else grid
    best = (-1.0, 0.5)
    for t in grid:
        c = match_counts(preds, recs, t)
        _, _, f = prf(c["tp"].sum(), c["fp"].sum(), c["fn"].sum())
        best = max(best, (float(f), float(t)))
    return {"threshold": best[1], "val_micro_f1": best[0]}


def confusion(preds, recs, thr: float, iou: float = 0.5) -> np.ndarray:
    """Class-agnostic IoU matching, then compare labels. Rows = ground truth (last row: background, i.e. false
    alarms), columns = predicted (last column: missed)."""
    k = len(config.CLASSES)
    cm = np.zeros((k + 1, k + 1), int)
    for p, r in zip(preds, recs, strict=True):
        keep = p["scores"] >= thr
        boxes, scores, labels = p["boxes"][keep], p["scores"][keep], p["labels"][keep]
        order = np.argsort(-scores)
        boxes, labels = boxes[order], labels[order]
        hit, used = _greedy_match(iou_matrix(boxes, r.boxes), len(r.boxes), iou)
        for i, j in enumerate(hit):
            if j >= 0:
                cm[r.labels[j] - 1, labels[i] - 1] += 1
            else:
                cm[k, labels[i] - 1] += 1
        for j in np.flatnonzero(~used):
            cm[r.labels[j] - 1, k] += 1
    return cm


def violation_scores(preds, recs):
    """Image-level compliance view: does the photo contain any 'no_*' violation? Score = highest confidence
    of any violation detection."""
    viol = [config.CLASSES.index(c) + 1 for c in config.VIOLATIONS]
    y = np.array([bool(np.isin(r.labels, viol).any()) for r in recs])
    s = np.array([float(p["scores"][np.isin(p["labels"], viol)].max()) if np.isin(p["labels"], viol).any() else 0.0
                  for p in preds])
    return y, s


def latency_ms(model, recs: list[Record], short: int, long: int, n: int = 25, threads: int = 4) -> dict:
    """Single-image CPU latency (median and p90, ms) after a warm-up; image resize excluded."""
    torch.set_num_threads(threads)
    ds = DetDataset(recs[: n + 3], short, long)
    ts = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            x, _ = ds[i]
            t = time.perf_counter()
            model([x])
            ts.append((time.perf_counter() - t) * 1000)
    ts = ts[3:]
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "threads": threads}
