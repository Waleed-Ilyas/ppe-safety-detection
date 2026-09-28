"""README figures from artifacts/results.json + saved test predictions. Run: python -m ppedet.report"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from . import config, data  # noqa: E402
from .evaluate import MIN_GT, load_preds  # noqa: E402

GOLD, CYAN, IVORY, BG, GREY, RED, GREEN = "#D9B26A", "#4FD1C5", "#F2EDE4", "#0A0A0C", "#8a8578", "#E0625A", "#5fcf80"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": IVORY,
                     "axes.labelcolor": IVORY, "xtick.color": IVORY, "ytick.color": IVORY,
                     "axes.edgecolor": GREY, "font.size": 11})
KIND_COLOR = {"fast": CYAN, "accurate": GOLD}
KIND_LABEL = {"fast": "Fast (320 px)", "accurate": "Accurate (416 px)"}


def fig_data_audit(res) -> None:
    official = {s: data.class_counts(data.load_split(s)) for s in ("train", "valid", "test")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(config.CLASSES))
    for j, (s, c) in enumerate(official.items()):
        a.bar(x + (j - 1) * 0.27, c / max(1, c.sum()) * 100, 0.27, color=[GREY, CYAN, GOLD][j], label=f"official {s}")
    a.set_xticks(x, config.CLASSES, rotation=35, ha="right")
    a.set_ylabel("share of that split's boxes (%)")
    a.set_title("The official splits have different class mixes\n(train: zero helmet / mask / no_helmet / no_mask boxes)",
                fontsize=10)
    a.legend(frameon=False, fontsize=9)
    d = res["dataset"]
    b.bar(["files in\nthe 3 splits", "distinct\nimages"], [d["files_in_source"], d["distinct_images"]], color=[GREY, GOLD])
    for i, v in enumerate([d["files_in_source"], d["distinct_images"]]):
        b.text(i, v + 150, f"{v:,}", ha="center")
    b.set_title("Duplicate photos (perceptual hash)", fontsize=10)
    b.set_ylim(0, d["files_in_source"] * 1.12)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "01_data_audit.png", dpi=150)


def fig_per_class(res) -> None:
    models = res["models"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(config.CLASSES))
    for j, (k, m) in enumerate(models.items()):
        v = [m["coco"]["per_class"][c]["ap50"] for c in config.CLASSES]
        ax.bar(x + (j - 0.5) * 0.38, np.nan_to_num(v), 0.38, color=KIND_COLOR[k], label=KIND_LABEL[k])
    n_gt = [next(iter(models.values()))["coco"]["per_class"][c]["n_gt"] for c in config.CLASSES]
    ax.set_xticks(x, [f"{c}\n(n={n})" for c, n in zip(config.CLASSES, n_gt, strict=True)], fontsize=9)
    for i, n in enumerate(n_gt):
        if n < MIN_GT:
            ax.axvspan(i - 0.5, i + 0.5, color=GREY, alpha=0.18, lw=0)
    ax.set_ylabel("AP at IoU 0.5 (test set)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class accuracy; grey = fewer than {MIN_GT} test boxes (too few to trust)", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "02_per_class_ap.png", dpi=150)


def fig_size_sweep(res) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for k, col in KIND_COLOR.items():
        sw = res["models"][k]["size_sweep"]
        native = res["models"][k]["input"]["short"]
        ax.plot([s["short"] for s in sw], [s["map"] for s in sw], marker="o", color=col, lw=2, label=KIND_LABEL[k])
        ax.axvline(native, color=col, ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("test-time input short side (px)")
    ax.set_ylabel("COCO mAP@[.5:.95]")
    ax.set_title("Both models peak at the resolution they were trained on", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "05b_size_sweep.png", dpi=150)


def fig_confusion(res, kind="accurate") -> None:
    m = res["models"][kind]
    cm = np.array(m["confusion"], dtype=float)
    labels = [*config.CLASSES, "missed / false alarm"]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.6, 7))
    ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="black" if norm[i, j] > 0.55 else IVORY)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), [*config.CLASSES, "background"], fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth (rows sum to 100%)")
    ax.set_title(f"{KIND_LABEL[kind]} at threshold {m['operating_point']['threshold']:.2f}: "
                 "what gets confused with what", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "03_confusion.png", dpi=150)


def fig_training(res) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for k, m in res["models"].items():
        h = m["history"]
        e = [r["epoch"] for r in h]
        a.plot(e, [r["train_loss"] for r in h], marker="o", color=KIND_COLOR[k], label=KIND_LABEL[k])
        b.plot(e, [r["val_ap50"] for r in h], marker="o", color=KIND_COLOR[k], label=KIND_LABEL[k])
    a.set_title("Training loss", fontsize=10)
    b.set_title("Validation AP50 (300 images)", fontsize=10)
    for ax in (a, b):
        ax.set_xlabel("epoch")
    b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "04_training.png", dpi=150)


def fig_speed(res) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, m in res["models"].items():
        ax.scatter(m["latency"]["median_ms"], m["coco"]["map"], s=220, color=KIND_COLOR[k], zorder=3)
        ax.annotate(f"{KIND_LABEL[k]}\n{m['latency']['median_ms']:.0f} ms, mAP {m['coco']['map']:.3f}",
                    (m["latency"]["median_ms"], m["coco"]["map"]), textcoords="offset points", xytext=(10, -4), fontsize=9)
    ax.set_xlabel("CPU latency per image (ms, median, 4 threads, i7-8650U)")
    ax.set_ylabel("COCO mAP@[.5:.95] on the test set")
    ax.set_title("Speed vs accuracy", fontsize=11)
    lats = [m["latency"]["median_ms"] for m in res["models"].values()]
    maps = [m["coco"]["map"] for m in res["models"].values()]
    xpad, ypad = (max(lats) - min(lats)) * 0.6 + 15, (max(maps) - min(maps)) * 0.6 + 0.01
    ax.set_xlim(min(lats) - xpad * 0.3, max(lats) + xpad)
    ax.set_ylim(min(maps) - ypad, max(maps) + ypad)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "05_speed_accuracy.png", dpi=150)


def _draw(ax, rec, boxes, labels, scores=None, color=GREEN):
    img = Image.open(rec.path).convert("RGB")
    ax.imshow(img)
    for i, b in enumerate(boxes):
        name = config.CLASSES[int(labels[i]) - 1]
        c = RED if name in config.VIOLATIONS else color
        ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=2))
        txt = name if scores is None else f"{name} {scores[i]:.2f}"
        ax.text(b[0], max(b[1] - 3, 0), txt, color="black", fontsize=7, bbox=dict(fc=c, ec="none", pad=1))
    ax.axis("off")


def fig_examples(res, kind="accurate", n=6) -> list[int]:
    splits = data.make_splits()
    test = splits["test"]
    preds = load_preds(config.CKPT / f"preds_{kind}.npz")
    thr = res["models"][kind]["operating_point"]["threshold"]
    rng = np.random.default_rng(3)
    with_boxes = [i for i, r in enumerate(test) if len(r.labels) >= 2]
    pick = sorted(rng.choice(with_boxes, n, replace=False))  # random, not cherry-picked
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.4))
    for c, i in enumerate(pick):
        r, p = test[i], preds[i]
        keep = p["scores"] >= thr
        _draw(axes[0, c], r, r.boxes, r.labels)
        _draw(axes[1, c], r, p["boxes"][keep], p["labels"][keep], p["scores"][keep])
    axes[0, 0].set_title("ground truth", loc="left", fontsize=10)
    axes[1, 0].set_title(f"{KIND_LABEL[kind]} predictions (score >= {thr:.2f})", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "06_examples.png", dpi=110)
    return [int(i) for i in pick]


if __name__ == "__main__":
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    r = json.loads((config.ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    fig_data_audit(r)
    fig_per_class(r)
    fig_confusion(r)
    fig_training(r)
    fig_speed(r)
    fig_size_sweep(r)
    print("random example picks:", fig_examples(r))
