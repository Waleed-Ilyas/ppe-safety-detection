"""Grade the fine-tuned detectors once on the untouched test split.

    python -m ppedet.run_eval fast accurate

The score threshold is tuned on validation images that were NOT used for early stopping, then applied to the
test split. Writes artifacts/results.json and data/checkpoints/preds_<kind>.npz (for the figures)."""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from . import config, data
from . import evaluate as E
from . import model as M

SWEEP = [(256, 448), (320, 576), (416, 740), (512, 900)]  # (short, long) test-time input sizes


def auc_scores(y: np.ndarray, s: np.ndarray) -> dict:
    """ROC-AUC (rank statistic) and PR-AUC (average precision) for an image-level detector."""
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    ap = float((prec * y).sum() / max(1, y.sum()))
    ranks = np.argsort(np.argsort(s)) + 1
    n1, n0 = int(y.sum()), int((~y).sum())
    roc = float((ranks[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))
    return {"roc_auc": roc, "pr_auc": ap}


def summarise_operating_point(preds, recs, thr: float) -> dict:
    c = E.match_counts(preds, recs, thr)
    p, r, f = E.prf(c["tp"], c["fp"], c["fn"])
    n_gt = c["tp"] + c["fn"]
    per = {cls: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "n_gt": int(n_gt[i]),
                 "tp": int(c["tp"][i]), "fp": int(c["fp"][i])}
           for i, cls in enumerate(config.CLASSES) if n_gt[i] + c["fp"][i] > 0}
    mp, mr, mf = E.prf(c["tp"].sum(), c["fp"].sum(), c["fn"].sum())
    y, s = E.violation_scores(preds, recs)
    flagged = s >= thr
    vp = float((flagged & y).sum() / max(1, flagged.sum()))
    vr = float((flagged & y).sum() / max(1, y.sum()))
    return {"threshold": thr, "micro": {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
            "per_class": per,
            "violation_images": {"n_images": int(len(y)), "n_with_violation": int(y.sum()), "precision": vp,
                                 "recall": vr, "f1": 2 * vp * vr / max(vp + vr, 1e-9), **auc_scores(y, s)}}


def main():
    kinds = sys.argv[1:] or list(config.MODELS)
    torch.set_num_threads(6)
    splits = data.make_splits()
    n_files = sum(len(data.load_split(x)) for x in ("train", "valid", "test"))
    es = {r.image_id for r in data.balanced_subset(splits["val"], config.N_VAL, seed=1)}  # early-stopping images
    tune = [r for r in splits["val"] if r.image_id not in es]  # threshold tuning: validation images not used above
    test = splits["test"]
    counts = lambda recs: dict(zip(config.CLASSES, data.class_counts(recs).tolist(), strict=True))  # noqa: E731

    res = {"dataset": {"files_in_source": n_files, "distinct_images": sum(len(v) for v in splits.values()),
                       "split_images": {k: len(v) for k, v in splits.items()},
                       "train_subset": config.N_TRAIN, "early_stopping_images": len(es),
                       "threshold_tuning_images": len(tune), "test_images": len(test),
                       "class_counts": {k: counts(v) for k, v in splits.items()},
                       "classes": config.CLASSES}, "models": {}}
    for kind in kinds:
        _, short, long = config.MODELS[kind]
        model = M.load(kind)
        t0 = time.time()
        tune_preds = E.predict(model, tune, short, long)
        thr = E.tune_threshold(tune_preds, tune)
        print(kind, "tuned threshold", thr, flush=True)
        test_preds = E.predict(model, test, short, long)
        print(kind, "test inference", round(time.time() - t0), "s", flush=True)
        metrics = E.coco_metrics(test_preds, test)
        op = summarise_operating_point(test_preds, test, thr["threshold"])
        cm = E.confusion(test_preds, test, thr["threshold"])
        lat = E.latency_ms(model, test, short, long)
        size = (config.ARTIFACTS / f"{kind}.pt").stat().st_size / 1e6
        hist = json.loads((config.ARTIFACTS / f"{kind}_history.json").read_text(encoding="utf-8"))
        res["models"][kind] = {"coco": metrics, "operating_point": op, "threshold_tuning": thr,
                               "confusion": cm.tolist(), "latency": lat, "weights_mb": size,
                               "input": {"short": short, "long": long}, "history": hist,
                               "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
        print(kind, {k: round(v, 3) for k, v in metrics.items() if isinstance(v, float)}, "F1",
              round(op["micro"]["f1"], 3), lat, flush=True)
        E.save_preds(test_preds, config.CKPT / f"preds_{kind}.npz")
        # test-time input-size sweep with the same weights (no retraining): resolution vs accuracy vs speed
        sweep = []
        for sh, lg in SWEEP:
            model.transform.min_size, model.transform.max_size = (sh,), lg
            pr = test_preds if (sh, lg) == (short, long) else E.predict(model, test, sh, lg)
            cm_ = E.coco_metrics(pr, test)
            o = summarise_operating_point(pr, test, thr["threshold"])
            sweep.append({"short": sh, "long": lg, "map": cm_["map"], "ap50": cm_["ap50"],
                          "reliable_map": cm_["reliable_map"], "f1": o["micro"]["f1"],
                          "latency": E.latency_ms(model, test, sh, lg)})
            print(kind, "size", sh, {k: round(v, 3) for k, v in sweep[-1].items() if isinstance(v, float)}, flush=True)
        res["models"][kind]["size_sweep"] = sweep
    (config.ARTIFACTS / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
