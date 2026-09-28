"""Fine-tune a COCO-pretrained Faster R-CNN on the PPE data (CPU).

    python -m ppedet.train fast --epochs 7
    python -m ppedet.train accurate --epochs 6

Resumable: a checkpoint is written after every epoch; re-run the same command to continue."""

from __future__ import annotations

import argparse
import json
import math
import time

import mlflow
import torch

from . import config, data
from . import model as M
from .evaluate import coco_metrics, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(config.MODELS))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--n-train", type=int, default=config.N_TRAIN)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    torch.set_num_threads(4)
    config.CKPT.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS.mkdir(exist_ok=True)
    _, short, long = config.MODELS[args.kind]

    splits = data.make_splits()
    train = data.balanced_subset(splits["train"], args.n_train)
    val = data.balanced_subset(splits["val"], config.N_VAL, seed=1)  # early stopping / model selection
    print(f"train {len(train)} imgs {dict(zip(config.CLASSES, data.class_counts(train).tolist(), strict=True))}", flush=True)

    dl = torch.utils.data.DataLoader(data.DetDataset(train, short, long, train=True), batch_size=args.batch,
                                     shuffle=True, num_workers=2, collate_fn=data.collate, drop_last=True)
    model = M.build(args.kind)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(dl)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, total) / total)))

    last, start_epoch, history, best = config.CKPT / f"{args.kind}_last.pt", 0, [], -1.0
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, history, best = ck["epoch"] + 1, ck["history"], ck["best"]
        print(f"resumed after epoch {ck['epoch'] + 1}", flush=True)

    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("ppedet")
    with mlflow.start_run(run_name=f"train-{args.kind}"):
        mlflow.log_params({"kind": args.kind, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                           "n_train": len(train), "short": short, "long": long})
        for epoch in range(start_epoch, args.epochs):
            model.train()
            t0, run = time.time(), 0.0
            for it, (imgs, tgts) in enumerate(dl):
                tg = [{"boxes": t["boxes"], "labels": t["labels"]} for t in tgts]
                losses = model(list(imgs), tg)
                loss = sum(losses.values())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                sched.step()
                run += float(loss)
                if it % 50 == 0:
                    print(f"ep {epoch + 1} it {it}/{len(dl)} loss {run / (it + 1):.3f} "
                          f"{(time.time() - t0) / (it + 1):.2f}s/it", flush=True)
            train_loss = run / len(dl)
            model.eval()
            val_metrics = coco_metrics(predict(model, val, short, long), val)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_map": val_metrics["map"],
                            "val_ap50": val_metrics["ap50"], "minutes": (time.time() - t0) / 60})
            print(json.dumps(history[-1]), flush=True)
            mlflow.log_metrics({"train_loss": train_loss, "val_map": val_metrics["map"],
                                "val_ap50": val_metrics["ap50"]}, step=epoch + 1)
            if val_metrics["ap50"] > best:
                best = val_metrics["ap50"]
                M.save_fp16(model, config.ARTIFACTS / f"{args.kind}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": epoch, "history": history, "best": best}, last)
        (config.ARTIFACTS / f"{args.kind}_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print("done, best val AP50", best, flush=True)


if __name__ == "__main__":
    main()
