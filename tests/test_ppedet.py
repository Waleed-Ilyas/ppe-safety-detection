"""Offline tests on synthetic data: no dataset download, no trained weights."""

import numpy as np
import pytest
import torch
from PIL import Image

from ppedet import config, data
from ppedet import evaluate as E
from ppedet import model as M


def _rec(boxes, labels, path="x.jpg", w=200, h=100, image_id=0):
    return data.Record(image_id, path, w, h, np.array(boxes, dtype=np.float32).reshape(-1, 4),
                       np.array(labels, dtype=np.int64))


def _pred(boxes, scores, labels):
    return {"boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4), "scores": np.array(scores, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64)}


# ---------------------------------------------------------------- data
def test_scale_for_respects_short_and_long_side():
    assert data.scale_for(1600, 500, 416, 740) == pytest.approx(740 / 1600)  # long side would exceed the cap
    assert data.scale_for(1280, 720, 416, 740) == pytest.approx(416 / 720)
    assert data.scale_for(500, 333, 320, 576) == pytest.approx(320 / 333)


def test_dataset_resizes_image_and_boxes_together(tmp_path):
    Image.new("RGB", (200, 100), (10, 20, 30)).save(tmp_path / "a.jpg")
    r = _rec([[20, 10, 120, 60]], [3], path=str(tmp_path / "a.jpg"))
    x, t = data.DetDataset([r], short=50, long=400)[0]
    assert x.shape == (3, 50, 100)
    assert t["scale"] == pytest.approx(0.5)
    assert t["boxes"].tolist() == [[10.0, 5.0, 60.0, 30.0]]


def test_horizontal_flip_keeps_boxes_valid(tmp_path):
    Image.new("RGB", (200, 100)).save(tmp_path / "a.jpg")
    r = _rec([[20, 10, 120, 60]], [3], path=str(tmp_path / "a.jpg"))
    ds = data.DetDataset([r], short=100, long=400, train=True)
    for _ in range(20):
        _, t = ds[0]
        b = t["boxes"][0]
        assert b[0] < b[2] and b[1] < b[3] and 0 <= b[0] and b[2] <= 200
        assert (b[2] - b[0]) == pytest.approx(100.0, abs=1e-4)  # width unchanged by the flip


def test_balanced_subset_keeps_rare_classes_more_often():
    rng = np.random.default_rng(0)
    recs = [_rec([[0, 0, 10, 10]], [3], image_id=i) for i in range(900)]  # helmets
    recs += [_rec([[0, 0, 10, 10]], [1], image_id=900 + i) for i in range(100)]  # gloves (rare)
    picked = data.balanced_subset(recs, 200, seed=int(rng.integers(0, 5)))
    share = np.mean([r.labels[0] == 1 for r in picked])
    assert share > 0.1  # rarer than 10% in the pool, over-represented in the sample


# ---------------------------------------------------------------- metrics
def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert E.iou_matrix(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_match_counts_counts_tp_fp_fn_and_never_double_matches():
    gt = _rec([[0, 0, 10, 10], [50, 50, 60, 60]], [3, 3])
    p = _pred([[0, 0, 10, 10], [1, 1, 10, 10], [80, 80, 90, 90]], [0.9, 0.8, 0.7], [3, 3, 3])
    c = E.match_counts([p], [gt], 0.5)
    k = 2  # class 'helmet' is label 3 -> index 2
    assert (c["tp"][k], c["fp"][k], c["fn"][k]) == (1, 2, 1)  # 2nd box duplicates a matched GT -> FP


def test_threshold_removes_low_score_detections():
    gt = _rec([[0, 0, 10, 10]], [3])
    p = _pred([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.2], [3, 3])
    lo, hi = E.match_counts([p], [gt], 0.1), E.match_counts([p], [gt], 0.5)
    assert lo["fp"][2] == 1 and hi["fp"][2] == 0 and hi["tp"][2] == 1


def test_coco_map_is_one_for_perfect_and_zero_for_wrong_class():
    gt = [_rec([[10, 10, 60, 60]], [3]), _rec([[20, 20, 80, 90]], [7], image_id=1)]
    perfect = [_pred(r.boxes, [0.99], r.labels) for r in gt]
    wrong = [_pred(r.boxes, [0.99], [1]) for r in gt]
    assert E.coco_metrics(perfect, gt)["map"] == pytest.approx(1.0)
    assert E.coco_metrics(wrong, gt)["map"] == pytest.approx(0.0)


def test_confusion_matrix_shows_label_mix_ups_and_misses():
    gt = _rec([[0, 0, 10, 10], [50, 50, 60, 60]], [3, 7])  # helmet, no_helmet
    p = _pred([[0, 0, 10, 10], [80, 80, 90, 90]], [0.9, 0.8], [7, 3])  # helmet predicted as no_helmet; a false alarm
    cm = E.confusion([p], [gt], 0.5)
    k = len(config.CLASSES)
    assert cm[2, 6] == 1  # true helmet -> predicted no_helmet
    assert cm[k, 2] == 1  # false alarm 'helmet'
    assert cm[6, k] == 1  # the real no_helmet was missed


def test_violation_scores_flags_images_with_no_star_classes():
    a = _rec([[0, 0, 10, 10]], [7])  # no_helmet -> a violation
    b = _rec([[0, 0, 10, 10]], [3], image_id=1)
    preds = [_pred([[0, 0, 10, 10]], [0.8], [7]), _pred([[0, 0, 10, 10]], [0.9], [3])]
    y, s = E.violation_scores(preds, [a, b])
    assert y.tolist() == [True, False] and s.tolist() == pytest.approx([0.8, 0.0])


# ---------------------------------------------------------------- model
def test_model_forward_shapes_and_fp16_roundtrip(tmp_path):
    m = M.build("fast", pretrained=False)
    m.eval()
    with torch.no_grad():
        out = m([torch.rand(3, 160, 240)])[0]
    assert set(out) == {"boxes", "labels", "scores"} and out["boxes"].shape[1] == 4
    assert m.roi_heads.box_predictor.cls_score.out_features == config.NUM_LABELS
    p = tmp_path / "w.pt"
    M.save_fp16(m, p)
    m2 = M.load("fast", p)
    a, b = m.state_dict(), m2.state_dict()
    k = "roi_heads.box_predictor.cls_score.weight"
    assert torch.allclose(a[k], b[k], atol=1e-2)
