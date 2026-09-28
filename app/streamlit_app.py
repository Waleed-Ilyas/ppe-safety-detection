"""PPE safety-compliance detector: upload a site photo (or pick a sample) and see what gear is worn or missing."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppedet import config  # noqa: E402
from ppedet import model as M  # noqa: E402
from ppedet.data import scale_for  # noqa: E402

st.set_page_config(page_title="PPE compliance detector", page_icon="🦺", layout="wide")
GOLD, CYAN, RED, GREEN, GREY = "#D9B26A", "#4FD1C5", "#E0625A", "#5fcf80", "#8a8578"
SAMPLES = Path(__file__).resolve().parents[1] / "assets" / "samples"
KIND_LABEL = {"accurate": "Accurate (416 px)", "fast": "Fast (320 px)"}

torch.set_num_threads(2)


@st.cache_resource(show_spinner="Loading the detector...")
def load_model(kind: str):
    return M.load(kind, score_thresh=0.05)


@st.cache_data
def load_results():
    return json.loads((config.ARTIFACTS / "results.json").read_text(encoding="utf-8"))


def detect(img: Image.Image, kind: str):
    _, short, long = config.MODELS[kind]
    s = scale_for(img.width, img.height, short, long)
    small = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)
    t = time.perf_counter()
    with torch.no_grad():
        out = load_model(kind)([x])[0]
    ms = (time.perf_counter() - t) * 1000
    return (out["boxes"] / s).numpy(), out["scores"].numpy(), out["labels"].numpy(), ms


def draw(img: Image.Image, boxes, labels, scores=None) -> Image.Image:
    im = img.copy()
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 350)
    font = ImageFont.load_default(size=max(12, im.width // 60))
    for i, b in enumerate(boxes):
        name = config.CLASSES[int(labels[i]) - 1]
        col = "#E0625A" if name in config.VIOLATIONS else "#5fcf80"
        d.rectangle([*map(float, b)], outline=col, width=lw)
        txt = name if scores is None else f"{name} {scores[i]:.2f}"
        tw = d.textlength(txt, font=font)
        y0 = max(float(b[1]) - font.size - 4, 0)
        d.rectangle([float(b[0]), y0, float(b[0]) + tw + 6, y0 + font.size + 4], fill=col)
        d.text((float(b[0]) + 3, y0 + 1), txt, fill="black", font=font)
    return im


res = load_results()
tuned = res["models"]["accurate"]["operating_point"]["threshold"]

with st.sidebar:
    st.header("🦺 PPE compliance")
    st.caption("Object detector fine-tuned on construction / industrial photos to find safety gear and "
               "the *absence* of it.")
    kind = st.radio("Model", list(KIND_LABEL), format_func=KIND_LABEL.get,
                    help="Both are Faster R-CNN with a MobileNetV3 backbone; they differ in input resolution.")
    thr = st.slider("Confidence threshold", 0.05, 0.95, float(tuned), 0.05,
                    help="Detections below this score are hidden. Lower = catch more, more false alarms. The default "
                         "was tuned on validation images.")
    st.divider()
    with st.expander("How to read this"):
        st.markdown(
            "- **Green boxes** are gear that is present (helmet, glove, goggles, mask, shoes); **red boxes** are "
            "*missing* gear (no_helmet, no_glove, ...).\n"
            "- The detector was trained on ~1,600 photos, mostly 1280x720 site-camera frames. It works best on that "
            "kind of scene and is weak on masks and shoes (few examples).\n"
            "- A screening aid, **not** a safety system: it misses things and raises false alarms. See the results tab.")

st.title("PPE compliance detector")
st.caption("Finds safety helmets, gloves, goggles, masks and shoes, and flags when they are missing.")

tab_try, tab_res = st.tabs(["Try it", "How well does it work?"])

with tab_try:
    samples = sorted(SAMPLES.glob("*.jpg"))
    src = st.radio("Image", ["Sample photo", "Upload my own"], horizontal=True)
    img, gt = None, None
    if src == "Sample photo" and samples:
        pick = st.selectbox("Sample", samples, format_func=lambda p: p.stem.replace("_", " "))
        img = Image.open(pick).convert("RGB")
        gt_file = pick.with_suffix(".json")
        gt = json.loads(gt_file.read_text()) if gt_file.exists() else None
    elif src == "Upload my own":
        up = st.file_uploader("A photo of a worker or site (JPEG / PNG)", type=["jpg", "jpeg", "png"])
        if up:
            img = Image.open(up).convert("RGB")
            img.thumbnail((1600, 1600))
    if img is None:
        st.info("Choose a sample photo or upload one.")
        st.stop()

    boxes, scores, labels, ms = detect(img, kind)
    keep = scores >= thr
    b, s, lab = boxes[keep], scores[keep], labels[keep]
    names = [config.CLASSES[int(i) - 1] for i in lab]
    viol = [n for n in names if n in config.VIOLATIONS]

    cols = st.columns([3, 2])
    with cols[0]:
        st.image(draw(img, b, lab, s), caption=f"{KIND_LABEL[kind]} · {ms:.0f} ms on the server CPU · threshold {thr:.2f}",
                 use_container_width=True)
    with cols[1]:
        if viol:
            st.error(f"⚠ Possible violation: {', '.join(f'{n} x{viol.count(n)}' for n in sorted(set(viol)))}")
        elif names:
            st.success("✓ No missing-gear detections at this threshold")
        else:
            st.info("Nothing detected at this threshold. Try lowering it.")
        if names:
            counts = pd.Series(names).value_counts().rename_axis("class").reset_index(name="detections")
            counts["kind"] = np.where(counts["class"].isin(config.VIOLATIONS), "missing gear", "gear present")
            st.dataframe(counts, hide_index=True, use_container_width=True)
        if gt:
            with st.expander("Ground truth for this sample"):
                st.image(draw(img, [g["box"] for g in gt], [g["label"] for g in gt]), use_container_width=True)
        st.caption("Scores are the model's own confidence, not calibrated probabilities.")

with tab_res:
    st.markdown(f"**Test set: {res['dataset']['test_images']:,} images the model never saw** "
                f"(split made after removing duplicate photos; see the README).")
    rows = []
    for k, m in res["models"].items():
        rows.append({"Model": KIND_LABEL[k], "mAP@[.5:.95]": m["coco"]["map"], "AP50": m["coco"]["ap50"],
                     "mAP (6 well-covered classes)": m["coco"].get("reliable_map", float("nan")),
                     "Precision": m["operating_point"]["micro"]["precision"],
                     "Recall": m["operating_point"]["micro"]["recall"],
                     "Latency (ms)": m["latency"]["median_ms"], "Weights (MB)": m["weights_mb"]})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ("mAP@[.5:.95]", "AP50", "mAP (6 well-covered classes)",
                                                                       "Precision", "Recall")} | {"Latency (ms)": "{:.0f}", "Weights (MB)": "{:.0f}"}),
                 hide_index=True, use_container_width=True)
    m = res["models"][kind]
    pc = m["coco"]["per_class"]
    fig = go.Figure(go.Bar(x=[f"{c} (n={pc[c]['n_gt']})" for c in config.CLASSES],
                           y=[0 if pc[c]["ap50"] != pc[c]["ap50"] else pc[c]["ap50"] for c in config.CLASSES],
                           marker_color=[GREY if pc[c]["n_gt"] < 30 else GOLD for c in config.CLASSES]))
    fig.update_layout(title=f"{KIND_LABEL[kind]}: AP50 by class (grey = fewer than 30 test boxes)", height=360,
                      yaxis_range=[0, 1], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#F2EDE4", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    v = m["operating_point"]["violation_images"]
    st.markdown(f"**Image-level 'any violation?' check** ({v['n_with_violation']} of {v['n_images']} test images contain "
                f"missing gear): precision {v['precision']:.0%}, recall {v['recall']:.0%}, PR-AUC {v['pr_auc']:.2f}.")
