"""Step 6 (source D): facial landmarks, region masks, geometric measures and action-unit-like blendshapes. CPU.

Uses the MediaPipe Face Landmarker (478 points, 52 blendshapes). The 100 x 100 RAF-DB crops are enlarged to 256 x 256
before detection, which helps on small faces. Outputs in --out:
  features/landmarks.npz      image_id, landmarks (N, 478, 2) in 100-pixel coordinates (NaN when not found), found
  results/D_geometry.csv      image_id, split, label, found, the 8 geometric measures, 52 blendshape scores
  results/D_by_class.csv      mean of each measure per class on the training split (sanity check)
  figures/region_masks.png    the seven regions drawn on 8 faces (check by eye)
Usage:
  python scripts/06_landmarks.py --out /kaggle/working/outputs
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from xfer.io import load_manifest
from xfer.labels import CLASSES
from xfer.regions import GEOMETRY, REGIONS, geometry, region_map

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
             "face_landmarker.task")


def make_detector(model_path):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                        output_face_blendshapes=True, num_faces=1,
                                        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
    det = vision.FaceLandmarker.create_from_options(opts)

    def run(path):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        big = np.asarray(im.resize((256, 256), Image.BICUBIC))
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(big)))
        if not res.face_landmarks:
            return None, None
        lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        return lm, bs
    return run


def draw_regions(m, lms, path, n=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(m), min(n, len(m)), replace=False)
    fig, axes = plt.subplots(2, len(idx), figsize=(len(idx) * 1.3, 2.8))
    for j, i in enumerate(idx):
        im = np.asarray(Image.open(m.path.iloc[i]).convert("RGB").resize((100, 100)))
        lm = lms[i] if not np.isnan(lms[i]).any() else None
        lab = region_map(lm)
        axes[0, j].imshow(im); axes[0, j].axis("off")
        axes[1, j].imshow(im, alpha=0.6)
        shown = np.ma.masked_where(lab < 0, lab)
        axes[1, j].imshow(shown, cmap="Greys", alpha=0.6, vmin=-1, vmax=len(REGIONS))
        axes[1, j].axis("off")
        axes[1, j].set_title("mesh" if lm is not None else "template", fontsize=6)
    plt.suptitle("Regions: " + ", ".join(REGIONS), fontsize=7)
    plt.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="path to face_landmarker.task (downloaded if missing)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    m = load_manifest(a.out)
    if a.limit:
        m = m.groupby("split", group_keys=False).head(a.limit).reset_index(drop=True)
    for d in ("features", "results", "figures"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    model = a.model or os.path.join(a.out, "models", "face_landmarker.task")
    if not os.path.exists(model):
        os.makedirs(os.path.dirname(model), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model)
    run = make_detector(model)

    lms = np.full((len(m), 478, 2), np.nan, np.float32)
    rows = []
    for i, p in enumerate(m.path):
        lm, bs = run(p)
        r = dict(image_id=m.image_id.iloc[i], split=m.split.iloc[i], label=int(m.label.iloc[i]), found=lm is not None)
        if lm is not None:
            lms[i] = lm
            r.update(geometry(lm))
            r.update(bs)
        rows.append(r)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(m)}  found so far: {np.mean([x['found'] for x in rows]):.3f}")
    g = pd.DataFrame(rows)
    np.savez_compressed(os.path.join(a.out, "features", "landmarks.npz"), image_id=m.image_id.astype(str).values,
                        landmarks=lms, found=g.found.values)
    g.to_csv(os.path.join(a.out, "results", "D_geometry.csv"), index=False)
    found = g.groupby("split").found.mean()
    print("face found, by split:", found.round(4).to_dict())
    tr = g[(g.split == "train") & g.found]
    by = tr.groupby("label")[GEOMETRY].mean()
    by.index = [CLASSES[i] for i in by.index]
    by.round(3).to_csv(os.path.join(a.out, "results", "D_by_class.csv"))
    print(by.round(3).to_string())
    print("Expect: happiness highest lip_corner_lift; surprise highest mouth_opening, eye_opening and brow raise.")
    draw_regions(m, lms, os.path.join(a.out, "figures", "region_masks.png"))


if __name__ == "__main__":
    main()
