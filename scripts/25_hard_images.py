"""What is special about the test images POSTER++ gets wrong? CPU only, a few minutes.

Per test image: brightness, contrast, sharpness (variance of the Laplacian), whether MediaPipe found the face, and a
head-turn proxy (horizontal offset of the nose tip from the eye midpoint, relative to the eye distance). Compares
POSTER++'s errors with its correct answers, gives the error rate by quartile of each measure, and saves a contact
sheet of the error images (true label / POSTER++ answer / VLM answer) for visual inspection.
Outputs: results/hard_images.md, figures/poster_errors_sheet_<k>.png
Usage:  python scripts/25_hard_images.py --out OUT
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import laplace

from xfer.io import load_manifest, pred_path, read_predictions
from xfer.labels import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = load_manifest(a.out)
    m = m[m.split == "test"].copy()
    m["image_id"] = m.image_id.astype(str)
    B = read_predictions(pred_path(a.out, "B", "test")); B["image_id"] = B.image_id.astype(str)
    V = read_predictions(pred_path(a.out, "V2", "test")); V["image_id"] = V.image_id.astype(str)
    m = m.merge(B[["image_id", "pred"]], on="image_id").merge(V[["image_id", "pred"]].rename(columns={"pred": "vlm"}),
                                                               on="image_id")
    z = np.load(os.path.join(a.out, "features", "landmarks.npz"), allow_pickle=True)
    lms = {str(i): (lm if f else None) for i, lm, f in zip(z["image_id"], z["landmarks"], z["found"])}
    rows = []
    for r in m.itertuples(index=False):
        g = np.asarray(Image.open(r.path).convert("L"), float)
        lm = lms.get(r.image_id)
        yaw = np.nan
        if lm is not None:
            el, er, nose = lm[[33, 133]].mean(0), lm[[362, 263]].mean(0), lm[1]
            yaw = (nose[0] - (el[0] + er[0]) / 2) / (np.linalg.norm(er - el) + 1e-6)
        rows.append(dict(image_id=r.image_id, brightness=g.mean(), contrast=g.std(), sharpness=laplace(g).var(),
                         face_found=lm is not None, head_turn=abs(yaw)))
    q = m.merge(pd.DataFrame(rows), on="image_id")
    q["error"] = q.pred != q.label
    L = ["# What is special about POSTER++'s errors? (test set)", "",
         f"{int(q.error.sum())} errors of {len(q)} images ({100 * q.error.mean():.1f}%).", "",
         "| measure | correct (median) | errors (median) | error rate by quartile (low to high) |", "|---|---|---|---|"]
    for c in ["brightness", "contrast", "sharpness", "head_turn"]:
        qq = pd.qcut(q[c].rank(method="first"), 4, labels=False)
        by = q.groupby(qq).error.mean().mul(100).round(1).tolist()
        L.append(f"| {c} | {q.loc[~q.error, c].median():.2f} | {q.loc[q.error, c].median():.2f} | "
                 f"{' / '.join(f'{v}%' for v in by)} |")
    L.append(f"| face found by MediaPipe | {100 * q.loc[~q.error, 'face_found'].mean():.1f}% | "
             f"{100 * q.loc[q.error, 'face_found'].mean():.1f}% | error rate {100 * q.loc[~q.face_found, 'error'].mean():.1f}% "
             f"when not found vs {100 * q.loc[q.face_found, 'error'].mean():.1f}% when found |")
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    err = q[q.error].sort_values(["label", "pred"])
    per, cols, s = 60, 10, 100
    for k in range(0, len(err), per):
        part = err.iloc[k:k + per]
        rows_n = int(np.ceil(len(part) / cols))
        sheet = Image.new("RGB", (cols * s, rows_n * (s + 26)), "white")
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(part.itertuples(index=False)):
            x, y = (i % cols) * s, (i // cols) * (s + 26)
            sheet.paste(Image.open(r.path).convert("RGB").resize((s, s)), (x, y))
            d.text((x + 2, y + s + 1), f"T {CLASSES[r.label][:5]}", fill="black")
            d.text((x + 2, y + s + 12), f"P {CLASSES[r.pred][:5]} V {CLASSES[r.vlm][:5]}", fill="black")
        sheet.save(os.path.join(a.out, "figures", f"poster_errors_sheet_{k // per + 1}.png"))
    L += ["", f"Contact sheets of all errors: figures/poster_errors_sheet_*.png (T = annotated label, P = POSTER++, "
          f"V = VLM)."]
    q.to_csv(os.path.join(a.out, "results", "hard_images_test.csv"), index=False)
    open(os.path.join(a.out, "results", "hard_images.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
