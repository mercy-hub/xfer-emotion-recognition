"""Step 5: experiment E1, each source on its own. Fills the single-source rows of Table 16.

Reads predictions/<source>_<split>.csv for whichever of V, A, B, C exist, and writes to results/:
  e1_<split>.csv / .md        accuracy and macro-F1 with 95% bootstrap intervals, ECE, per-class recall
  e1_mcnemar_<split>.csv      pairwise McNemar tests between sources (Equation 4.1)
  confusion_<source>_<split>.png   row-normalised confusion matrices (greyscale)
Usage:
  python scripts/05_evaluate_e1.py --out /kaggle/working/outputs --split test
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xfer.io import load_manifest, pred_path, read_predictions
from xfer.labels import CLASSES, prob_columns
from xfer.metrics import accuracy, bootstrap_ci, confusion, ece, macro_f1, mcnemar, per_class_recall

NAMES = {"V": "V: LLaVA-1.5 (zero-shot)", "A": "A: CLIP (zero-shot)", "B": "B: POSTER++", "C": "C: HOG/LBP + SVM",
         "Vcal": "V with prior recalibration", "Acal": "A with prior recalibration",
         "Bcal": "B with prior recalibration", "Ccal": "C with prior recalibration"}


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels([c[:4] for c in CLASSES], fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > 0.55 else "black")
    ax.set_title(title, fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--sources", default="V,A,B,C")
    ap.add_argument("--tag", default="e1", help="prefix of the result files")
    a = ap.parse_args()
    res = os.path.join(a.out, "results"); os.makedirs(res, exist_ok=True)
    m = load_manifest(a.out)
    expected = set(m[m.split == a.split].image_id.astype(str))

    preds, rows = {}, []
    for s in a.sources.split(","):
        p = pred_path(a.out, s, a.split)
        if not os.path.exists(p):
            print(f"[E1] {s}: no predictions yet ({p})"); continue
        df = read_predictions(p)
        df["image_id"] = df.image_id.astype(str)
        missing = len(expected - set(df.image_id))
        if missing:
            print(f"[E1] {s}: {missing} of {len(expected)} images not predicted yet; reporting on the rest")
        df = df[df.image_id.isin(expected)].sort_values("image_id")
        preds[s] = df
        y, P = df.label.values, df[prob_columns()].values
        yhat = P.argmax(1)
        acc = bootstrap_ci(accuracy, y, yhat, a.B)
        f1 = bootstrap_ci(macro_f1, y, yhat, a.B)
        rec = per_class_recall(y, yhat)
        rows.append(dict(source=NAMES.get(s, s), n=len(y), accuracy=acc[0], acc_low=acc[1], acc_high=acc[2],
                         macro_f1=f1[0], f1_low=f1[1], f1_high=f1[2], ece=ece(y, P),
                         **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}))
        plot_confusion(confusion(y, yhat, normalize="true"), NAMES.get(s, s), os.path.join(res, f"confusion_{s}_{a.split}.png"))

    if not rows:
        print("nothing to evaluate"); return
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(res, f"{a.tag}_{a.split}.csv"), index=False)
    with open(os.path.join(res, f"{a.tag}_{a.split}.md"), "w") as fh:
        fh.write(f"| Source | n | Accuracy (95% CI) | Macro-F1 (95% CI) | ECE |\n|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['source']} | {r['n']} | {r['accuracy']:.4f} ({r['acc_low']:.4f}-{r['acc_high']:.4f}) | "
                     f"{r['macro_f1']:.4f} ({r['f1_low']:.4f}-{r['f1_high']:.4f}) | {r['ece']:.4f} |\n")
    print(open(os.path.join(res, f"{a.tag}_{a.split}.md")).read())

    mc = []
    for s1, s2 in itertools.combinations(preds, 2):
        j = preds[s1].merge(preds[s2], on="image_id", suffixes=("_1", "_2"))
        r = mcnemar(j.label_1.values, j.pred_1.values, j.pred_2.values)
        mc.append(dict(system_1=s1, system_2=s2, n=len(j), **r))
    if mc:
        pd.DataFrame(mc).to_csv(os.path.join(res, f"{a.tag}_mcnemar_{a.split}.csv"), index=False)
        print(pd.DataFrame(mc).to_string(index=False))


if __name__ == "__main__":
    main()
