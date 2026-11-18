"""Step 1 check: how good is the new VLM, and how does it compare with LLaVA-1.5 (the old source V)?

Writes results/vlm_compare_<split>.md and prints it. Reports, for each VLM present (V = LLaVA, <tag> = new):
accuracy and macro-F1 with 95% bootstrap intervals, ECE, per-class recall, McNemar against LLaVA, and for the
structured descriptions: share of replies parsed, share with all five face fields, and agreement between the
described emotion and the probability-based prediction.
Usage:
  python scripts/19_eval_vlm.py --out OUT --split test --tag V2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from xfer.io import pred_path, read_predictions
from xfer.labels import CLASSES, prob_columns
from xfer.metrics import accuracy, bootstrap_ci, ece, macro_f1, mcnemar, per_class_recall


def summary(df):
    y, p = df.label.values, df.pred.values
    a = bootstrap_ci(accuracy, y, p)
    f = bootstrap_ci(macro_f1, y, p)
    rec = per_class_recall(y, p)
    return {"n": len(df), "accuracy": f"{100 * a[0]:.1f} ({100 * a[1]:.1f}-{100 * a[2]:.1f})",
            "macro_F1": f"{f[0]:.3f} ({f[1]:.3f}-{f[2]:.3f})", "ECE": f"{ece(y, df[prob_columns()].values):.3f}",
            **{f"recall_{c[:4]}": f"{r:.2f}" for c, r in zip(CLASSES, rec)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="V2")
    a = ap.parse_args()

    rows, frames = [], {}
    for src, name in [("V", "LLaVA-1.5-7B (old V)"), (a.tag, f"New VLM ({a.tag})")]:
        path = pred_path(a.out, src, a.split)
        if os.path.exists(path):
            frames[src] = read_predictions(path)
            rows.append({"system": name, **summary(frames[src])})
    if not rows:
        sys.exit("no prediction files found")
    lines = [f"# VLM comparison on {a.split}", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    if "V" in frames and a.tag in frames:
        j = frames["V"].merge(frames[a.tag], on="image_id", suffixes=("_old", "_new"))
        t = mcnemar(j.label_old.values, j.pred_old.values, j.pred_new.values)
        lines += [f"McNemar, LLaVA vs new VLM on {len(j)} shared images: {t}", ""]

    dpath = os.path.join(a.out, "descriptions", f"{a.tag}_{a.split}_describe.csv")
    if os.path.exists(dpath):
        d = pd.read_csv(dpath).drop_duplicates("image_id", keep="last")
        ok = d.emotion_idx >= 0
        lines += [f"Structured descriptions: {len(d)} images; emotion parsed {100 * ok.mean():.1f}%; "
                  f"all five face fields {100 * (d.parsed_fields == 5).mean():.1f}%; "
                  f"described-emotion accuracy {100 * (d.emotion_idx[ok] == d.label[ok]).mean():.1f}%"]
        if a.tag in frames:
            j = d.merge(frames[a.tag][["image_id", "pred"]], on="image_id")
            lines.append(f"Agreement of described emotion with probability prediction: "
                         f"{100 * (j.emotion_idx == j.pred).mean():.1f}%")
        lines += ["", "Examples:"] + [f"- [{CLASSES[r.label]}] {r.text}" for r in d.head(5).itertuples()]

    text = "\n".join(lines)
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    open(os.path.join(a.out, "results", f"vlm_compare_{a.split}.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
