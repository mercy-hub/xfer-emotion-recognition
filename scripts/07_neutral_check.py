"""Step 7: does LLaVA-1.5 ever say "neutral"? A check on the scoring of Equation 3.2. GPU.

E1 found that LLaVA-1.5 never predicted neutral. This lets the model answer freely (greedy decoding, a few tokens)
on N validation images per class, and compares its own words with the first-token prediction. If the free answers
say "Neutral" for neutral faces while the first-token score does not, the scoring is at fault; if the free answers
avoid "Neutral" too, the bias belongs to the model and is left for the compensator.
Writes results/V_free_answers.csv and results/V_free_vs_scored.csv.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from xfer.io import batches, load_manifest, pred_path, read_predictions
from xfer.labels import CLASSES, NAME_ALIASES
from xfer.sources.llava_zs import QUESTION, LlavaZeroShot


def first_class_word(text):
    for w in re.findall(r"[A-Za-z]+", text.lower()):
        if w in NAME_ALIASES:
            return NAME_ALIASES[w]
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_class", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    m = load_manifest(a.out)
    val = m[m.split == "val"]
    sub = val.groupby("label", group_keys=False).apply(lambda g: g.sample(min(a.per_class, len(g)), random_state=0))
    model = LlavaZeroShot()
    rows = []
    for b in batches(sub, a.batch):
        txt = model.generate_rationales(b.path.tolist(), max_new_tokens=6, question=QUESTION)
        for iid, lab, t in zip(b.image_id, b.label, txt):
            rows.append(dict(image_id=str(iid), label=CLASSES[int(lab)], free_text=t, free_class=first_class_word(t)))
    f = pd.DataFrame(rows)
    sc = read_predictions(pred_path(a.out, "V", "val"))[["image_id", "pred"]]
    sc["image_id"] = sc.image_id.astype(str)
    f = f.merge(sc, on="image_id", how="left")
    f["scored_class"] = [CLASSES[int(p)] if p == p else "missing" for p in f.pred]
    f.drop(columns="pred").to_csv(os.path.join(a.out, "results", "V_free_answers.csv"), index=False)
    t = pd.crosstab(f.label, f.free_class).reindex(CLASSES)
    print("Free answers (rows: true class; columns: the class word the model wrote):")
    print(t.fillna(0).astype(int).to_string())
    agree = (f.free_class == f.scored_class).mean()
    print(f"\nFree answer agrees with the first-token prediction on {agree:.1%} of images")
    print(f"'neutral' written for {int((f.free_class == 'neutral').sum())} of {len(f)} images "
          f"({int(((f.free_class == 'neutral') & (f.label == 'neutral')).sum())} of them truly neutral)")
    t.to_csv(os.path.join(a.out, "results", "V_free_vs_scored.csv"))


if __name__ == "__main__":
    main()
