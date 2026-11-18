"""Step 10: surrogate decision tree for the base model V (section 3.4, Equation 3.4). CPU, seconds.

A shallow tree is trained on the training split to reproduce V's predicted labels (not the true labels) from
interpretable inputs: the geometric measures of source D and, where they exist, V's region SHAP values. Its fidelity
(the share of images on which the tree and V agree) is reported on the validation and test splits for several depths,
and the rules of the chosen depth are written out. Rules are trusted only as far as their fidelity.
Writes results/surrogate_fidelity.csv, results/surrogate_rules_depth<d>.txt and models/surrogate_tree.joblib.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from xfer.io import pred_path, read_predictions
from xfer.labels import CLASSES
from xfer.regions import GEOMETRY, REGIONS


def table(out, split, with_shap):
    g = pd.read_csv(os.path.join(out, "results", "D_geometry.csv"))
    g["image_id"] = g.image_id.astype(str)
    g = g[(g.split == split) & g.found]
    v = read_predictions(pred_path(out, "V", split))[["image_id", "pred"]]
    v["image_id"] = v.image_id.astype(str)
    df = g.merge(v, on="image_id")
    cols = list(GEOMETRY)
    if with_shap:
        e = pd.read_csv(os.path.join(out, "explanations", f"V_{split}.csv"))
        e["image_id"] = e.image_id.astype(str)
        df = df.merge(e[["image_id"] + [f"shap_{r}" for r in REGIONS]], on="image_id")
        cols += [f"shap_{r}" for r in REGIONS]
    return df, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="2,3,4,5,6")
    ap.add_argument("--choose", type=int, default=4, help="depth whose rules are written out")
    ap.add_argument("--with_shap", action="store_true", help="also use V's region SHAP values (needs step 9 merged)")
    a = ap.parse_args()
    tr, cols = table(a.out, "train", a.with_shap)
    ev = {s: table(a.out, s, a.with_shap)[0] for s in ("val", "test")
          if os.path.exists(pred_path(a.out, "V", s))}
    print(f"training the surrogate on {len(tr)} training images with inputs: {', '.join(cols)}")
    rows = []
    for depth in [int(x) for x in a.depths.split(",")]:
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=0)
        t.fit(tr[cols].values, tr.pred.values)
        r = dict(depth=depth, leaves=t.get_n_leaves(), fidelity_train=float((t.predict(tr[cols].values) == tr.pred).mean()))
        for s, df in ev.items():
            r[f"fidelity_{s}"] = float((t.predict(df[cols].values) == df.pred).mean())
            r[f"tree_accuracy_{s}"] = float((t.predict(df[cols].values) == df.label).mean())
        rows.append(r)
        if depth == a.choose:
            joblib.dump(dict(tree=t, columns=cols), os.path.join(a.out, "models", "surrogate_tree.joblib"))
            names = [c.replace("shap_", "SHAP of ") for c in cols]
            txt = export_text(t, feature_names=names, class_names=[CLASSES[i] for i in t.classes_], decimals=3)
            with open(os.path.join(a.out, "results", f"surrogate_rules_depth{depth}.txt"), "w") as fh:
                fh.write(txt)
    f = pd.DataFrame(rows)
    f.to_csv(os.path.join(a.out, "results", "surrogate_fidelity.csv"), index=False)
    print(f.round(4).to_string(index=False))
    print(f"rules of the depth-{a.choose} tree: results/surrogate_rules_depth{a.choose}.txt")


if __name__ == "__main__":
    main()
