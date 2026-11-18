"""Stage 2 'weight and aggregate' as a learned meta-learner: CatBoost over the experts' class probabilities.

Replaces the hand-tuned weighted average (script 21) with a CatBoost classifier trained on the VALIDATION split only
(every expert's validation predictions are out-of-sample), then applied once to the test split. Reports accuracy,
macro-F1, McNemar against POSTER++ (B) and against the weighted average, stability over seeds, and per-expert
importance (mean |SHAP| of each expert's seven inputs), which is the explanation of the weighting step.
Usage:  python scripts/26_catboost_fusion.py --out OUT [--sources B,V2,C,A,U,V2s]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from xfer.labels import CLASSES

P = [f"p_{c}" for c in CLASSES]


def load(out, s, split):
    f = os.path.join(out, "predictions", f"{s}_{split}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d["image_id"] = d.image_id.astype(str)
    return d.set_index("image_id")


def mcnemar(p, ref, y):
    a, b = int(((p == y) & (ref != y)).sum()), int(((p != y) & (ref == y)).sum())
    return a, b, (binomtest(a, a + b).pvalue if a + b else 1.0)


def main():
    from catboost import CatBoostClassifier
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="B,V2,C,A,U,V2s")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    data, used = {}, []
    for s in a.sources.split(","):
        v, t = load(a.out, s, "val"), load(a.out, s, "test")
        if v is None or t is None:
            print(f"[skip] {s}: no val/test predictions")
            continue
        data[s] = (v, t)
        used.append(s)
    assert "B" in used, "POSTER++ (B) predictions are required"
    ids = {sp: sorted(set.intersection(*[set(data[s][k].index) for s in used])) for k, sp in enumerate(["val", "test"])}
    X = {sp: np.hstack([np.log(np.clip(data[s][k].loc[ids[sp], P].values, 1e-6, 1)) for s in used])
         for k, sp in enumerate(["val", "test"])}
    y = {sp: data["B"][k].loc[ids[sp], "label"].values.astype(int) for k, sp in enumerate(["val", "test"])}
    post = data["B"][1].loc[ids["test"], P].values.argmax(1)
    yt = y["test"]
    # weighted average baseline (weights tuned on val by coordinate search)
    w = np.zeros(len(used)); w[used.index("B")] = 1.0
    Lv = [X["val"][:, 7 * i:7 * i + 7] for i in range(len(used))]
    Lt = [X["test"][:, 7 * i:7 * i + 7] for i in range(len(used))]
    for _ in range(3):
        for i, s in enumerate(used):
            if s == "B":
                continue
            best = max(np.linspace(0, 1, 21), key=lambda c: ((sum(w[j] * Lv[j] for j in range(len(used)) if j != i)
                                                             + c * Lv[i]).argmax(1) == y["val"]).mean())
            w[i] = best
    wa = sum(w[i] * Lt[i] for i in range(len(used))).argmax(1)
    prm = dict(loss_function="MultiClass", iterations=300, learning_rate=0.05, depth=4, l2_leaf_reg=5, verbose=0)
    preds, models = [], []
    for seed in range(a.seeds):
        m = CatBoostClassifier(**prm, random_seed=seed).fit(X["val"], y["val"])
        preds.append(m.predict_proba(X["test"]))
        models.append(m)
    pc = np.mean(preds, 0).argmax(1)
    accs = [(p.argmax(1) == yt).mean() for p in preds]
    from catboost import Pool
    sh = np.mean([np.abs(m.get_feature_importance(Pool(X["test"], yt), type="ShapValues")[:, :, :-1]).mean((0, 1))
                  for m in models[:1]], 0)
    imp = {s: sh[7 * i:7 * i + 7].sum() for i, s in enumerate(used)}
    tot = sum(imp.values())
    L = ["# CatBoost meta-learner (stage 2 weight and aggregate)", "",
         f"Experts used: {', '.join(used)}. Trained on validation ({len(ids['val'])} images), tested once on "
         f"{len(yt)} test images.", "",
         "| system | accuracy | macro-F1 | vs POSTER++ fixed/broken (McNemar p) |", "|---|---|---|---|"]
    for name, p in [("POSTER++ alone", post), ("weighted average (val-tuned weights "
                                                 + ", ".join(f"{s} {w[i]:.2f}" for i, s in enumerate(used)) + ")", wa),
                    (f"CatBoost meta-learner ({a.seeds} seeds averaged)", pc)]:
        f, b, pv = mcnemar(p, post, yt)
        L.append(f"| {name} | {100 * (p == yt).mean():.2f}% | {f1_score(yt, p, average='macro'):.3f} | "
                 f"{f}/{b} (p = {pv:.2g}) |")
    f, b, pv = mcnemar(pc, wa, yt)
    L += ["", f"CatBoost vs weighted average: {f} fixed / {b} broken, McNemar p = {pv:.2g}.",
          f"Single-seed accuracy range: {100 * min(accs):.2f}% to {100 * max(accs):.2f}%.", "",
          "Share of the meta-learner's decisions explained by each expert (mean |SHAP|):", ""]
    L += [f"- {s}: {100 * imp[s] / tot:.1f}%" for s in sorted(imp, key=imp.get, reverse=True)]
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    open(os.path.join(a.out, "results", "catboost_fusion.md"), "w").write("\n".join(L))
    pd.DataFrame({"image_id": ids["test"], "label": yt, "pred": pc}).to_csv(
        os.path.join(a.out, "predictions", "CB_test.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
