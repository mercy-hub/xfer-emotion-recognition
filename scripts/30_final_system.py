"""Fit and store the final system of section 4.7, so that the demonstration app can use it on any face.

Experts (in this order): POSTER++ (B), CLIP fine-tuned (A2), InternVL3-8B (V2), HOG/LBP + SVM (C), CLIP zero-shot (A).
Combiner: the average of a CatBoost classifier (five seeds) and a weighted average of log-probabilities, both trained on
the validation split only. Decision thresholds tau and delta: chosen on out-of-fold validation probabilities for a
selective risk of at most 5 per cent. Also stored: the action-unit thresholds (75th training percentile).
Writes xfer/final/: catboost_s{0..4}.cbm, final.json. Prints the test result once as a check.
Usage:  python scripts/30_final_system.py --out OUT
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from xfer.labels import CLASSES, K, prob_columns
from xfer.reasoner import au_thresholds, aus

EXPERTS = ["B", "A2", "V2", "C", "A"]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(HERE, "xfer", "final")
CB = dict(loss_function="MultiClass", iterations=300, learning_rate=0.05, depth=4, l2_leaf_reg=5, verbose=0)
CONF = [("fear", "surprise"), ("anger", "disgust"), ("sadness", "neutral")]


def logp(out, split):
    d = {s: pd.read_csv(os.path.join(out, "predictions", f"{s}_{split}.csv")).set_index("image_id") for s in EXPERTS}
    ids = d["B"].index
    L = np.hstack([np.log(np.clip(d[s].loc[ids, prob_columns()].values, 1e-6, 1)) for s in EXPERTS])
    return L, d["B"].label.values.astype(int)


def wa_fit(L, y):
    k = L.shape[1] // K
    w = np.zeros(k); w[0] = 1
    for _ in range(3):
        for i in range(1, k):
            w[i] = max(np.linspace(0, 1.5, 31), key=lambda c: (sum((c if j == i else w[j]) * L[:, K * j:K * j + K]
                                                                    for j in range(k)).argmax(1) == y).mean())
    return w


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def combine(models, w, L):
    """The final combiner: 0.5 x CatBoost (mean of seeds) + 0.5 x softmax of the weighted log-probabilities."""
    pc = np.mean([m.predict_proba(L) for m in models], 0)
    pw = softmax(sum(w[j] * L[:, K * j:K * j + K] for j in range(len(w))))
    return 0.5 * pc + 0.5 * pw


def decide(P, tau, delta, conf):
    o = np.argsort(-P, 1)
    c1, c2 = o[:, 0], o[:, 1]
    r = np.arange(len(P))
    p1, p2 = P[r, c1], P[r, c2]
    lab = (p1 >= tau) & (p1 - p2 >= delta)
    pair = np.array([frozenset((a, b)) in conf for a, b in zip(c1, c2)])
    comp = ~lab & pair & (p1 + p2 >= tau)
    return lab, comp, c1, c2


def main():
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Lv, yv = logp(a.out, "val")
    conf = {frozenset((CLASSES.index(x), CLASSES.index(y))) for x, y in CONF}
    # out-of-fold validation probabilities, for the thresholds
    Pv = np.zeros((len(yv), K))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Lv, yv):
        ms = [CatBoostClassifier(**CB, random_seed=s).fit(Lv[tr], yv[tr]) for s in range(5)]
        Pv[te] = combine(ms, wa_fit(Lv[tr], yv[tr]), Lv[te])
    best = (0, 0.5, 0.2)
    for tau in np.linspace(0.2, 0.95, 31):
        for delta in np.linspace(0, 0.6, 25):
            lab, comp, c1, c2 = decide(Pv, tau, delta, conf)
            acc = lab | comp
            err = (lab & (c1 != yv)) | (comp & (c1 != yv) & (c2 != yv))
            if acc.any() and err[acc].mean() <= 0.05 and acc.mean() > best[0]:
                best = (acc.mean(), tau, delta)
    models = [CatBoostClassifier(**CB, random_seed=s).fit(Lv, yv) for s in range(5)]
    w = wa_fit(Lv, yv)
    os.makedirs(FINAL, exist_ok=True)
    for s, m in enumerate(models):
        m.save_model(os.path.join(FINAL, f"catboost_s{s}.cbm"))
    thr, meas = au_thresholds(a.out)
    json.dump(dict(experts=EXPERTS, wa_weights=w.tolist(), tau=float(best[1]), delta=float(best[2]),
                   confusable=CONF, au_names=aus(), au_thr=[float(x) for x in thr], au_measurable=[bool(x) for x in meas],
                   note="combiner = 0.5 * mean(CatBoost seeds) + 0.5 * softmax(sum_j w_j log p_j); experts in this order"),
              open(os.path.join(FINAL, "final.json"), "w"), indent=1)
    Lt, yt = logp(a.out, "test")
    P = combine(models, w, Lt)
    print(f"saved to {FINAL}; tau {best[1]:.3f} delta {best[2]:.3f}; weights {np.round(w, 2).tolist()}")
    print(f"check: test accuracy {100 * (P.argmax(1) == yt).mean():.2f}%")


if __name__ == "__main__":
    main()
