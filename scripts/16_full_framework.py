"""Step 16: the full framework (E2 final row and E3 repeated): the reasoner with source V replaced by its compensated
output (step 12 or 15). CPU, a few minutes.

V's probabilities are replaced by the compensated prediction (one-hot) on the validation and test splits; A, B, C and
D are unchanged. Weights, lambda, tau and delta are fitted again on validation exactly as in step 11 (a fresh random
generator with the same seed), and the test split is used once.
Writes results/e2_full_test.md and results/full_framework.json.

    python scripts/16_full_framework.py --out OUT --tag Vcomp-pred+expl+face
"""
import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from xfer import ontology as ONT
from xfer.metrics import accuracy, bootstrap_ci, macro_f1, mcnemar
from xfer.reasoner import (SOURCES, au_thresholds, brackets, consistency, decide, load, score, selective, translate)

spec = importlib.util.spec_from_file_location("r11", os.path.join(ROOT, "scripts", "11_reasoner.py"))
r11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="Vcomp-pred+expl+face", help="prediction files of the compensated V")
    ap.add_argument("--n_search", type=int, default=3000)
    ap.add_argument("--risk", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--owl", default=os.path.join(ROOT, "ontology", "xfer_emonto.owl"))
    a = ap.parse_args()
    if a.owl:
        ONT.use_owl(a.owl)
    rng = np.random.default_rng(a.seed)
    thr, meas = au_thresholds(a.out)
    ev = {s: load(a.out, s, {"V": a.tag}) for s in ("val", "test")}
    raw = {s: load(a.out, s) for s in ("val", "test")}
    tr = {(s, m): translate(ev[s], thr, meas, m) for s in ev for m in ("soft", "hard")}
    trv = {m: tr[("val", m)] for m in ("soft", "hard")}
    f0, p0 = r11.search(ev["val"], trv, a.n_search, rng, probs_only=True)
    f, p = r11.search(ev["val"], trv, a.n_search, rng, start=p0)
    yv, yt = ev["val"]["y"], ev["test"]["y"]

    def scores(split, prm, lam=None):
        return score(brackets(ev[split], tr[(split, prm["mode"])], prm["lam"] if lam is None else lam),
                     np.array(prm["alpha"]))
    scv, sct = scores("val", p), scores("test", p)
    full, probs_only = sct.argmax(1), scores("test", p0, 1.0).argmax(1)
    pb = raw["test"]["P_B"].argmax(1)
    w = np.array([macro_f1(yv, raw["val"][f"P_{s}"].argmax(1)) for s in SOURCES]); w = w / w.sum()
    cwa = sum(wi * raw["test"][f"P_{s}"] for wi, s in zip(w, SOURCES)).argmax(1)
    avg = sum(raw["test"][f"P_{s}"] for s in SOURCES)
    top2 = np.argsort(-avg, 1)[:, :2]
    subset = np.array([frozenset(t) in ONT.confusable_index() for t in top2])

    rows = {}
    for name, pr in (("V with compensator", ev["test"]["P_V"].argmax(1)), ("Full framework, probabilities only", probs_only),
                     ("Full framework", full)):
        acc, f1 = bootstrap_ci(accuracy, yt, pr, 1000), bootstrap_ci(macro_f1, yt, pr, 1000)
        rows[name] = dict(accuracy=acc, macro_f1=f1, confusable=accuracy(yt[subset], pr[subset]),
                          vs_B=mcnemar(yt, pr, pb), vs_weighted_average=mcnemar(yt, pr, cwa))
    bs = np.random.default_rng(1)
    dd, da = [], []
    for _ in range(2000):
        i = bs.integers(0, len(yt), len(yt))
        dd.append(macro_f1(yt[i], full[i]) - macro_f1(yt[i], pb[i])); da.append(accuracy(yt[i], full[i]) - accuracy(yt[i], pb[i]))
    diff = dict(macro_f1=[macro_f1(yt, full) - macro_f1(yt, pb), *np.percentile(dd, [2.5, 97.5])],
                accuracy=[accuracy(yt, full) - accuracy(yt, pb), *np.percentile(da, [2.5, 97.5])])

    best = None
    for tau in np.linspace(0.2, 0.9, 36):
        for delta in np.linspace(0, 0.6, 31):
            k, c1, c2 = decide(scv, tau, delta)
            cov, risk, _, _ = selective(yv, k, c1, c2)
            if risk <= a.risk and (best is None or cov > best[0]):
                best = (cov, tau, delta)
    k, c1, c2 = decide(sct, best[1], best[2])
    cov, risk, lab, bl = selective(yt, k, c1, c2)
    cons = consistency(brackets(ev["test"], tr[("test", p["mode"])], p["lam"]), c1)
    margin = np.sort(sct, 1)[:, -1] - np.sort(sct, 1)[:, -2]
    PB = raw["test"]["P_B"]
    e3 = dict(tau=best[1], delta=best[2], val_coverage=best[0], coverage=cov, selective_risk=risk, single=lab, blends=bl,
              abstentions=1 - cov, blend_contains_truth=float(((c1 == yt) | (c2 == yt))[k == 1].mean()),
              consistency_accepted=float(cons[k < 2].mean()), consistency_abstained=float(cons[k == 2].mean()),
              B_risk_same_coverage=r11.risk_at(PB.max(1), PB.argmax(1) != yt, cov),
              margin_risk_same_coverage=r11.risk_at(margin, c1 != yt, cov))
    res = os.path.join(a.out, "results"); os.makedirs(res, exist_ok=True)
    out = dict(tag=a.tag, params=p, val_macro_f1=f, probs_only_params=p0, probs_only_val_macro_f1=f0, e2=rows,
               diff_vs_B=diff, e3=e3, confusable_subset=int(subset.sum()))
    with open(os.path.join(res, "full_framework.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    with open(os.path.join(res, "e2_full_test.md"), "w") as fh:
        fh.write(f"V replaced by {a.tag}\n\n| System | Accuracy (95% CI) | Macro-F1 (95% CI) | Confusable acc. | vs B n01/n10 p | vs weighted avg n01/n10 p |\n|---|---|---|---|---|---|\n")
        for name, r in rows.items():
            fh.write(f"| {name} | {r['accuracy'][0]:.4f} ({r['accuracy'][1]:.4f}-{r['accuracy'][2]:.4f}) | "
                     f"{r['macro_f1'][0]:.4f} ({r['macro_f1'][1]:.4f}-{r['macro_f1'][2]:.4f}) | {r['confusable']:.4f} | "
                     f"{r['vs_B']['n01']}/{r['vs_B']['n10']} {r['vs_B']['p_value']:.3g} | "
                     f"{r['vs_weighted_average']['n01']}/{r['vs_weighted_average']['n10']} {r['vs_weighted_average']['p_value']:.3g} |\n")
        fh.write(f"\nPaired bootstrap, full framework minus B: macro-F1 {diff['macro_f1']}, accuracy {diff['accuracy']}\n")
        fh.write(f"\nE3: {json.dumps(e3, default=float)}\n")
    print(open(os.path.join(res, "e2_full_test.md")).read())


if __name__ == "__main__":
    main()
